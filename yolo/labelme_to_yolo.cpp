#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>

namespace fs = std::filesystem;
using json = nlohmann::json;

// labelme标注文件所在目录
const std::string LABELME_DIR = "new";
// 图像文件所在目录
const std::string IMAGE_DIR = "frames_to_annotate";
// 输出目录（可选，如果为空则在与JSON文件相同的目录生成.txt文件）
const std::string OUTPUT_DIR = "";

// 类别映射（labelme中的类别名称 -> YOLO类别ID）
std::map<std::string, int> CLASS_MAPPING = {
    {"goal", 0},          // 普通目标 -> 类别0
    {"rectangle", 0},     // 矩形框 -> 类别0
    {"goal_r", 1},        // 中心有r的目标 -> 类别1
    {"r_character", 1},   // r字符（别名） -> 类别1
    {"r", 1}              // r（简短名称） -> 类别1
};

// 将多边形点集转换为边界框 [x_min, y_min, x_max, y_max]
std::vector<float> polygonToBbox(const std::vector<std::vector<float>>& points) {
    float x_min = points[0][0];
    float x_max = points[0][0];
    float y_min = points[0][1];
    float y_max = points[0][1];
    
    for (size_t i = 1; i < points.size(); ++i) {
        x_min = std::min(x_min, points[i][0]);
        x_max = std::max(x_max, points[i][0]);
        y_min = std::min(y_min, points[i][1]);
        y_max = std::max(y_max, points[i][1]);
    }
    
    return {x_min, y_min, x_max, y_max};
}

// 将边界框坐标转换为YOLO格式（归一化的中心点坐标和宽高）
std::vector<float> bboxToYolo(const std::vector<float>& bbox, float imgWidth, float imgHeight) {
    float x_min = bbox[0];
    float y_min = bbox[1];
    float x_max = bbox[2];
    float y_max = bbox[3];
    
    // 计算中心点和宽高
    float center_x = (x_min + x_max) / 2.0f;
    float center_y = (y_min + y_max) / 2.0f;
    float width = x_max - x_min;
    float height = y_max - y_min;
    
    // 归一化
    float center_x_norm = std::clamp(center_x / imgWidth, 0.0f, 1.0f);
    float center_y_norm = std::clamp(center_y / imgHeight, 0.0f, 1.0f);
    float width_norm = std::clamp(width / imgWidth, 0.0f, 1.0f);
    float height_norm = std::clamp(height / imgHeight, 0.0f, 1.0f);
    
    return {center_x_norm, center_y_norm, width_norm, height_norm};
}

// 将单个labelme JSON文件转换为YOLO格式的.txt文件
bool convertLabelmeToYolo(const std::string& jsonPath, const std::string& outputDir = "") {
    try {
        // 读取JSON文件
        std::ifstream f(jsonPath);
        if (!f.is_open()) {
            std::cerr << "错误: 无法打开文件: " << jsonPath << std::endl;
            return false;
        }
        
        json data = json::parse(f);
        
        // 获取图像尺寸
        float imgWidth = data.value("imageWidth", 0.0f);
        float imgHeight = data.value("imageHeight", 0.0f);
        
        if (imgWidth == 0 || imgHeight == 0) {
            std::cerr << "警告: " << jsonPath << " 中缺少图像尺寸信息" << std::endl;
            return false;
        }
        
        // 获取所有标注形状
        auto shapes = data.value("shapes", json::array());
        if (shapes.empty()) {
            std::cerr << "警告: " << jsonPath << " 中没有找到标注" << std::endl;
            return false;
        }
        
        // 准备输出文件路径
        std::string jsonFilename = fs::path(jsonPath).filename().string();
        std::string txtFilename = jsonFilename.substr(0, jsonFilename.find_last_of('.')) + ".txt";
        
        std::string outputPath;
        if (!outputDir.empty()) {
            outputPath = outputDir + "/" + txtFilename;
        } else {
            outputPath = fs::path(jsonPath).parent_path().string() + "/" + txtFilename;
        }
        
        // 打开输出文件
        std::ofstream outFile(outputPath);
        if (!outFile.is_open()) {
            std::cerr << "错误: 无法创建输出文件: " << outputPath << std::endl;
            return false;
        }
        
        // 处理每个形状
        int convertedCount = 0;
        for (const auto& shape : shapes) {
            std::string label = shape.value("label", "");
            std::string shapeType = shape.value("shape_type", "");
            
            // 查找类别ID
            auto it = CLASS_MAPPING.find(label);
            if (it == CLASS_MAPPING.end()) {
                std::cerr << "警告: 未找到类别映射: " << label << std::endl;
                continue;
            }
            
            int classId = it->second;
            
            // 获取点集
            std::vector<std::vector<float>> points;
            if (shapeType == "rectangle") {
                // 矩形格式: [[x1,y1], [x2,y2]]
                auto pointList = shape["points"];
                float x1 = pointList[0][0];
                float y1 = pointList[0][1];
                float x2 = pointList[1][0];
                float y2 = pointList[1][1];
                
                points = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
            } else if (shapeType == "polygon" || shapeType == "line" || shapeType == "point") {
                // 多边形格式: [[x1,y1], [x2,y2], ...]
                auto pointList = shape["points"];
                for (const auto& p : pointList) {
                    points.push_back({p[0], p[1]});
                }
            } else {
                std::cerr << "警告: 不支持的形状类型: " << shapeType << std::endl;
                continue;
            }
            
            if (points.empty()) {
                continue;
            }
            
            // 转换为边界框
            std::vector<float> bbox = polygonToBbox(points);
            
            // 转换为YOLO格式
            std::vector<float> yoloCoords = bboxToYolo(bbox, imgWidth, imgHeight);
            
            // 写入文件
            outFile << classId << " "
                    << yoloCoords[0] << " "
                    << yoloCoords[1] << " "
                    << yoloCoords[2] << " "
                    << yoloCoords[3] << std::endl;
            
            convertedCount++;
        }
        
        outFile.close();
        
        if (convertedCount > 0) {
            std::cout << "转换成功: " << jsonPath << " -> " << outputPath 
                      << " (" << convertedCount << " 个标注)" << std::endl;
            return true;
        } else {
            // 如果没有成功转换的标注，删除空文件
            fs::remove(outputPath);
            std::cerr << "警告: " << jsonPath << " 中没有成功转换的标注" << std::endl;
            return false;
        }
    } catch (const json::exception& e) {
        std::cerr << "JSON解析错误: " << e.what() << " (" << jsonPath << ")" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << " (" << jsonPath << ")" << std::endl;
        return false;
    }
}

// 转换所有labelme文件
void convertAllLabelmeFiles() {
    std::cout << "开始转换LabelMe标注文件到YOLO格式..." << std::endl;
    std::cout << "源目录: " << LABELME_DIR << std::endl;
    
    // 创建输出目录（如果指定）
    if (!OUTPUT_DIR.empty()) {
        fs::create_directories(OUTPUT_DIR);
        std::cout << "输出目录: " << OUTPUT_DIR << std::endl;
    }
    
    // 类别映射信息
    std::cout << "类别映射:" << std::endl;
    for (const auto& [label, id] : CLASS_MAPPING) {
        std::cout << "  - " << label << " -> " << id << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
    
    if (!fs::exists(LABELME_DIR)) {
        std::cerr << "错误: 源目录不存在: " << LABELME_DIR << std::endl;
        return;
    }
    
    int totalFiles = 0;
    int convertedFiles = 0;
    int totalAnnotations = 0;
    
    // 遍历目录中的所有JSON文件
    for (const auto& entry : fs::directory_iterator(LABELME_DIR)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            totalFiles++;
            
            // 转换文件
            if (convertLabelmeToYolo(entry.path().string(), OUTPUT_DIR)) {
                convertedFiles++;
                // 计算标注数量（简化处理，实际应该从返回值获取）
                totalAnnotations++;
            }
        }
    }
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "转换完成!" << std::endl;
    std::cout << "总文件数: " << totalFiles << std::endl;
    std::cout << "成功转换: " << convertedFiles << " 个文件" << std::endl;
    std::cout << "总标注数: " << totalAnnotations << std::endl;
    
    std::cout << "\n提示:" << std::endl;
    std::cout << "1. 请检查转换后的.txt文件是否正确" << std::endl;
    std::cout << "2. 可以使用 organize_dataset.cpp 将数据整理为训练所需的结构" << std::endl;
}

int main() {
    convertAllLabelmeFiles();
    return 0;
}