#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

// 标注好的图像和标注文件所在的目录
const std::string SOURCE_DIR = "frames_to_annotate";  // 或者你的标注目录

// 目标数据集目录
const std::string DATASET_DIR = "dataset";
const std::string TRAIN_IMAGES_DIR = DATASET_DIR + "/images/train";
const std::string VAL_IMAGES_DIR = DATASET_DIR + "/images/val";
const std::string TRAIN_LABELS_DIR = DATASET_DIR + "/labels/train";
const std::string VAL_LABELS_DIR = DATASET_DIR + "/labels/val";

// 训练/验证比例（80%训练，20%验证）
const float TRAIN_RATIO = 0.8f;

// 创建必要的目录结构
void createDirectories() {
    std::vector<std::string> dirs = {TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR};
    for (const auto& dirPath : dirs) {
        fs::create_directories(dirPath);
        std::cout << "创建目录: " << dirPath << std::endl;
    }
}

// 获取所有图像文件
std::vector<std::string> getImageFiles(const std::string& sourceDir) {
    std::vector<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp"};
    std::vector<std::string> imageFiles;
    
    if (!fs::exists(sourceDir)) {
        std::cerr << "错误: 源目录不存在: " << sourceDir << std::endl;
        return imageFiles;
    }
    
    for (const auto& entry : fs::directory_iterator(sourceDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            // 转换扩展名到小写进行比较
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end()) {
                imageFiles.push_back(entry.path().filename().string());
            }
        }
    }
    
    // 排序以确保一致性
    std::sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

// 整理数据集
void organizeDataset() {
    std::cout << "开始整理数据集..." << std::endl;
    std::cout << "源目录: " << SOURCE_DIR << std::endl;
    std::cout << "目标目录: " << DATASET_DIR << std::endl;
    
    // 创建目录结构
    createDirectories();
    
    // 获取所有图像文件
    std::vector<std::string> imageFiles = getImageFiles(SOURCE_DIR);
    
    if (imageFiles.empty()) {
        std::cerr << "错误: 在 " << SOURCE_DIR << " 中没有找到图像文件" << std::endl;
        std::cerr << "请确认:" << std::endl;
        std::cerr << "1. 已运行 extract_frames.cpp 提取帧" << std::endl;
        std::cerr << "2. 已使用 LabelImg 标注图像" << std::endl;
        return;
    }
    
    std::cout << "找到 " << imageFiles.size() << " 张图像" << std::endl;
    
    // 随机打乱（设置随机种子以便复现）
    std::mt19937 g(42);
    std::shuffle(imageFiles.begin(), imageFiles.end(), g);
    
    // 计算训练集和验证集的数量
    size_t trainCount = static_cast<size_t>(imageFiles.size() * TRAIN_RATIO);
    std::vector<std::string> trainFiles(imageFiles.begin(), imageFiles.begin() + trainCount);
    std::vector<std::string> valFiles(imageFiles.begin() + trainCount, imageFiles.end());
    
    std::cout << "训练集: " << trainFiles.size() << " 张图像" << std::endl;
    std::cout << "验证集: " << valFiles.size() << " 张图像" << std::endl;
    
    // 复制文件
    size_t copiedTrain = 0;
    size_t copiedVal = 0;
    
    // 复制训练集
    for (const auto& imgFile : trainFiles) {
        // 复制图像
        std::string srcImg = SOURCE_DIR + "/" + imgFile;
        std::string dstImg = TRAIN_IMAGES_DIR + "/" + imgFile;
        try {
            fs::copy_file(srcImg, dstImg, fs::copy_options::overwrite_existing);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "复制图像失败: " << e.what() << std::endl;
            continue;
        }
        
        // 复制对应的标注文件
        std::string baseName = imgFile.substr(0, imgFile.find_last_of('.'));
        std::string labelFile = baseName + ".txt";
        std::string srcLabel = SOURCE_DIR + "/" + labelFile;
        std::string dstLabel = TRAIN_LABELS_DIR + "/" + labelFile;
        
        if (fs::exists(srcLabel)) {
            try {
                fs::copy_file(srcLabel, dstLabel, fs::copy_options::overwrite_existing);
                copiedTrain++;
            } catch (const fs::filesystem_error& e) {
                std::cerr << "复制标注文件失败: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "警告: 未找到标注文件: " << srcLabel << std::endl;
        }
    }
    
    // 复制验证集
    for (const auto& imgFile : valFiles) {
        // 复制图像
        std::string srcImg = SOURCE_DIR + "/" + imgFile;
        std::string dstImg = VAL_IMAGES_DIR + "/" + imgFile;
        try {
            fs::copy_file(srcImg, dstImg, fs::copy_options::overwrite_existing);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "复制图像失败: " << e.what() << std::endl;
            continue;
        }
        
        // 复制对应的标注文件
        std::string baseName = imgFile.substr(0, imgFile.find_last_of('.'));
        std::string labelFile = baseName + ".txt";
        std::string srcLabel = SOURCE_DIR + "/" + labelFile;
        std::string dstLabel = VAL_LABELS_DIR + "/" + labelFile;
        
        if (fs::exists(srcLabel)) {
            try {
                fs::copy_file(srcLabel, dstLabel, fs::copy_options::overwrite_existing);
                copiedVal++;
            } catch (const fs::filesystem_error& e) {
                std::cerr << "复制标注文件失败: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "警告: 未找到标注文件: " << srcLabel << std::endl;
        }
    }
    
    std::cout << "\n数据集整理完成！" << std::endl;
    std::cout << "训练集复制: " << copiedTrain << " 个文件对" << std::endl;
    std::cout << "验证集复制: " << copiedVal << " 个文件对" << std::endl;
    std::cout << "\n数据集结构:" << std::endl;
    std::cout << "  - " << TRAIN_IMAGES_DIR << " (训练图像)" << std::endl;
    std::cout << "  - " << TRAIN_LABELS_DIR << " (训练标注)" << std::endl;
    std::cout << "  - " << VAL_IMAGES_DIR << " (验证图像)" << std::endl;
    std::cout << "  - " << VAL_LABELS_DIR << " (验证标注)" << std::endl;
}

int main() {
    organizeDataset();
    return 0;
}