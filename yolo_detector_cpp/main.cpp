#include "yolo.h"
#include <iostream>
#include <windows.h>

// 获取可执行文件所在目录
std::string get_executable_dir() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string exe_path(buffer);
    size_t last_slash = exe_path.find_last_of("\\/");
    if (last_slash != std::string::npos) {
        return exe_path.substr(0, last_slash);
    }
    return ".";
}

int main() {
    // 获取可执行文件目录
    std::string script_dir = get_executable_dir();
    std::cout << "脚本目录: " << script_dir << std::endl;
    
    // 创建检测器
    YOLODetector detector;
    
    // 查找最新的模型
    std::string model_path = detector.find_latest_model(script_dir);
    
    // 如果找不到，尝试其他路径
    if (model_path.empty()) {
        std::vector<std::string> fallback_paths = {
            join_path(script_dir, "runs\\detect\\rectangle_detector4\\weights\\best.onnx"),
            join_path(script_dir, "runs\\detect\\yes2\\weights\\best.onnx"),
            join_path(script_dir, "runs\\detect\\yes\\weights\\best.onnx"),
            join_path(script_dir, "best.onnx"),
            join_path(script_dir, "runs\\detect\\no_rectangle_detector\\weights\\best.onnx")
        };
        
        for (const auto& path : fallback_paths) {
            if (file_exists(path)) {
                model_path = path;
                break;
            }
        }
    }
    
    // 加载模型
    if (model_path.empty() || !detector.load_model(model_path)) {
        std::cerr << "⚠ 未找到训练好的模型" << std::endl;
        std::cerr << "提示: 请确保模型文件存在，或使用Python脚本将.pt模型转换为.onnx格式" << std::endl;
        return 1;
    }
    
    std::cout << "✓ 置信度阈值: 0.25" << std::endl;
    std::cout << "✓ 目标颜色: all" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // 检测参数
    float conf_thresh = 0.25f;
    float iou_thresh = 0.45f;
    std::string target_color = "all";
    bool enable_profiling = true;
    bool save_result = false;
    
    // 视频或图像路径
    std::string video_path = join_path(script_dir, "自瞄大作业_红色装甲板.mp4");
    std::string image_path = "";  // 如果检测图像，设置图像路径
    
    // 选择检测模式
    if (!image_path.empty() && file_exists(image_path)) {
        // 检测图像
        detector.detect_image(image_path, conf_thresh, iou_thresh, target_color, save_result);
    } else if (file_exists(video_path)) {
        // 检测视频
        detector.detect_video(video_path, conf_thresh, iou_thresh, target_color, enable_profiling, save_result);
    } else {
        std::cerr << "错误: 未找到视频文件: " << video_path << std::endl;
        std::cerr << "提示: 可以设置 image_path 来检测单张图像" << std::endl;
        return 1;
    }
    
    return 0;
}


