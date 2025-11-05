#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <format>

namespace fs = std::filesystem;

const std::string VIDEO_PATH = "能量机关视频.mp4";
const std::string OUTPUT_DIR = "frames_to_annotate";
const int INTERVAL = 10;  // 每INTERVAL帧提取一帧

int main() {
    // 创建输出目录
    fs::create_directories(OUTPUT_DIR);
    
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return 1;
    }
    
    int frameCount = 0;
    int savedCount = 0;
    
    std::cout << "开始从视频提取帧，保存到 " << OUTPUT_DIR << " 目录..." << std::endl;
    
    cv::Mat frame;
    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            break;
        }
        
        // 每隔INTERVAL帧保存一帧
        if (frameCount % INTERVAL == 0) {
            std::string outputPath = OUTPUT_DIR + "/" + 
                                    std::format("frame_{:06d}.jpg", frameCount);
            cv::imwrite(outputPath, frame);
            savedCount++;
            std::cout << "已保存: " << outputPath << std::endl;
        }
        
        frameCount++;
    }
    
    cap.release();
    std::cout << "\n提取完成！共提取 " << savedCount << " 帧图像" << std::endl;
    std::cout << "图像保存在: " << OUTPUT_DIR << std::endl;
    std::cout << "\n下一步：使用 LabelImg 或 Roboflow 标注这些图像" << std::endl;
    
    return 0;
}