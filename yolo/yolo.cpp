#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

// 配置参数
const std::string TRAINED_MODEL_PATH = "runs/detect/rectangle_detector/weights/best.onnx"; // 使用ONNX格式的模型
const std::string PRETRAINED_MODEL = "yolov8n.onnx";
const std::string VIDEO_PATH = "能量机关视频.mp4";
const std::string IMAGE_PATH = "";
const float CONF_THRESH = 0.25f;
const float IOU_THRESH = 0.45f;
const bool SHOW_INFO = true;
const bool SAVE_RESULT = false;
const std::string OUTPUT_DIR = "detections";
const std::string WINDOW_NAME = "YOLOv8 检测结果";
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

// 检测结果结构体
struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
};

// 自动查找最新的训练模型
std::string findLatestModel() {
    std::vector<std::string> patterns = {
        "runs/detect/rectangle_detector*/weights/best.onnx",
        "runs/detect/*/weights/best.onnx"
    };
    
    std::vector<fs::path> models;
    for (const auto& pattern : patterns) {
        size_t pos = pattern.find_last_of('/');
        std::string dir = pattern.substr(0, pos);
        std::string file_pattern = pattern.substr(pos + 1);
        
        if (fs::exists(dir)) {
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().filename().string() == "best.onnx") {
                    models.push_back(entry.path());
                }
            }
        }
    }
    
    if (!models.empty()) {
        // 返回最新的模型（按修改时间）
        return std::max_element(models.begin(), models.end(), 
            [](const fs::path& a, const fs::path& b) {
                return fs::last_write_time(a) < fs::last_write_time(b);
            })->string();
    }
    return "";
}

// 加载ONNX模型
Ort::Session* loadModel(const std::string& modelPath) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions session_options;
    
    // 启用GPU（如果可用）
    OrtCUDAProviderOptionsV2 cuda_options;
    try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options));
        std::cout << "✓ 使用CUDA加速" << std::endl;
    } catch (...) {
        std::cout << "⚠ CUDA不可用，使用CPU推理" << std::endl;
    }
    
    // 创建会话
    try {
        return new Ort::Session(env, modelPath.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "错误: 无法加载模型: " << e.what() << std::endl;
        return nullptr;
    }
}

// 图像预处理
cv::Mat preprocessImage(const cv::Mat& image, int inputWidth, int inputHeight, float& scaleW, float& scaleH) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputWidth, inputHeight));
    
    // 计算缩放比例
    scaleW = static_cast<float>(image.cols) / inputWidth;
    scaleH = static_cast<float>(image.rows) / inputHeight;
    
    // 归一化到0-1
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    // 转换为NCHW格式
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    
    // 创建输入张量
    cv::Mat input(inputHeight, inputWidth, CV_32F, cv::Scalar(0));
    for (int i = 0; i < inputHeight; ++i) {
        for (int j = 0; j < inputWidth; ++j) {
            input.at<float>(i, j) = channels[0].at<float>(i, j); // B
            input.at<float>(i, j + inputWidth) = channels[1].at<float>(i, j); // G
            input.at<float>(i, j + 2 * inputWidth) = channels[2].at<float>(i, j); // R
        }
    }
    
    return input;
}

// 非极大值抑制
std::vector<Detection> nonMaximumSuppression(std::vector<Detection>& detections, float iouThreshold) {
    std::vector<Detection> result;
    
    // 按置信度排序
    std::sort(detections.begin(), detections.end(), 
        [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    
    std::vector<bool> processed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (processed[i]) continue;
        
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (processed[j]) continue;
            
            // 计算IoU
            float intersectionX1 = std::max(detections[i].bbox.x, detections[j].bbox.x);
            float intersectionY1 = std::max(detections[i].bbox.y, detections[j].bbox.y);
            float intersectionX2 = std::min(detections[i].bbox.x + detections[i].bbox.width, 
                                          detections[j].bbox.x + detections[j].bbox.width);
            float intersectionY2 = std::min(detections[i].bbox.y + detections[i].bbox.height, 
                                          detections[j].bbox.y + detections[j].bbox.height);
            
            float intersectionArea = std::max(0.0f, intersectionX2 - intersectionX1) * 
                                  std::max(0.0f, intersectionY2 - intersectionY1);
            float unionArea = detections[i].bbox.area() + detections[j].bbox.area() - intersectionArea;
            float iou = intersectionArea / unionArea;
            
            if (iou > iouThreshold) {
                processed[j] = true;
            }
        }
    }
    
    return result;
}

// 处理推理结果
std::vector<Detection> processOutput(const float* output, int outputSize, 
                                    int inputWidth, int inputHeight, 
                                    float scaleW, float scaleH, 
                                    const std::vector<std::string>& classNames) {
    std::vector<Detection> detections;
    int numClasses = classNames.size();
    int elementsPerDetection = 5 + numClasses; // x, y, w, h, confidence, class probabilities
    int numDetections = outputSize / elementsPerDetection;
    
    for (int i = 0; i < numDetections; ++i) {
        int offset = i * elementsPerDetection;
        
        // 获取检测信息
        float x = output[offset];
        float y = output[offset + 1];
        float w = output[offset + 2];
        float h = output[offset + 3];
        float confidence = output[offset + 4];
        
        if (confidence < CONF_THRESH) continue;
        
        // 找到最高概率的类别
        int classId = 0;
        float maxClassProb = 0;
        for (int j = 0; j < numClasses; ++j) {
            float prob = output[offset + 5 + j];
            if (prob > maxClassProb) {
                maxClassProb = prob;
                classId = j;
            }
        }
        
        // 转换为原始图像坐标
        float left = (x - w / 2) * scaleW;
        float top = (y - h / 2) * scaleH;
        float right = (x + w / 2) * scaleW;
        float bottom = (y + h / 2) * scaleH;
        
        // 创建检测结果
        Detection det;
        det.bbox = cv::Rect(cv::Point(left, top), cv::Point(right, bottom));
        det.confidence = confidence;
        det.class_id = classId;
        det.class_name = classNames[classId];
        
        detections.push_back(det);
    }
    
    // 应用NMS
    return nonMaximumSuppression(detections, IOU_THRESH);
}

// 检测视频
void detectVideo(Ort::Session* session, const std::string& videoPath, 
                const std::vector<std::string>& classNames) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频文件: " << videoPath << std::endl;
        return;
    }
    
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "视频信息: " << width << "x" << height << ", " << fps << " FPS, 总帧数: " << totalFrames << std::endl;
    std::cout << "按 'q' 键退出，按空格键暂停/继续" << std::endl;
    std::cout << "-" << std::string(60, '-') << std::endl;
    
    // 设置显示窗口
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    
    // 视频保存（如果需要）
    cv::VideoWriter videoWriter;
    if (SAVE_RESULT) {
        fs::create_directories(OUTPUT_DIR);
        std::string outputPath = OUTPUT_DIR + "/detected_video.mp4";
        videoWriter.open(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
        std::cout << "检测结果将保存到: " << outputPath << std::endl;
    }
    
    int frameCount = 0;
    bool pause = false;
    
    // 获取模型输入尺寸（假设是640x640）
    int inputWidth = 640;
    int inputHeight = 640;
    
    cv::Mat frame;
    while (true) {
        if (!pause) {
            if (!cap.read(frame)) {
                std::cout << "视频播放完成" << std::endl;
                break;
            }
            frameCount++;
        }
        
        // 预处理
        float scaleW, scaleH;
        cv::Mat input = preprocessImage(frame, inputWidth, inputHeight, scaleW, scaleH);
        
        // 创建ONNX张量
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, input.ptr<float>(0), input.total(), inputShape.data(), inputShape.size());
        
        // 推理
        const char* inputNames[] = {"images"};
        const char* outputNames[] = {"output0"};
        
        Ort::Value outputTensor = session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);
        
        // 获取输出数据
        float* outputData = outputTensor.GetTensorMutableData<float>();
        size_t outputSize = outputTensor.GetTensorTypeAndShapeInfo().GetElementCount();
        
        // 处理输出
        std::vector<Detection> detections = processOutput(outputData, outputSize, 
                                                       inputWidth, inputHeight, 
                                                       scaleW, scaleH, classNames);
        
        // 绘制检测结果
        cv::Mat annotatedFrame = frame.clone();
        for (const auto& detection : detections) {
            // 绘制边界框
            cv::rectangle(annotatedFrame, detection.bbox, cv::Scalar(0, 255, 0), 2);
            
            // 添加标签
            std::string label = detection.class_name + ": " + 
                               std::to_string(detection.confidence).substr(0, 4);
            cv::putText(annotatedFrame, label, 
                       cv::Point(detection.bbox.x, detection.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        
        // 添加帧数信息
        cv::putText(annotatedFrame, "Frame: " + std::to_string(frameCount) + "/" + 
                   std::to_string(totalFrames),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        // 打印检测信息
        if (SHOW_INFO && !pause && !detections.empty()) {
            std::cout << "帧 " << frameCount << ": 检测到 " << detections.size() << " 个目标" << std::endl;
            for (size_t i = 0; i < detections.size(); ++i) {
                const auto& det = detections[i];
                std::cout << "  [" << i + 1 << "] " << det.class_name << ": 置信度=" 
                          << det.confidence << ", 位置=(" << det.bbox.x << "," << det.bbox.y << "," 
                          << det.bbox.x + det.bbox.width << "," << det.bbox.y + det.bbox.height << ")" << std::endl;
            }
        }
        
        // 保存视频帧
        if (videoWriter.isOpened()) {
            videoWriter.write(annotatedFrame);
        }
        
        // 显示检测结果
        cv::imshow(WINDOW_NAME, annotatedFrame);
        
        // 键盘控制
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        } else if (key == ' ') { // 空格键暂停/继续
            pause = !pause;
            std::cout << (pause ? "暂停" : "继续") << std::endl;
        }
    }
    
    cap.release();
    if (videoWriter.isOpened()) {
        videoWriter.release();
    }
    cv::destroyAllWindows();
    std::cout << "检测完成！" << std::endl;
}

// 检测图像
void detectImage(Ort::Session* session, const std::string& imagePath, 
                const std::vector<std::string>& classNames) {
    if (!fs::exists(imagePath)) {
        std::cerr << "错误: 图像文件不存在: " << imagePath << std::endl;
        return;
    }
    
    std::cout << "检测图像: " << imagePath << std::endl;
    
    // 读取图像
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "错误: 无法读取图像文件" << std::endl;
        return;
    }
    
    // 预处理
    int inputWidth = 640;
    int inputHeight = 640;
    float scaleW, scaleH;
    cv::Mat input = preprocessImage(frame, inputWidth, inputHeight, scaleW, scaleH);
    
    // 创建ONNX张量
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, input.ptr<float>(0), input.total(), inputShape.data(), inputShape.size());
    
    // 推理
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};
    
    Ort::Value outputTensor = session->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1);
    
    // 获取输出数据
    float* outputData = outputTensor.GetTensorMutableData<float>();
    size_t outputSize = outputTensor.GetTensorTypeAndShapeInfo().GetElementCount();
    
    // 处理输出
    std::vector<Detection> detections = processOutput(outputData, outputSize, 
                                                   inputWidth, inputHeight, 
                                                   scaleW, scaleH, classNames);
    
    // 绘制检测结果
    cv::Mat annotatedFrame = frame.clone();
    for (const auto& detection : detections) {
        // 绘制边界框
        cv::rectangle(annotatedFrame, detection.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 添加标签
        std::string label = detection.class_name + ": " + 
                           std::to_string(detection.confidence).substr(0, 4);
        cv::putText(annotatedFrame, label, 
                   cv::Point(detection.bbox.x, detection.bbox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    
    // 打印检测信息
    std::cout << "检测到 " << detections.size() << " 个目标:" << std::endl;
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        std::cout << "  [" << i + 1 << "] " << det.class_name << ": 置信度=" 
                  << det.confidence << ", 位置=(" << det.bbox.x << "," << det.bbox.y << "," 
                  << det.bbox.x + det.bbox.width << "," << det.bbox.y + det.bbox.height << ")" << std::endl;
    }
    
    // 保存结果
    if (SAVE_RESULT) {
        fs::create_directories(OUTPUT_DIR);
        std::string outputPath = OUTPUT_DIR + "/detected_" + fs::path(imagePath).filename().string();
        cv::imwrite(outputPath, annotatedFrame);
        std::cout << "检测结果已保存到: " << outputPath << std::endl;
    }
    
    // 显示结果
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    cv::imshow(WINDOW_NAME, annotatedFrame);
    std::cout << "按任意键关闭窗口..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    // 类别名称
    std::vector<std::string> classNames = {"rectangle", "goal_r"}; // 根据实际类别修改
    
    // 查找最新模型
    std::string modelPath = findLatestModel();
    if (modelPath.empty()) {
        modelPath = TRAINED_MODEL_PATH;
    }
    
    // 加载模型
    std::cout << "正在加载模型: " << modelPath << std::endl;
    Ort::Session* model = loadModel(modelPath);
    if (!model) {
        std::cerr << "错误: 无法加载模型，程序退出" << std::endl;
        return 1;
    }
    
    std::cout << "✓ 模型加载成功" << std::endl;
    std::cout << "✓ 置信度阈值: " << CONF_THRESH << std::endl;
    std::cout << "-" << std::string(60, '-') << std::endl;
    
    // 选择检测模式
    if (!IMAGE_PATH.empty()) {
        detectImage(model, IMAGE_PATH, classNames);
    } else if (fs::exists(VIDEO_PATH)) {
        detectVideo(model, VIDEO_PATH, classNames);
    } else {
        std::cerr << "错误: 未找到视频文件: " << VIDEO_PATH << std::endl;
        std::cout << "提示: 可以设置 IMAGE_PATH 来检测单张图像" << std::endl;
    }
    
    // 释放模型资源
    delete model;
    
    return 0;
}