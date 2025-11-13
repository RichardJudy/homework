#include "yolo.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <windows.h>
#include <functional>

// ========== 工具函数 ==========
std::string join_path(const std::string& dir, const std::string& filename) {
    if (dir.empty()) return filename;
    if (dir.back() == '\\' || dir.back() == '/') {
        return dir + filename;
    }
    return dir + "\\" + filename;
}

bool file_exists(const std::string& path) {
    DWORD dwAttrib = GetFileAttributesA(path.c_str());
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && 
            !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

std::vector<std::string> find_files(const std::string& dir, const std::string& pattern) {
    std::vector<std::string> files;
    std::string search_path = join_path(dir, "*");
    
    WIN32_FIND_DATAA find_data;
    HANDLE find_handle = FindFirstFileA(search_path.c_str(), &find_data);
    
    if (find_handle == INVALID_HANDLE_VALUE) {
        return files;
    }
    
    do {
        if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            std::string filename = find_data.cFileName;
            if (filename.find(pattern) != std::string::npos) {
                files.push_back(join_path(dir, filename));
            }
        }
    } while (FindNextFileA(find_handle, &find_data) != 0);
    
    FindClose(find_handle);
    return files;
}

// ========== PerformanceProfiler 实现 ==========
PerformanceProfiler::PerformanceProfiler() {
    current_frame_start_ = std::chrono::high_resolution_clock::now();
    current_step_start_ = std::chrono::high_resolution_clock::time_point();
}

void PerformanceProfiler::start_frame() {
    current_frame_start_ = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::end_frame() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - current_frame_start_);
    frame_times_.push_back(duration.count() / 1000.0); // 转换为毫秒
}

void PerformanceProfiler::start_step(const std::string& step_name) {
    if (!current_step_name_.empty()) {
        end_step();
    }
    current_step_name_ = step_name;
    current_step_start_ = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::end_step(const std::string& step_name) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - current_step_start_);
    std::string step = step_name.empty() ? current_step_name_ : step_name;
    if (!step.empty() && current_step_start_.time_since_epoch().count() > 0) {
        timings_[step].push_back(duration.count() / 1000.0); // 转换为毫秒
    }
    current_step_start_ = std::chrono::high_resolution_clock::time_point();
    current_step_name_.clear();
}

double PerformanceProfiler::calculate_mean(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double PerformanceProfiler::calculate_median(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    }
    return sorted[n/2];
}

double PerformanceProfiler::calculate_std(const std::vector<double>& values, double mean) const {
    if (values.empty()) return 0.0;
    double sum_sq_diff = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / values.size());
}

double PerformanceProfiler::calculate_percentile(const std::vector<double>& values, double percentile) const {
    if (values.empty()) return 0.0;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    size_t index = static_cast<size_t>(percentile / 100.0 * sorted.size());
    if (index >= sorted.size()) index = sorted.size() - 1;
    return sorted[index];
}

std::map<std::string, PerformanceProfiler::StepStats> PerformanceProfiler::get_step_statistics() const {
    std::map<std::string, StepStats> stats;
    for (const auto& pair : timings_) {
        if (!pair.second.empty()) {
            StepStats s;
            s.mean = calculate_mean(pair.second);
            s.median = calculate_median(pair.second);
            s.min = *std::min_element(pair.second.begin(), pair.second.end());
            s.max = *std::max_element(pair.second.begin(), pair.second.end());
            s.std = calculate_std(pair.second, s.mean);
            s.total = std::accumulate(pair.second.begin(), pair.second.end(), 0.0);
            s.count = static_cast<int>(pair.second.size());
            stats[pair.first] = s;
        }
    }
    return stats;
}

PerformanceProfiler::FrameStats PerformanceProfiler::get_frame_statistics() const {
    FrameStats stats;
    if (frame_times_.empty()) return stats;
    
    stats.mean = calculate_mean(frame_times_);
    stats.median = calculate_median(frame_times_);
    stats.min = *std::min_element(frame_times_.begin(), frame_times_.end());
    stats.max = *std::max_element(frame_times_.begin(), frame_times_.end());
    stats.std = calculate_std(frame_times_, stats.mean);
    stats.p95 = calculate_percentile(frame_times_, 95.0);
    stats.p99 = calculate_percentile(frame_times_, 99.0);
    
    return stats;
}

void PerformanceProfiler::print_report(int frame_count) const {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "性能分析报告 (共处理 " << frame_count << " 帧)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    FrameStats ft = get_frame_statistics();
    if (frame_times_.size() > 0) {
        std::cout << "\n【每帧总耗时统计】" << std::endl;
        std::cout << "  平均耗时: " << std::fixed << std::setprecision(2) << ft.mean << " ms" << std::endl;
        std::cout << "  中位数:   " << ft.median << " ms" << std::endl;
        std::cout << "  最小耗时: " << ft.min << " ms" << std::endl;
        std::cout << "  最大耗时: " << ft.max << " ms" << std::endl;
        std::cout << "  标准差:   " << ft.std << " ms" << std::endl;
        std::cout << "  95%分位:  " << ft.p95 << " ms" << std::endl;
        std::cout << "  99%分位:  " << ft.p99 << " ms" << std::endl;
        std::cout << "  是否满足10ms要求: " << (ft.mean <= 10 ? "✓ 是" : "✗ 否") << std::endl;
    }
    
    auto step_stats = get_step_statistics();
    if (!step_stats.empty()) {
        std::cout << "\n【各步骤耗时统计】" << std::endl;
        std::vector<std::pair<std::string, StepStats>> sorted_stats(step_stats.begin(), step_stats.end());
        std::sort(sorted_stats.begin(), sorted_stats.end(), 
                 [](const auto& a, const auto& b) { return a.second.mean > b.second.mean; });
        
        for (const auto& pair : sorted_stats) {
            std::cout << "\n  " << pair.first << ":" << std::endl;
            std::cout << "    平均耗时: " << std::fixed << std::setprecision(2) << pair.second.mean << " ms" << std::endl;
            std::cout << "    最大耗时: " << pair.second.max << " ms" << std::endl;
            std::cout << "    总耗时:   " << pair.second.total << " ms" << std::endl;
            std::cout << "    执行次数: " << pair.second.count << std::endl;
            if (frame_times_.size() > 0) {
                double percentage = (pair.second.mean / ft.mean) * 100;
                std::cout << "    占比:     " << std::fixed << std::setprecision(1) << percentage << "%" << std::endl;
            }
        }
        
        if (!sorted_stats.empty()) {
            std::cout << "\n【最耗时步骤】" << std::endl;
            std::cout << "  " << sorted_stats[0].first << ": " 
                      << std::fixed << std::setprecision(2) << sorted_stats[0].second.mean << " ms (平均)" << std::endl;
        }
    }
    
    std::cout << std::string(70, '=') << "\n" << std::endl;
}

void PerformanceProfiler::save_report(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法保存性能报告到: " << filename << std::endl;
        return;
    }
    
    int frame_count = static_cast<int>(frame_times_.size());
    file << std::string(70, '=') << "\n";
    file << "性能分析报告 (共处理 " << frame_count << " 帧)\n";
    file << std::string(70, '=') << "\n\n";
    
    FrameStats ft = get_frame_statistics();
    if (frame_times_.size() > 0) {
        file << "【每帧总耗时统计】\n";
        file << "  平均耗时: " << std::fixed << std::setprecision(2) << ft.mean << " ms\n";
        file << "  中位数:   " << ft.median << " ms\n";
        file << "  最小耗时: " << ft.min << " ms\n";
        file << "  最大耗时: " << ft.max << " ms\n";
        file << "  标准差:   " << ft.std << " ms\n";
        file << "  95%分位:  " << ft.p95 << " ms\n";
        file << "  99%分位:  " << ft.p99 << " ms\n\n";
    }
    
    auto step_stats = get_step_statistics();
    if (!step_stats.empty()) {
        file << "【各步骤耗时统计】\n";
        std::vector<std::pair<std::string, StepStats>> sorted_stats(step_stats.begin(), step_stats.end());
        std::sort(sorted_stats.begin(), sorted_stats.end(), 
                 [](const auto& a, const auto& b) { return a.second.mean > b.second.mean; });
        
        for (const auto& pair : sorted_stats) {
            file << "\n  " << pair.first << ":\n";
            file << "    平均耗时: " << std::fixed << std::setprecision(2) << pair.second.mean << " ms\n";
            file << "    最大耗时: " << pair.second.max << " ms\n";
            file << "    总耗时:   " << pair.second.total << " ms\n";
            file << "    执行次数: " << pair.second.count << "\n";
            if (frame_times_.size() > 0) {
                double percentage = (pair.second.mean / ft.mean) * 100;
                file << "    占比:     " << std::fixed << std::setprecision(1) << percentage << "%\n";
            }
        }
    }
    
    file.close();
    std::cout << "性能报告已保存到: " << filename << std::endl;
}

// ========== YOLODetector 实现 ==========
YOLODetector::YOLODetector() : env_(ORT_LOGGING_LEVEL_WARNING, "YOLODetector"), session_(nullptr) {
    input_width_ = 640;
    input_height_ = 640;
    
    // 初始化类别名称（根据dataset.yaml: blue=0, red=1）
    class_names_[0] = "blue";
    class_names_[1] = "red";
}

YOLODetector::~YOLODetector() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

std::string YOLODetector::find_latest_model(const std::string& base_dir) {
    std::string runs_dir = join_path(base_dir, "runs\\detect");
    
    std::vector<std::string> models;
    
    // 查找 no_rectangle_detector* 模型
    std::string search_path = join_path(runs_dir, "*");
    WIN32_FIND_DATAA find_data;
    HANDLE find_handle = FindFirstFileA(search_path.c_str(), &find_data);
    
    if (find_handle != INVALID_HANDLE_VALUE) {
        do {
            if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                std::string dir_name = find_data.cFileName;
                if (dir_name != "." && dir_name != ".." && dir_name.find("no_rectangle_detector") == 0) {
                    std::string model_path = join_path(join_path(runs_dir, dir_name), "weights\\best.onnx");
                    if (file_exists(model_path)) {
                        models.push_back(model_path);
                    }
                }
            }
        } while (FindNextFileA(find_handle, &find_data) != 0);
        FindClose(find_handle);
    }
    
    // 如果没找到，查找所有模型（递归搜索）
    if (models.empty()) {
        std::function<void(const std::string&)> search_recursive = [&](const std::string& dir) {
            std::string search = join_path(dir, "*");
            WIN32_FIND_DATAA fd;
            HANDLE h = FindFirstFileA(search.c_str(), &fd);
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                        std::string subdir = fd.cFileName;
                        if (subdir != "." && subdir != "..") {
                            search_recursive(join_path(dir, subdir));
                        }
                    } else {
                        std::string filename = fd.cFileName;
                        if (filename == "best.onnx" || filename == "best.pt") {
                            models.push_back(join_path(dir, filename));
                        }
                    }
                } while (FindNextFileA(h, &fd) != 0);
                FindClose(h);
            }
        };
        search_recursive(runs_dir);
    }
    
    if (models.empty()) {
        return "";
    }
    
    // 返回最新的模型（按修改时间）
    std::string latest = models[0];
    FILETIME latest_time = {0, 0};
    HANDLE h = CreateFileA(latest.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h != INVALID_HANDLE_VALUE) {
        GetFileTime(h, NULL, NULL, &latest_time);
        CloseHandle(h);
    }
    
    for (const auto& model : models) {
        HANDLE h = CreateFileA(model.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            FILETIME time;
            GetFileTime(h, NULL, NULL, &time);
            CloseHandle(h);
            if (CompareFileTime(&time, &latest_time) > 0) {
                latest = model;
                latest_time = time;
            }
        }
    }
    
    return latest;
}

bool YOLODetector::load_model(const std::string& model_path) {
    if (!file_exists(model_path)) {
        std::cerr << "模型文件不存在: " << model_path << std::endl;
        return false;
    }
    
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // 尝试使用CUDA（如果可用）
        // OrtCUDAProviderOptions cuda_options{};
        // session_options.AppendExecutionProvider_CUDA(cuda_options);
        
        // 转换路径为宽字符（Windows需要）
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, NULL, 0);
        std::vector<wchar_t> wpath(size_needed);
        MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, wpath.data(), size_needed);
        
        session_ = new Ort::Session(env_, wpath.data(), session_options);
        
        // 获取输入输出信息
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        if (num_input_nodes == 0 || num_output_nodes == 0) {
            std::cerr << "模型输入或输出节点数量为0" << std::endl;
            return false;
        }
        
        // 获取输入信息
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();
        
        if (input_shape_.size() == 4) {
            input_height_ = static_cast<int>(input_shape_[2]);
            input_width_ = static_cast<int>(input_shape_[3]);
        }
        
        // 获取输入输出名称（使用新的API）
        for (size_t i = 0; i < num_input_nodes; i++) {
            input_names_str_.push_back(session_->GetInputNameAllocated(i, allocator_).get());
            input_names_.push_back(input_names_str_.back().c_str());
        }
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            output_names_str_.push_back(session_->GetOutputNameAllocated(i, allocator_).get());
            output_names_.push_back(output_names_str_.back().c_str());
        }
        
        std::cout << "✓ 模型加载成功: " << model_path << std::endl;
        std::cout << "✓ 输入尺寸: " << input_width_ << "x" << input_height_ << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat YOLODetector::preprocess(const cv::Mat& frame) {
    cv::Mat resized, blob;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_));
    resized.convertTo(blob, CV_32F, 1.0 / 255.0);
    
    // 转换为RGB并调整维度顺序为NCHW
    cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
    
    return blob;
}

std::vector<Detection> YOLODetector::postprocess(const std::vector<float>& output,
                                                  int img_width, int img_height,
                                                  float conf_thresh, float iou_thresh) {
    std::vector<Detection> detections;
    
    // YOLOv8输出格式: [batch, num_detections, 4+num_classes]
    // 假设输出形状为 [1, 8400, 6] (4个坐标 + 2个类别)
    size_t num_detections = output.size() / 6; // 假设6个值：x, y, w, h, conf1, conf2
    
    for (size_t i = 0; i < num_detections; i++) {
        size_t base_idx = i * 6;
        
        // 获取中心点和宽高（归一化坐标）
        float cx = output[base_idx + 0];
        float cy = output[base_idx + 1];
        float w = output[base_idx + 2];
        float h = output[base_idx + 3];
        
        // 找到最大置信度的类别
        float conf0 = output[base_idx + 4];
        float conf1 = output[base_idx + 5];
        float max_conf = (conf0 > conf1) ? conf0 : conf1;
        int class_id = (conf1 > conf0) ? 1 : 0;
        
        if (max_conf < conf_thresh) continue;
        
        // 转换为边界框坐标
        float x1 = (cx - w / 2.0f) * img_width;
        float y1 = (cy - h / 2.0f) * img_height;
        float x2 = (cx + w / 2.0f) * img_width;
        float y2 = (cy + h / 2.0f) * img_height;
        
        Detection det;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        det.confidence = max_conf;
        det.class_id = class_id;
        det.class_name = get_class_name(class_id);
        
        detections.push_back(det);
    }
    
    // 非极大值抑制
    nms(detections, iou_thresh);
    
    return detections;
}

float YOLODetector::calculate_iou(const Detection& a, const Detection& b) {
    float x1 = (a.x1 > b.x1) ? a.x1 : b.x1;
    float y1 = (a.y1 > b.y1) ? a.y1 : b.y1;
    float x2 = (a.x2 < b.x2) ? a.x2 : b.x2;
    float y2 = (a.y2 < b.y2) ? a.y2 : b.y2;
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

void YOLODetector::nms(std::vector<Detection>& detections, float iou_thresh) {
    if (detections.empty()) return;
    
    // 按置信度排序
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;
            
            float iou = calculate_iou(detections[i], detections[j]);
            if (iou > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    
    // 移除被抑制的检测
    std::vector<Detection> filtered;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            filtered.push_back(detections[i]);
        }
    }
    
    detections = filtered;
}

std::string YOLODetector::get_class_name(int class_id) const {
    auto it = class_names_.find(class_id);
    if (it != class_names_.end()) {
        return it->second;
    }
    return "unknown";
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& frame, float conf_thresh, float iou_thresh) {
    if (!session_) {
        std::cerr << "模型未加载" << std::endl;
        return {};
    }
    
    int img_height = frame.rows;
    int img_width = frame.cols;
    
    // 预处理
    cv::Mat blob = preprocess(frame);
    
    // 准备输入张量
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    size_t input_tensor_size = 3 * input_width_ * input_height_;
    std::vector<float> input_tensor_values(input_tensor_size);
    
    // 将OpenCV Mat转换为float数组
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_height_; h++) {
            for (int w = 0; w < input_width_; w++) {
                input_tensor_values[c * input_height_ * input_width_ + h * input_width_ + w] = 
                    blob.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    
    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size,
        input_shape.data(), input_shape.size());
    
    // 运行推理
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                       input_names_.data(), &input_tensor, 1,
                                       output_names_.data(), 1);
    
    // 获取输出
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    
    std::vector<float> output(output_data, output_data + output_size);
    
    // 后处理
    return postprocess(output, img_width, img_height, conf_thresh, iou_thresh);
}

std::vector<int> YOLODetector::filter_by_color(const std::vector<Detection>& detections, const std::string& target_color) {
    std::vector<int> filtered_indices;
    
    if (target_color == "all") {
        for (size_t i = 0; i < detections.size(); i++) {
            filtered_indices.push_back(static_cast<int>(i));
        }
        return filtered_indices;
    }
    
    for (size_t i = 0; i < detections.size(); i++) {
        const Detection& det = detections[i];
        if (target_color == "blue" && det.class_name == "blue") {
            filtered_indices.push_back(static_cast<int>(i));
        } else if (target_color == "red" && det.class_name == "red") {
            filtered_indices.push_back(static_cast<int>(i));
        }
    }
    
    return filtered_indices;
}

cv::Mat YOLODetector::draw_detections(const cv::Mat& frame, const std::vector<Detection>& detections, const std::vector<int>& filtered_indices) {
    cv::Mat annotated_frame = frame.clone();
    
    // 定义颜色映射 (BGR格式)
    cv::Scalar blue_color(255, 0, 0);   // 蓝色
    cv::Scalar red_color(0, 0, 255);    // 红色
    
    for (int idx : filtered_indices) {
        if (idx < 0 || idx >= static_cast<int>(detections.size())) continue;
        
        const Detection& det = detections[idx];
        cv::Point pt1(static_cast<int>(det.x1), static_cast<int>(det.y1));
        cv::Point pt2(static_cast<int>(det.x2), static_cast<int>(det.y2));
        
        cv::Scalar color = (det.class_name == "blue") ? blue_color : red_color;
        
        // 绘制边界框
        cv::rectangle(annotated_frame, pt1, pt2, color, 2);
        
        // 绘制标签
        std::ostringstream label;
        label << det.class_name << " " << std::fixed << std::setprecision(2) << det.confidence;
        std::string label_str = label.str();
        
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label_str, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
        cv::rectangle(annotated_frame, pt1, cv::Point(pt1.x + label_size.width, pt1.y - label_size.height - 10),
                     color, -1);
        cv::putText(annotated_frame, label_str, cv::Point(pt1.x, pt1.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
    
    return annotated_frame;
}

void YOLODetector::detect_video(const std::string& video_path,
                                float conf_thresh,
                                float iou_thresh,
                                const std::string& target_color,
                                bool enable_profiling,
                                bool save_result) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频文件: " << video_path << std::endl;
        return;
    }
    
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "视频信息: " << width << "x" << height << ", " << fps << " FPS, 总帧数: " << total_frames << std::endl;
    std::cout << "按 'q' 键退出，按空格键暂停/继续，按 'b' 切换蓝色，按 'r' 切换红色，按 'a' 显示全部" << std::endl;
    if (enable_profiling) {
        std::cout << "性能分析已启用，每 100 帧输出一次报告" << std::endl;
    }
    std::cout << std::string(60, '-') << std::endl;
    
    PerformanceProfiler* profiler = enable_profiling ? new PerformanceProfiler() : nullptr;
    std::string current_color = target_color;
    
    const std::string window_name = "YOLOv8 检测结果";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 800, 600);
    
    cv::VideoWriter video_writer;
    if (save_result) {
        std::string output_path = "detections\\detected_video.mp4";
        CreateDirectoryA("detections", NULL);
        video_writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
        std::cout << "检测结果将保存到: " << output_path << std::endl;
    }
    
    int frame_count = 0;
    bool pause = false;
    
    while (true) {
        if (!pause) {
            if (profiler) profiler->start_frame();
            
            if (profiler) profiler->start_step("read_frame");
            cv::Mat frame;
            bool success = cap.read(frame);
            if (!success) {
                std::cout << "视频播放完成" << std::endl;
                break;
            }
            frame_count++;
            if (profiler) profiler->end_step("read_frame");
            
            if (profiler) profiler->start_step("yolo_detection");
            std::vector<Detection> detections = detect(frame, conf_thresh, iou_thresh);
            if (profiler) profiler->end_step("yolo_detection");
            
            if (profiler) profiler->start_step("filter_results");
            std::vector<int> filtered_indices = filter_by_color(detections, current_color);
            if (profiler) profiler->end_step("filter_results");
            
            if (profiler) profiler->start_step("draw_results");
            cv::Mat annotated_frame = draw_detections(frame, detections, filtered_indices);
            
            // 添加帧数和颜色信息
            std::ostringstream info;
            info << "Frame: " << frame_count << "/" << total_frames << " | Color: " << current_color;
            cv::putText(annotated_frame, info.str(), cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // 添加检测统计信息
            std::ostringstream stats;
            stats << "Total: " << detections.size() << " | Filtered: " << filtered_indices.size();
            cv::putText(annotated_frame, stats.str(), cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            
            if (profiler) profiler->end_step("draw_results");
            
            if (profiler) profiler->start_step("save_video");
            if (video_writer.isOpened()) {
                video_writer.write(annotated_frame);
            }
            if (profiler) profiler->end_step("save_video");
            
            if (profiler) profiler->start_step("display");
            cv::imshow(window_name, annotated_frame);
            if (profiler) profiler->end_step("display");
            
            if (profiler) profiler->end_frame();
            
            // 打印检测信息
            if (filtered_indices.size() > 0) {
                std::cout << "帧 " << frame_count << ": 检测到 " << detections.size() 
                         << " 个目标，显示 " << filtered_indices.size() << " 个 (" << current_color << ")" << std::endl;
            }
            
            // 定期输出性能报告
            if (profiler && frame_count % 100 == 0) {
                profiler->print_report(frame_count);
            }
        }
        
        // 键盘控制
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        } else if (key == ' ') {
            pause = !pause;
            std::cout << (pause ? "暂停" : "继续") << std::endl;
        } else if (key == 'b') {
            current_color = "blue";
            std::cout << "切换到蓝色装甲板检测" << std::endl;
        } else if (key == 'r') {
            current_color = "red";
            std::cout << "切换到红色装甲板检测" << std::endl;
        } else if (key == 'a') {
            current_color = "all";
            std::cout << "显示所有装甲板" << std::endl;
        }
    }
    
    cap.release();
    if (video_writer.isOpened()) {
        video_writer.release();
    }
    cv::destroyAllWindows();
    
    if (profiler) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "最终性能分析报告" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        profiler->print_report(frame_count);
        profiler->save_report("performance_report.txt");
        delete profiler;
    }
    
    std::cout << "检测完成！" << std::endl;
}

void YOLODetector::detect_image(const std::string& image_path,
                               float conf_thresh,
                               float iou_thresh,
                               const std::string& target_color,
                               bool save_result) {
    if (!file_exists(image_path)) {
        std::cerr << "错误: 图像文件不存在: " << image_path << std::endl;
        return;
    }
    
    std::cout << "检测图像: " << image_path << std::endl;
    
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "错误: 无法读取图像文件" << std::endl;
        return;
    }
    
    std::vector<Detection> detections = detect(frame, conf_thresh, iou_thresh);
    std::vector<int> filtered_indices = filter_by_color(detections, target_color);
    cv::Mat annotated_frame = draw_detections(frame, detections, filtered_indices);
    
    // 添加颜色信息
    std::ostringstream info;
    info << "Color: " << target_color;
    cv::putText(annotated_frame, info.str(), cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    // 打印检测信息
    std::cout << "检测到 " << detections.size() << " 个目标，显示 " << filtered_indices.size() 
             << " 个 (" << target_color << "):" << std::endl;
    for (size_t i = 0; i < filtered_indices.size(); i++) {
        int idx = filtered_indices[i];
        const Detection& det = detections[idx];
        std::cout << "  [" << (i + 1) << "] " << det.class_name << ": 置信度=" 
                 << std::fixed << std::setprecision(2) << det.confidence
                 << ", 位置=(" << static_cast<int>(det.x1) << "," << static_cast<int>(det.y1)
                 << "," << static_cast<int>(det.x2) << "," << static_cast<int>(det.y2) << ")" << std::endl;
    }
    
    // 保存结果
    if (save_result) {
        CreateDirectoryA("detections", NULL);
        size_t last_slash = image_path.find_last_of("\\/");
        std::string filename = (last_slash != std::string::npos) ? 
                              image_path.substr(last_slash + 1) : image_path;
        std::string output_path = join_path("detections", "detected_" + filename);
        cv::imwrite(output_path, annotated_frame);
        std::cout << "检测结果已保存到: " << output_path << std::endl;
    }
    
    // 显示结果
    const std::string window_name = "YOLOv8 检测结果";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 800, 600);
    cv::imshow(window_name, annotated_frame);
    std::cout << "按任意键关闭窗口..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

