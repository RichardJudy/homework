#ifndef YOLO_H
#define YOLO_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <windows.h>

// 检测结果结构
struct Detection {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;      // 置信度
    int class_id;          // 类别ID
    std::string class_name; // 类别名称
};

// 性能分析器类
class PerformanceProfiler {
public:
    PerformanceProfiler();
    void start_frame();
    void end_frame();
    void start_step(const std::string& step_name);
    void end_step(const std::string& step_name = "");
    
    struct StepStats {
        double mean;
        double median;
        double min;
        double max;
        double std;
        double total;
        int count;
    };
    
    struct FrameStats {
        double mean;
        double median;
        double min;
        double max;
        double std;
        double p95;
        double p99;
    };
    
    std::map<std::string, StepStats> get_step_statistics() const;
    FrameStats get_frame_statistics() const;
    void print_report(int frame_count) const;
    void save_report(const std::string& filename = "performance_report.txt") const;

private:
    std::map<std::string, std::vector<double>> timings_;
    std::vector<double> frame_times_;
    std::chrono::time_point<std::chrono::high_resolution_clock> current_frame_start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> current_step_start_;
    std::string current_step_name_;
    
    double calculate_mean(const std::vector<double>& values) const;
    double calculate_median(const std::vector<double>& values) const;
    double calculate_std(const std::vector<double>& values, double mean) const;
    double calculate_percentile(const std::vector<double>& values, double percentile) const;
};

// YOLO检测器类
class YOLODetector {
public:
    YOLODetector();
    ~YOLODetector();
    
    bool load_model(const std::string& model_path);
    std::string find_latest_model(const std::string& base_dir);
    std::vector<Detection> detect(const cv::Mat& frame, float conf_thresh = 0.25f, float iou_thresh = 0.45f);
    std::vector<int> filter_by_color(const std::vector<Detection>& detections, const std::string& target_color);
    cv::Mat draw_detections(const cv::Mat& frame, const std::vector<Detection>& detections, const std::vector<int>& filtered_indices);
    
    void detect_video(const std::string& video_path, 
                     float conf_thresh = 0.25f,
                     float iou_thresh = 0.45f,
                     const std::string& target_color = "all",
                     bool enable_profiling = true,
                     bool save_result = false);
    
    void detect_image(const std::string& image_path,
                     float conf_thresh = 0.25f,
                     float iou_thresh = 0.45f,
                     const std::string& target_color = "all",
                     bool save_result = false);
    
    std::map<int, std::string> get_class_names() const { return class_names_; }

private:
    Ort::Env env_;
    Ort::Session* session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    
    int input_width_;
    int input_height_;
    std::map<int, std::string> class_names_;
    
    cv::Mat preprocess(const cv::Mat& frame);
    std::vector<Detection> postprocess(const std::vector<float>& output, 
                                      int img_width, int img_height,
                                      float conf_thresh, float iou_thresh);
    float calculate_iou(const Detection& a, const Detection& b);
    void nms(std::vector<Detection>& detections, float iou_thresh);
    std::string get_class_name(int class_id) const;
};

// 工具函数
std::string join_path(const std::string& dir, const std::string& filename);
bool file_exists(const std::string& path);
std::vector<std::string> find_files(const std::string& dir, const std::string& pattern);

#endif // YOLO_H

