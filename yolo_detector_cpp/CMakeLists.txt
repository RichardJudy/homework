cmake_minimum_required(VERSION 3.10)
project(YOLODetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# 配置路径 - 请根据你的实际安装路径修改
# ============================================

# OpenCV 路径（指向包含 OpenCVConfig.cmake 的目录）
# 例如: set(OpenCV_DIR "E:/opencv/build")
# 或者: set(OpenCV_DIR "C:/opencv/build")
set(OpenCV_DIR "E:/迅雷下载/opencv/build")

# ONNX Runtime 路径（指向解压后的根目录）
# 例如: set(ONNXRUNTIME_ROOT "E:/onnxruntime-win-x64-1.23.2/onnxruntime-win-x64-1.23.2")
set(ONNXRUNTIME_ROOT "E:/迅雷下载/onnxruntime-win-x64-1.23.2/onnxruntime-win-x64-1.23.2")

# ============================================
# 自动查找和配置
# ============================================

# 查找 OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "找到 OpenCV: ${OpenCV_VERSION}")
    message(STATUS "OpenCV 库路径: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "未找到 OpenCV，请检查 OpenCV_DIR 路径")
endif()

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIRS "${ONNXRUNTIME_ROOT}/lib")

# 检查 ONNX Runtime 路径
if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "ONNX Runtime 路径不存在: ${ONNXRUNTIME_ROOT}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}")
    message(FATAL_ERROR "ONNX Runtime 头文件目录不存在: ${ONNXRUNTIME_INCLUDE_DIRS}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIB_DIRS}/onnxruntime.lib")
    message(FATAL_ERROR "ONNX Runtime 库文件不存在: ${ONNXRUNTIME_LIB_DIRS}/onnxruntime.lib")
endif()

message(STATUS "ONNX Runtime 路径: ${ONNXRUNTIME_ROOT}")

# ============================================
# 包含目录
# ============================================
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# ============================================
# 源文件
# ============================================
set(SOURCES
    main.cpp
    yolo.cpp
)

set(HEADERS
    yolo.h
)

# ============================================
# 创建可执行文件
# ============================================
add_executable(yolo_detector ${SOURCES} ${HEADERS})

# ============================================
# 链接库
# ============================================
target_link_libraries(yolo_detector
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIB_DIRS}/onnxruntime.lib"
)

# ============================================
# 复制 DLL 文件到输出目录
# ============================================
add_custom_command(TARGET yolo_detector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/bin/opencv_world4120.dll"
        "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/bin/opencv_videoio_ffmpeg4120_64.dll"
        "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
        "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
    COMMENT "复制 DLL 文件到输出目录"
)

# ============================================
# 输出信息
# ============================================
message(STATUS "")
message(STATUS "===========================================")
message(STATUS "配置完成！")
message(STATUS "===========================================")
message(STATUS "OpenCV 版本: ${OpenCV_VERSION}")
message(STATUS "OpenCV 路径: ${OpenCV_DIR}")
message(STATUS "ONNX Runtime 路径: ${ONNXRUNTIME_ROOT}")
message(STATUS "===========================================")
message(STATUS "")
