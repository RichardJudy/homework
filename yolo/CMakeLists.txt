cmake_minimum_required(VERSION 3.15)
project(YOLOv8_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 寻找OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("Found OpenCV: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "Could not find OpenCV")
endif()

# 寻找ONNX Runtime
# 可以通过ONNX_RUNTIME_ROOT_DIR环境变量指定ONNX Runtime的安装位置
find_path(ONNX_RUNTIME_INCLUDE_DIRS onnxruntime_cxx_api.h
    PATHS ${ONNX_RUNTIME_ROOT_DIR}/include
    $ENV{ONNX_RUNTIME_ROOT_DIR}/include
    "C:/Program Files/onnxruntime/include"
    "C:/Program Files (x86)/onnxruntime/include"
)

find_library(ONNX_RUNTIME_LIB onnxruntime
    PATHS ${ONNX_RUNTIME_ROOT_DIR}/lib
    $ENV{ONNX_RUNTIME_ROOT_DIR}/lib
    "C:/Program Files/onnxruntime/lib"
    "C:/Program Files (x86)/onnxruntime/lib"
)

if(ONNX_RUNTIME_INCLUDE_DIRS AND ONNX_RUNTIME_LIB)
    message("Found ONNX Runtime: ${ONNX_RUNTIME_LIB}")
else()
    message(WARNING "Could not find ONNX Runtime. Please set ONNX_RUNTIME_ROOT_DIR environment variable.")
endif()

# 寻找nlohmann/json库（用于labelme_to_yolo.cpp）
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    message("Found nlohmann/json: ${nlohmann_json_VERSION}")
else()
    message("nlohmann/json not found. Will download from GitHub.")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.10.5
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# 创建可执行文件
# 1. extract_frames.cpp
add_executable(extract_frames extract_frames.cpp)
target_link_libraries(extract_frames PRIVATE ${OpenCV_LIBS})

# 2. organize_dataset.cpp
add_executable(organize_dataset organize_dataset.cpp)
target_link_libraries(organize_dataset PRIVATE ${OpenCV_LIBS})

# 3. labelme_to_yolo.cpp
add_executable(labelme_to_yolo labelme_to_yolo.cpp)
target_link_libraries(labelme_to_yolo PRIVATE nlohmann_json::nlohmann_json)

# 4. yolo.cpp（主检测程序）
if(ONNX_RUNTIME_INCLUDE_DIRS AND ONNX_RUNTIME_LIB)
    add_executable(yolo yolo.cpp)
    target_include_directories(yolo PRIVATE ${ONNX_RUNTIME_INCLUDE_DIRS})
    target_link_libraries(yolo PRIVATE ${OpenCV_LIBS} ${ONNX_RUNTIME_LIB})
    
    # 复制ONNX Runtime DLL到输出目录（Windows）
    if(WIN32)
        find_file(ONNX_RUNTIME_DLL onnxruntime.dll
            PATHS ${ONNX_RUNTIME_ROOT_DIR}/bin
            $ENV{ONNX_RUNTIME_ROOT_DIR}/bin
            "C:/Program Files/onnxruntime/bin"
            "C:/Program Files (x86)/onnxruntime/bin"
        )
        if(ONNX_RUNTIME_DLL)
            add_custom_command(
                TARGET yolo POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNX_RUNTIME_DLL}
                $<TARGET_FILE_DIR:yolo>
            )
        endif()
    endif()
endif()

# 安装目标
install(TARGETS extract_frames organize_dataset labelme_to_yolo
    RUNTIME DESTINATION bin
)

if(TARGET yolo)
    install(TARGETS yolo
        RUNTIME DESTINATION bin
    )
endif()

# 输出构建信息
message("----------------------------------------")
message("构建配置总结:")
message("  C++标准: C++${CMAKE_CXX_STANDARD}")
message("  构建类型: ${CMAKE_BUILD_TYPE}")
message("  OpenCV: ${OpenCV_VERSION}")
if(ONNX_RUNTIME_INCLUDE_DIRS AND ONNX_RUNTIME_LIB)
    message("  ONNX Runtime: 已找到")
else()
    message("  ONNX Runtime: 未找到，yolo目标将不会构建")
endif()
message("----------------------------------------")