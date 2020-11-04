cmake_minimum_required(VERSION 3.0)
project(AlgoLib)

add_compile_options(-std=c++11 -fpermissive)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../kl_algo_com)

set(CUDA_NVCC_FLAGS -std=c++11 -Xcompiler -fexceptions -Xcompiler -fPIC 
-gencode arch=compute_35,code=sm_35
-gencode arch=compute_50,code=sm_50
-gencode arch=compute_60,code=sm_60
-gencode arch=compute_70,code=sm_70
-gencode arch=compute_75,code=sm_75
-gencode arch=compute_75,code=compute_75
)

#tensorrt
include (cmake/KLTensorRT.cmake)
link_directories(${KL_TENSORRT_LIBS_DIRS})
include_directories(${KL_TENSORRT_INCLUDE_DIRS})

#cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#opencv
include(cmake/FindOpenCV.cmake)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(
    ./Common
    ./face_feature_me
    ./Common/onnx_base
    ./face_feature_me_v2
)

list(APPEND SRC 
    ./Common/kl_rt_onnx_base_v7.cpp
    ./Common/kl_syncedmem.cpp
    ./face_feature_me/kl_face_infer.cpp
    ./face_feature_me/face_extractor_rt.cpp
    ./face_feature_me_v2/face_feature_v2_api.cpp
    ./face_feature_me_v2/face_feature_v2.cpp
    ./Common/onnx_base/onnx_dynamic_reshape_base_v1.cpp
    ./main.cpp
)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_link_libraries(${PROJECT_NAME} nvcaffe_parser)
target_link_libraries(${PROJECT_NAME} nvonnxparser)
target_link_libraries(${PROJECT_NAME} nvinfer_plugin)
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} libnvinfer_static.a)
target_link_libraries(${PROJECT_NAME} libmyelin_pattern_runtime_static.a)
target_link_libraries(${PROJECT_NAME} libmyelin_executor_static.a)
target_link_libraries(${PROJECT_NAME} libmyelin_compiler_static.a)
target_link_libraries(${PROJECT_NAME} libmyelin_pattern_library_static.a)


target_link_libraries(${PROJECT_NAME} blas)
target_link_libraries(${PROJECT_NAME} cudart)