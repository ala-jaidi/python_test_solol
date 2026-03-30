# CMakeLists.txt for SAM ONNX Inference Library
# Build shared library for Flutter FFI

cmake_minimum_required(VERSION 3.14)
project(sam_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# ONNX Runtime
# ============================================================
# Download ONNX Runtime from: https://github.com/microsoft/onnxruntime/releases
# Set ONNXRUNTIME_ROOT to the extracted directory

if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT not set. Download ONNX Runtime and set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
    REQUIRED
)

# ============================================================
# OpenCV (for ArUco)
# ============================================================
find_package(OpenCV REQUIRED COMPONENTS core imgproc aruco)

# ============================================================
# SAM Inference Library (with ArUco)
# ============================================================
add_library(sam_inference SHARED
    sam_inference.cpp
    aruco_calibration.cpp
)

target_include_directories(sam_inference PRIVATE
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(sam_inference PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
)

# Platform-specific settings
if(ANDROID)
    # Android NDK build
    set_target_properties(sam_inference PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${ANDROID_ABI}
    )
elseif(WIN32)
    # Windows DLL export
    target_compile_definitions(sam_inference PRIVATE SAM_EXPORTS)
elseif(APPLE)
    # macOS/iOS
    set_target_properties(sam_inference PROPERTIES
        FRAMEWORK TRUE
        MACOSX_FRAMEWORK_IDENTIFIER com.podiatry.sam
    )
endif()

# ============================================================
# Install
# ============================================================
install(TARGETS sam_inference
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES sam_inference.h
    DESTINATION include
)
