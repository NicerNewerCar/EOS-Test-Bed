cmake_minimum_required(VERSION 3.8)
# TODO:
# Add EOS Kernel
# Add Autoscoper TIFF reader
# Test and Debugging 

project(EOS)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp src/TiffImage.cpp src/EosDRR2D.cl)
set(HEADER_FILES src/TiffImage.h)

find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLICDLoader REQUIRED)
find_package(TIFF REQUIRED MODULE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

# OpenCL Stuff
target_compile_definitions(${PROJECT_NAME} PUBLIC CL_TARGET_OPENCL_VERSION=120) # Target 1.2.0
target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL)

# TIFF Stuff
target_link_libraries(${PROJECT_NAME} PUBLIC TIFF::TIFF)
target_compile_definitions(${PROJECT_NAME} PUBLIC USE_LIBTIFF)
