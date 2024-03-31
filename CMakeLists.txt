cmake_minimum_required(VERSION 3.8)
# May need to run the following command to get NVCC to work...
# export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'

project(EOS)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/TiffImage.cpp src/EOSDRR.cu main.cpp )
set(HEADER_FILES src/TiffImage.h src/EOSDRR.h src/cutil_math.h)

find_package(TIFF REQUIRED MODULE)
find_package(CUDA REQUIRED)

cuda_add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
# target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})

# TIFF Stuff
target_link_libraries(${PROJECT_NAME} TIFF::TIFF)
target_compile_definitions(${PROJECT_NAME} PUBLIC USE_LIBTIFF)
