cmake_minimum_required(VERSION 3.8)

project(EOS)

set(SOURCE_FILES main.cpp src/Raycaster.cpp)
set(HEADER_FILES src/Raycaster.hpp)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIR} )

# Teem stuff...
find_library(TEEM_LIB teem REQUIRED)
find_library(PNG_LIB png REQUIRED)
find_package(ZLIB REQUIRED)

# Really needs extreme optimizations
add_definitions(-O3)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries( EOS  ${ITK_LIBRARIES} Eigen3::Eigen ${TEEM_LIB} ${PNG_LIB} ZLIB::ZLIB)


