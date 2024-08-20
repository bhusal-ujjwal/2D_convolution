cmake_minimum_required(VERSION 3.0)

project(2DConvo LANGUAGES CXX CUDA) 

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(2DConvo main.cu utils/pngio.h utils/pngio.cpp)

target_link_libraries(2DConvo ${PNG_LIBRARY})