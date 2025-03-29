cmake_minimum_required(VERSION 3.10.0)
project(cuda-devops VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)