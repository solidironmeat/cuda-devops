cmake_minimum_required(VERSION 3.28)
project(operations VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GoogleTest)

find_package(CUDAToolkit REQUIRED)
find_package(GTest CONFIG REQUIRED)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
FILE(GLOB DIRS RELATIVE ${SOURCE_DIR} ${SOURCE_DIR}/*)
foreach(TARGET ${DIRS})
    if(IS_DIRECTORY ${SOURCE_DIR}/${TARGET})
        # Libraries
        file(GLOB_RECURSE TARGET_SRC_FILES "${SOURCE_DIR}/${TARGET}/*.cu")
        add_library(${TARGET} ${TARGET_SRC_FILES})
        target_link_libraries(${TARGET} PUBLIC
            CUDA::cudart      # CUDA Runtime API
        )
        target_include_directories(${TARGET} PUBLIC
            ${SOURCE_DIR}/${TARGET}/include
            ${CUDA_INCLUDE_DIRS}
        )

        set_target_properties(${TARGET} PROPERTIES
            CMAKE_CXX_STANDARD 20
            CMAKE_CXX_STANDARD_REQUIRED ON
            CMAKE_CXX_EXTENSIONS OFF
            CMAKE_EXPORT_COMPILE_COMMANDS ON
            CUDA_ARCHITECTURES 75
        )

        # Tests
        file(GLOB_RECURSE TARGET_TEST_SRC_FILES "${SOURCE_DIR}/${TARGET}/test/${TARGET}_test.cc")
        add_executable(${TARGET}_test ${TARGET_TEST_SRC_FILES})
        target_link_libraries(${TARGET}_test PRIVATE
            GTest::gtest_main
            pthread
            ${TARGET}
        )
        target_include_directories(${TARGET}_test PRIVATE
            ${GTEST_INCLUDE_DIRS}
        )
        gtest_discover_tests(${TARGET}_test)

        if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${TARGET} PUBLIC -Wall -Wextra -Wno-pedantic)
        else()
            target_compile_options(${TARGET} PUBLIC -W3)
        endif()
    endif()
endforeach()
