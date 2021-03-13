cmake_minimum_required(VERSION 3.5)
project(raisin_learning_controller)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

list(APPEND CMAKE_PREFIX_PATH "../raisin_torch/pytorch/libtorch")
find_package(ament_cmake REQUIRED)
find_package(Torch REQUIRED)
find_package(raisin_controller REQUIRED)
add_library(${PROJECT_NAME} SHARED
        src/${PROJECT_NAME}/hubodog_learning_controller.cpp
        )

target_link_libraries(${PROJECT_NAME} raisin_controller::raisin_controller ${TORCH_LIBRARIES})
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

ament_export_interfaces(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()