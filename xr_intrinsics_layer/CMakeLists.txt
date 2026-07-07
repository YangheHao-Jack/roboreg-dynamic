cmake_minimum_required(VERSION 3.20)
project(xr_intrinsics_layer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)   # only xrNegotiateLoaderApiLayerInterface is exported
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# OpenXR headers are fetched by the build.sh helper into ./openxr_headers/
# If you place them elsewhere, point -DOPENXR_INCLUDE_DIR=/path at config time.
if(NOT DEFINED OPENXR_INCLUDE_DIR)
    set(OPENXR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openxr_headers/include"
        CACHE PATH "Path containing openxr/openxr.h and openxr/openxr_loader_negotiation.h")
endif()

if(NOT EXISTS "${OPENXR_INCLUDE_DIR}/openxr/openxr.h")
    message(FATAL_ERROR
        "openxr.h not found at ${OPENXR_INCLUDE_DIR}/openxr/openxr.h\n"
        "Run build.sh (which fetches Khronos headers), or pass -DOPENXR_INCLUDE_DIR=<path>.")
endif()
if(NOT EXISTS "${OPENXR_INCLUDE_DIR}/openxr/openxr_loader_negotiation.h")
    message(FATAL_ERROR
        "openxr_loader_negotiation.h not found at ${OPENXR_INCLUDE_DIR}/openxr/openxr_loader_negotiation.h")
endif()

add_library(xr_intrinsics_layer SHARED xr_intrinsics_layer.cpp)
target_include_directories(xr_intrinsics_layer PRIVATE ${OPENXR_INCLUDE_DIR})
target_compile_options(xr_intrinsics_layer PRIVATE -Wall -Wextra -Wno-unused-parameter -O2)
set_target_properties(xr_intrinsics_layer PROPERTIES OUTPUT_NAME "xr_intrinsics_layer")

# After build, copy the manifest next to the .so so XR_API_LAYER_PATH can point
# at the build directory directly.
add_custom_command(TARGET xr_intrinsics_layer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/xr_intrinsics_layer.json"
        "$<TARGET_FILE_DIR:xr_intrinsics_layer>/xr_intrinsics_layer.json")
