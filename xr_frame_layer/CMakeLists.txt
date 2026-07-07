cmake_minimum_required(VERSION 3.20)
project(xr_frame_layer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# OpenXR headers — reuse from sibling intrinsics layer if present
if(NOT DEFINED OPENXR_INCLUDE_DIR)
    if(EXISTS "$ENV{HOME}/xr_intrinsics_layer/openxr_headers/include/openxr/openxr.h")
        set(OPENXR_INCLUDE_DIR "$ENV{HOME}/xr_intrinsics_layer/openxr_headers/include"
            CACHE PATH "OpenXR headers from sibling layer")
    else()
        set(OPENXR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openxr_headers/include"
            CACHE PATH "Path containing openxr/openxr.h")
    endif()
endif()

if(NOT EXISTS "${OPENXR_INCLUDE_DIR}/openxr/openxr.h")
    message(FATAL_ERROR "openxr.h not found at ${OPENXR_INCLUDE_DIR}/openxr/openxr.h")
endif()

# Headers that define XR_NV_opaque_data_channel — xrCreate/Send/Get/Shutdown/
# DestroyOpaqueDataChannelNV, XrOpaqueDataChannelCreateInfoNV, the STATUS enums.
# These are NOT in vanilla Khronos headers; they ship with the CloudXR SDK.
# Locate the dir, e.g.:
#   grep -rl OpaqueDataChannelNV ~/cloudxr-runtime ~/cloudxr-js ~/cloudxr* 2>/dev/null
# Then either:
#   (a) if it's a SEPARATE header, set CLOUDXR_OPENXR_INCLUDE_DIR to its dir, or
#   (b) if it's a full openxr.h superset, point OPENXR_INCLUDE_DIR at it instead
#       (and leave this empty) so you don't include two openxr.h's.
set(CLOUDXR_OPENXR_INCLUDE_DIR "" CACHE PATH "Dir with XR_NV_opaque_data_channel defs")

# Vulkan headers — find_package gives us include paths without forcing a link
find_package(Vulkan REQUIRED)

add_library(xr_frame_layer SHARED
    xr_frame_layer.cpp
    opaque_data_sender.cpp)        # <-- added: the opaque-data-channel sender

target_include_directories(xr_frame_layer PRIVATE
    ${OPENXR_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR})   # so opaque_data_sender.{h,cpp} find the header
if(CLOUDXR_OPENXR_INCLUDE_DIR)
    target_include_directories(xr_frame_layer PRIVATE ${CLOUDXR_OPENXR_INCLUDE_DIR})
endif()

# IMPORTANT: we link libdl for dlopen/dlsym but NOT libvulkan — we want the
# host process's existing libvulkan to be the one we dlsym against.
target_link_libraries(xr_frame_layer PRIVATE dl pthread)
target_compile_options(xr_frame_layer PRIVATE -Wall -Wextra -Wno-unused-parameter -O2)
set_target_properties(xr_frame_layer PROPERTIES OUTPUT_NAME "xr_frame_layer")

add_custom_command(TARGET xr_frame_layer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/xr_frame_layer.json"
        "$<TARGET_FILE_DIR:xr_frame_layer>/xr_frame_layer.json")
