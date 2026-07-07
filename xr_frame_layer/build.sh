#!/usr/bin/env bash
# Build the frame-grab layer.
#   - Reuses OpenXR headers from ~/xr_intrinsics_layer/openxr_headers if present
#   - Requires Vulkan headers installed system-wide:
#       sudo apt install -y libvulkan-dev
#     (this provides /usr/include/vulkan/vulkan.h; we do NOT link libvulkan,
#      we dlsym it at runtime from the host process's own libvulkan.so.1.)

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [ "$1" = "clean" ]; then
    rm -rf build openxr_headers
    echo "cleaned."
    exit 0
fi

# OpenXR headers
SIBLING="$HOME/xr_intrinsics_layer/openxr_headers/include"
LOCAL="$HERE/openxr_headers/include"

if [ -f "$SIBLING/openxr/openxr.h" ] && [ -f "$SIBLING/openxr/openxr_loader_negotiation.h" ]; then
    HDR_DIR="$SIBLING"
    echo "[build] Using OpenXR headers from $HDR_DIR"
elif [ -f "$LOCAL/openxr/openxr.h" ]; then
    HDR_DIR="$LOCAL"
    echo "[build] Using local OpenXR headers at $HDR_DIR"
else
    OPENXR_TAG="release-1.1.41"
    HDR_DIR="$LOCAL"
    echo "[build] Fetching Khronos OpenXR-SDK ($OPENXR_TAG)..."
    rm -rf "$HERE/openxr_headers" "$HERE/openxr_headers.tmp"
    mkdir -p "$HERE/openxr_headers.tmp"
    git clone --depth 1 --branch "$OPENXR_TAG" \
        https://github.com/KhronosGroup/OpenXR-SDK.git \
        "$HERE/openxr_headers.tmp/OpenXR-SDK"
    mkdir -p "$LOCAL/openxr"
    cp "$HERE/openxr_headers.tmp/OpenXR-SDK/include/openxr/openxr.h"                  "$LOCAL/openxr/"
    cp "$HERE/openxr_headers.tmp/OpenXR-SDK/include/openxr/openxr_platform.h"         "$LOCAL/openxr/"
    cp "$HERE/openxr_headers.tmp/OpenXR-SDK/include/openxr/openxr_platform_defines.h" "$LOCAL/openxr/"
    cp "$HERE/openxr_headers.tmp/OpenXR-SDK/include/openxr/openxr_reflection.h"       "$LOCAL/openxr/"
    if [ -f "$HERE/openxr_headers.tmp/OpenXR-SDK/include/openxr/openxr_loader_negotiation.h" ]; then
        cp "$HERE/openxr_headers.tmp/OpenXR-SDK/include/openxr/openxr_loader_negotiation.h" "$LOCAL/openxr/"
    else
        cp "$HERE/openxr_headers.tmp/OpenXR-SDK/src/common/loader_interfaces.h" \
            "$LOCAL/openxr/openxr_loader_negotiation.h"
    fi
    rm -rf "$HERE/openxr_headers.tmp"
fi

# Sanity check: Vulkan headers must be present
if [ ! -f /usr/include/vulkan/vulkan_core.h ]; then
    echo "[build] ERROR: vulkan_core.h not found."
    echo "        Install with:  sudo apt install -y libvulkan-dev"
    exit 1
fi

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DOPENXR_INCLUDE_DIR="$HDR_DIR"
cmake --build build -j"$(nproc)"

echo ""
echo "================================================================"
echo "  Built: $HERE/build/libxr_frame_layer.so"
echo "================================================================"
echo ""
echo "To use:"
echo "  export XR_API_LAYER_PATH=\"\$HOME/xr_intrinsics_layer/build:$HERE/build:\$XR_API_LAYER_PATH\""
echo "  export XR_ENABLE_API_LAYERS=\"XR_APILAYER_INTRINSICS_SNIFF:XR_APILAYER_FRAME_GRAB\""
echo "  # Then run Isaac Sim as normal; test_cloudxr.py --capture-mode=layer will use it."
