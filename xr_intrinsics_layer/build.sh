#!/usr/bin/env bash
# Fetch Khronos OpenXR headers and build the intrinsics-sniffing API layer.
#
# Usage:
#   bash build.sh                    # default: fetch headers, configure, build
#   bash build.sh clean              # wipe build/ and openxr_headers/
#
# After success:
#   build/libxr_intrinsics_layer.so
#   build/xr_intrinsics_layer.json
# Register (from this directory):
#   export XR_API_LAYER_PATH="$PWD/build"
#   export XR_ENABLE_API_LAYERS=XR_APILAYER_INTRINSICS_SNIFF

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [ "$1" = "clean" ]; then
    rm -rf build openxr_headers
    echo "cleaned."
    exit 0
fi

# ── 1. Fetch headers if missing ──────────────────────────────────────────────
OPENXR_TAG="release-1.1.41"   # stable release; matches loader API version 1.0+
HDR_DIR="$HERE/openxr_headers"

if [ ! -f "$HDR_DIR/include/openxr/openxr.h" ] || \
   [ ! -f "$HDR_DIR/include/openxr/openxr_loader_negotiation.h" ]; then
    echo "[build] Fetching Khronos OpenXR-SDK ($OPENXR_TAG) headers..."
    rm -rf "$HDR_DIR" "$HDR_DIR.tmp"
    mkdir -p "$HDR_DIR.tmp"
    # Shallow clone, headers only
    git clone --depth 1 --branch "$OPENXR_TAG" \
        https://github.com/KhronosGroup/OpenXR-SDK.git "$HDR_DIR.tmp/OpenXR-SDK"
    mkdir -p "$HDR_DIR/include/openxr"
    cp "$HDR_DIR.tmp/OpenXR-SDK/include/openxr/openxr.h"                    "$HDR_DIR/include/openxr/"
    cp "$HDR_DIR.tmp/OpenXR-SDK/include/openxr/openxr_platform.h"           "$HDR_DIR/include/openxr/"
    cp "$HDR_DIR.tmp/OpenXR-SDK/include/openxr/openxr_platform_defines.h"   "$HDR_DIR/include/openxr/"
    cp "$HDR_DIR.tmp/OpenXR-SDK/include/openxr/openxr_reflection.h"         "$HDR_DIR/include/openxr/"
    # Older SDK versions put loader negotiation under src/common, newer ones in include/openxr.
    if [ -f "$HDR_DIR.tmp/OpenXR-SDK/include/openxr/openxr_loader_negotiation.h" ]; then
        cp "$HDR_DIR.tmp/OpenXR-SDK/include/openxr/openxr_loader_negotiation.h" "$HDR_DIR/include/openxr/"
    elif [ -f "$HDR_DIR.tmp/OpenXR-SDK/src/common/loader_interfaces.h" ]; then
        # Older name; the contents are compatible.
        cp "$HDR_DIR.tmp/OpenXR-SDK/src/common/loader_interfaces.h" \
           "$HDR_DIR/include/openxr/openxr_loader_negotiation.h"
    else
        echo "[build] ERROR: loader negotiation header not found in checkout"
        ls -la "$HDR_DIR.tmp/OpenXR-SDK/include/openxr/"
        exit 1
    fi
    rm -rf "$HDR_DIR.tmp"
    echo "[build] Headers ready at $HDR_DIR/include/openxr/"
else
    echo "[build] Using existing headers at $HDR_DIR/include/openxr/"
fi

# ── 2. Configure + build ─────────────────────────────────────────────────────
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENXR_INCLUDE_DIR="$HDR_DIR/include"
cmake --build build -j"$(nproc)"

echo ""
echo "================================================================"
echo "  Built: $HERE/build/libxr_intrinsics_layer.so"
echo "  Manifest: $HERE/build/xr_intrinsics_layer.json"
echo ""
echo "  Register before running Isaac Sim (in the SAME terminal):"
echo "    export XR_API_LAYER_PATH=\"$HERE/build\""
echo "    export XR_ENABLE_API_LAYERS=XR_APILAYER_INTRINSICS_SNIFF"
echo ""
echo "  Start the consumer first (Python):"
echo "    python3 xr_intrinsics_consumer.py"
echo "  Then launch Isaac Sim, Start AR, and connect the Quest."
echo "================================================================"
