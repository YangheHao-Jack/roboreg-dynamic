// opaque_data_sender.h — interface to the opaque-data-channel sender.
//
// Two audiences:
//  - xr_frame_layer.cpp (same .so): calls ResolveProcs/Create/Ready/Destroy.
//    These stay normal C++ symbols; the layer's hidden-visibility preset is
//    fine because intra-.so calls don't need the dynamic symbol table.
//  - the Python app via ctypes: calls SendExtrinsic/SendJoints. Those MUST be
//    exported (default visibility) despite CMAKE_CXX_VISIBILITY_PRESET=hidden,
//    and have C linkage so dlsym finds the unmangled name.
#pragma once
#include <openxr/openxr.h>
#include <cstdint>

#define OPAQUE_EXPORT extern "C" __attribute__((visibility("default")))

// Internal (called from the layer). Pass the next-layer xrGetInstanceProcAddr;
// the layer can't call the global symbol (it isn't linked in).
void OpaqueSender_ResolveProcs(XrInstance instance, PFN_xrGetInstanceProcAddr gipa);
bool OpaqueSender_Create(XrInstance instance, XrSystemId systemId);
bool OpaqueSender_Ready();
void OpaqueSender_Destroy();

// Exported for the app (ctypes):
OPAQUE_EXPORT void OpaqueSender_SendExtrinsic(const float c2b_rowmajor[16]);
OPAQUE_EXPORT void OpaqueSender_SendJoints(const float* rads, uint8_t count);
