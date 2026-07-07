// opaque_data_sender.cpp — reference for sending camera_to_base over the
// CloudXR opaque data channel (XR_NV_opaque_data_channel), so the overlay
// receives the extrinsic on the SAME WebRTC connection as the video — no
// separate rosbridge hop, latency aligned with the frames.
//
// WHERE THIS GOES: in-process in your OpenXR app. xr_frame_layer.cpp is the
// natural home — it already holds the XrInstance and hooks xrEndFrame. The one
// thing it must obtain is camera_to_base each frame (eye_in_base = inv(H_base)
// @ H_eye_L, exactly as xr_ros_bridge.py computes it). The frame layer already
// has the eye pose from the XrCompositionLayerProjection views; feed it the
// base pose the same way your app feeds xr_ros_bridge (your existing socket
// payload already carries H_base + H_eye_L), and compute c2b here.
//
// ENABLE THE EXTENSION: add "XR_NV_opaque_data_channel" to the enabled
// extensions at xrCreateInstance. As an API layer you can request it in
// hk_CreateApiLayerInstance the same way the foveation strip is handled.
//
// WIRE PROTOCOL (must match the client parser in main.ts):
//   byte 0      : message type   (0x00 = extrinsic, 0x01 = joints)
//   type 0x00   : + 16 × float32 little-endian = camera_to_base, ROW-MAJOR
//                 (same 16-float flat layout as H_c2b in xr_ros_bridge.py)
//   type 0x01   : + uint8 count + count × float32 little-endian = joint rads
//
// UUID: must match the client byte-for-byte (16 bytes). Pick one and hard-code
// it on both ends.

#include <openxr/openxr.h>
#include <XR_NV_opaque_data_channel.h>   // PFNs/structs/enums for the NV opaque data channel (ships with CloudXR SDK)
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include "opaque_data_sender.h"   // C-linkage + default visibility for the exported fns

// ---- NV opaque data channel entry points (resolve via xrGetInstanceProcAddr) ----
// These are provided by the CloudXR runtime; declare the PFN types from the
// runtime's openxr headers (XR_NV_opaque_data_channel). Resolve once at init.
static PFN_xrCreateOpaqueDataChannelNV   pfnCreate   = nullptr;
static PFN_xrGetOpaqueDataChannelStateNV pfnGetState = nullptr;
static PFN_xrSendOpaqueDataChannelNV     pfnSend     = nullptr;
static PFN_xrShutdownOpaqueDataChannelNV pfnShutdown = nullptr;
static PFN_xrDestroyOpaqueDataChannelNV  pfnDestroy  = nullptr;

static XrOpaqueDataChannelNV g_chan = XR_NULL_HANDLE;
static bool                  g_connected = false;

// 16-byte UUID shared with the client. Example value — change to your own,
// and put the SAME bytes in main.ts (CHANNEL_UUID).
static const uint8_t kChannelUuid[16] = {
    0x52,0x6f,0x62,0x6f,0x52,0x65,0x67,0x00,   // "RoboReg\0"
    0x65,0x78,0x74,0x72,0x69,0x6e,0x73,0x00 }; // "extrins\0"

void OpaqueSender_ResolveProcs(XrInstance instance, PFN_xrGetInstanceProcAddr gipa) {
    if (!gipa) { fprintf(stderr, "[opaque] no gipa\n"); return; }
    gipa(instance, "xrCreateOpaqueDataChannelNV",
                          (PFN_xrVoidFunction*)&pfnCreate);
    gipa(instance, "xrGetOpaqueDataChannelStateNV",
                          (PFN_xrVoidFunction*)&pfnGetState);
    gipa(instance, "xrSendOpaqueDataChannelNV",
                          (PFN_xrVoidFunction*)&pfnSend);
    gipa(instance, "xrShutdownOpaqueDataChannelNV",
                          (PFN_xrVoidFunction*)&pfnShutdown);
    gipa(instance, "xrDestroyOpaqueDataChannelNV",
                          (PFN_xrVoidFunction*)&pfnDestroy);
}

// Create the channel once we have instance + systemId (e.g. after xrGetSystem).
bool OpaqueSender_Create(XrInstance instance, XrSystemId systemId) {
    if (!pfnCreate) { fprintf(stderr, "[opaque] procs not resolved\n"); return false; }
    XrOpaqueDataChannelCreateInfoNV ci{};
    ci.type     = XR_TYPE_OPAQUE_DATA_CHANNEL_CREATE_INFO_NV;
    ci.next     = nullptr;
    ci.systemId = systemId;
    memcpy(&ci.uuid, kChannelUuid, 16);
    XrResult r = pfnCreate(instance, &ci, &g_chan);
    if (r != XR_SUCCESS) { fprintf(stderr, "[opaque] create failed: %d\n", r); return false; }
    fprintf(stderr, "[opaque] channel created, waiting for client...\n");
    return true;
}

// Non-blocking connection check — call from the frame loop until it returns true.
bool OpaqueSender_Ready() {
    if (g_connected) return true;
    if (g_chan == XR_NULL_HANDLE || !pfnGetState) return false;
    XrOpaqueDataChannelStateNV st{};
    st.type = XR_TYPE_OPAQUE_DATA_CHANNEL_STATE_NV;
    if (pfnGetState(g_chan, &st) != XR_SUCCESS) return false;
    if (st.state == XR_OPAQUE_DATA_CHANNEL_STATUS_CONNECTED_NV) {
        g_connected = true;
        fprintf(stderr, "[opaque] channel CONNECTED\n");
    }
    return g_connected;
}

// Call once per xrEndFrame, AFTER you have camera_to_base for this frame.
// c2b_rowmajor: 16 floats, row-major (same as xr_ros_bridge H_c2b).
void OpaqueSender_SendExtrinsic(const float c2b_rowmajor[16]) {
    if (!OpaqueSender_Ready() || !pfnSend) return;
    uint8_t buf[1 + 16 * sizeof(float)];
    buf[0] = 0x00;                                  // type: extrinsic
    memcpy(buf + 1, c2b_rowmajor, 16 * sizeof(float));
    XrResult r = pfnSend(g_chan, (uint32_t)sizeof(buf), buf);
    if (r != XR_SUCCESS)
        fprintf(stderr, "[opaque] send failed: %d\n", r);
}

// Optional: joints alongside, same channel, type 0x01.
void OpaqueSender_SendJoints(const float* rads, uint8_t count) {
    if (!OpaqueSender_Ready() || !pfnSend) return;
    uint8_t buf[2 + 32 * sizeof(float)];
    buf[0] = 0x01;                                  // type: joints
    buf[1] = count;
    memcpy(buf + 2, rads, count * sizeof(float));
    pfnSend(g_chan, (uint32_t)(2 + count * sizeof(float)), buf);
}

void OpaqueSender_Destroy() {
    if (g_chan == XR_NULL_HANDLE) return;
    if (pfnShutdown) pfnShutdown(g_chan);
    if (pfnDestroy)  pfnDestroy(g_chan);
    g_chan = XR_NULL_HANDLE; g_connected = false;
}