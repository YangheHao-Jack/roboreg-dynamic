// cxr_service.cpp
//
// Minimal CloudXR service launcher for Linux.
// Implements Option 3 from:
//   https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_runtime/getting_started.html
//
// Creates, starts, and joins a CloudXR service so that OpenXR apps
// (e.g. Isaac Sim with XR_RUNTIME_JSON=.../openxr_cloudxr.json) can
// connect to the CloudXR IPC socket at /run/user/$UID/ipc_cloudxr.
//
// Build:
//   cd ~/cloudxr-runtime
//   g++ -std=c++17 -Iinclude cxr_service.cpp -L. -lcloudxr \
//       -Wl,-rpath,'$ORIGIN' -o cxr-service
//
// Run (in a terminal — keeps running until Ctrl+C):
//   cd ~/cloudxr-runtime
//   ./cxr-service
//
// Then in a DIFFERENT terminal, run Isaac Sim with:
//   export XR_RUNTIME_JSON=~/cloudxr-runtime/openxr_cloudxr.json
//   export LD_LIBRARY_PATH=~/cloudxr-runtime:$LD_LIBRARY_PATH
//   python3 ~/test_cloudxr.py

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

#include <cxrServiceAPI.h>

// ────────────────────────────────────────────────────────────────────────────
// Globals for signal handling
// ────────────────────────────────────────────────────────────────────────────
static std::atomic<bool>    g_stop_requested{false};
static struct nv_cxr_service* g_service = nullptr;

static const char* result_name(nv_cxr_result_t r) {
    switch (r) {
        case NV_CXR_SUCCESS:                        return "SUCCESS";
        case NV_CXR_INTERNAL_SERVICE_ERROR:         return "INTERNAL_SERVICE_ERROR";
        case NV_CXR_STARTUP_FAILED:                 return "STARTUP_FAILED";
        case NV_CXR_NULL_OBJECT:                    return "NULL_OBJECT";
        case NV_CXR_NULL_PTR:                       return "NULL_PTR";
        case NV_CXR_SERVICE_ALREADY_STARTED:        return "SERVICE_ALREADY_STARTED";
        case NV_CXR_SERVICE_NOT_STARTED:            return "SERVICE_NOT_STARTED";
        case NV_CXR_PROPERTY_NAME_MALFORMED:        return "PROPERTY_NAME_MALFORMED";
        case NV_CXR_PROPERTY_NAME_INVALID:          return "PROPERTY_NAME_INVALID";
        case NV_CXR_ERROR_PROPERTY_VALUE_INVALID:   return "PROPERTY_VALUE_INVALID";
        case NV_CXR_ERROR_BUFFER_SIZE_INSUFFICIENT: return "BUFFER_SIZE_INSUFFICIENT";
        case NV_CXR_ERROR_PROPERTY_READ_ONLY:       return "PROPERTY_READ_ONLY";
        case NV_CXR_PORT_UNAVAILABLE:               return "PORT_UNAVAILABLE";
        case NV_CXR_ERROR_PROPERTY_WRITE_ONLY:      return "PROPERTY_WRITE_ONLY";
    }
    return "(unknown)";
}

static void handle_signal(int sig) {
    fprintf(stderr, "\n[cxr-service] signal %d received, stopping...\n", sig);
    g_stop_requested = true;
    if (g_service) {
        nv_cxr_service_stop(g_service);   // signal-safe enough for our use
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Event polling thread — prints connect/disconnect events
// ────────────────────────────────────────────────────────────────────────────
static void event_loop() {
    while (!g_stop_requested.load()) {
        nv_cxr_event_t event{};
        nv_cxr_result_t r = nv_cxr_service_poll_event(g_service, &event);
        if (r == NV_CXR_SUCCESS) {
            switch (event.type) {
                case NV_CXR_EVENT_CLOUDXR_CLIENT_CONNECTED:
                    printf("[event] CloudXR client CONNECTED\n");
                    break;
                case NV_CXR_EVENT_CLOUDXR_CLIENT_DISCONNECTED:
                    printf("[event] CloudXR client DISCONNECTED\n");
                    break;
                case NV_CXR_EVENT_OPENXR_APP_CONNECTED:
                    printf("[event] OpenXR app CONNECTED (e.g. Isaac Sim)\n");
                    break;
                case NV_CXR_EVENT_OPENXR_APP_DISCONNECTED:
                    printf("[event] OpenXR app DISCONNECTED\n");
                    break;
                case NV_CXR_EVENT_NONE:
                default:
                    break;
            }
            fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────────
int main() {
    // ── Signal handlers so Ctrl+C shuts down cleanly ─────────────────────────
    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

    // ── Print versions (handy for debugging compatibility issues) ────────────
    uint32_t la_maj=0, la_min=0, la_patch=0;
    uint32_t rt_maj=0, rt_min=0, rt_patch=0;
    nv_cxr_get_library_api_version(&la_maj, &la_min, &la_patch);
    nv_cxr_get_runtime_version(&rt_maj, &rt_min, &rt_patch);
    printf("[cxr-service] Library API:    %u.%u.%u\n", la_maj, la_min, la_patch);
    printf("[cxr-service] Runtime version: %u.%u.%u\n", rt_maj, rt_min, rt_patch);

    // ── Create service ───────────────────────────────────────────────────────
    printf("[cxr-service] Creating service...\n");
    nv_cxr_result_t r = nv_cxr_service_create(&g_service);
    if (r != NV_CXR_SUCCESS) {
        fprintf(stderr, "[cxr-service] nv_cxr_service_create FAILED: %s\n", result_name(r));
        return 1;
    }

    // ── Device profile: "auto-webrtc" enables web browser clients (CloudXR.js).
    //    Required for Meta Quest 3 via browser per the CloudXR.js docs:
    //    https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/getting_started.html
    //    Other profiles: "apple-vision-pro", "ios". For Isaac Sim + Quest, use auto-webrtc.
    {
        const char prop[] = "device-profile";
        const char val[]  = "auto-webrtc";
        r = nv_cxr_service_set_string_property(g_service,
                                               prop, sizeof(prop) - 1,
                                               val,  sizeof(val)  - 1);
        if (r != NV_CXR_SUCCESS) {
            fprintf(stderr, "[cxr-service] set device-profile FAILED: %s\n", result_name(r));
            nv_cxr_service_destroy(g_service);
            return 3;
        }
        printf("[cxr-service] Device profile set to: %s\n", val);
    }

    // ── Start the service ────────────────────────────────────────────────────
    printf("[cxr-service] Starting service (this will create the IPC socket)...\n");
    r = nv_cxr_service_start(g_service);
    if (r != NV_CXR_SUCCESS) {
        fprintf(stderr, "[cxr-service] nv_cxr_service_start FAILED: %s\n", result_name(r));
        nv_cxr_service_destroy(g_service);
        return 2;
    }
    printf("[cxr-service] Service started. Ready for OpenXR apps.\n");
    printf("[cxr-service] Press Ctrl+C to stop.\n");
    fflush(stdout);

    // ── Run event polling on a background thread ─────────────────────────────
    std::thread events(event_loop);

    // ── Block here until nv_cxr_service_stop is called (via signal) ──────────
    r = nv_cxr_service_join(g_service);
    if (r != NV_CXR_SUCCESS && r != NV_CXR_SERVICE_NOT_STARTED) {
        fprintf(stderr, "[cxr-service] nv_cxr_service_join returned: %s\n", result_name(r));
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────
    g_stop_requested = true;
    if (events.joinable()) events.join();

    printf("[cxr-service] Destroying service...\n");
    nv_cxr_service_destroy(g_service);
    g_service = nullptr;

    printf("[cxr-service] Shutdown complete.\n");
    return 0;
}