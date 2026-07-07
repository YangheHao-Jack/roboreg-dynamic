// xr_intrinsics_layer.cpp
//
// OpenXR API layer that sniffs per-eye FoV + recommended resolution from
// Isaac Sim's XR session and publishes them to Python via a Unix socket.
//
// Architecture:
//   Isaac Sim  →  libopenxr_loader.so.1  →  THIS LAYER  →  libopenxr_cloudxr.so
//
// Intercepts:
//   - xrEnumerateViewConfigurationViews  → per-eye recommendedImageRectWidth/Height
//   - xrLocateViews                      → per-eye XrFovf (angleLeft/Right/Up/Down)
//
// Output: newline-delimited JSON on Unix socket /tmp/xr_intrinsics.sock
// Each frame's XrView values are written as one JSON line. A Python consumer
// connects to the socket, reads lines, converts XrFovf→K, and uses them.
//
// Build (see CMakeLists.txt):
//   cmake -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build
//
// Register:
//   export XR_API_LAYER_PATH=$PWD/build
//   export XR_ENABLE_API_LAYERS=XR_APILAYER_INTRINSICS_SNIFF
//
// Then run Isaac Sim normally. The layer auto-creates the socket and
// streams values once xrLocateViews starts firing.

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// ──────────────────────────────────────────────────────────────────────────────
// Globals: dispatch table for the "next" layer (the runtime in our case).
// ──────────────────────────────────────────────────────────────────────────────
namespace {

constexpr const char* LAYER_NAME = "XR_APILAYER_INTRINSICS_SNIFF";
constexpr const char* SOCKET_PATH = "/tmp/xr_intrinsics.sock";

struct NextDispatch {
    PFN_xrGetInstanceProcAddr                   GetInstanceProcAddr = nullptr;
    PFN_xrEnumerateViewConfigurationViews       EnumerateViewConfigurationViews = nullptr;
    PFN_xrLocateViews                           LocateViews = nullptr;
    PFN_xrDestroyInstance                       DestroyInstance = nullptr;
};

// One dispatch per XrInstance.
std::mutex                                g_instances_mu;
std::unordered_map<XrInstance, NextDispatch> g_instances;

// Socket state
std::mutex        g_sock_mu;
int               g_server_fd = -1;  // listening socket (published side)
int               g_client_fd = -1;  // accepted client (consumer connected)
std::atomic<bool> g_client_connected{false};

// Per-eye cached resolution from xrEnumerateViewConfigurationViews.
struct EyeRes { uint32_t width = 0; uint32_t height = 0; };
std::mutex                                   g_res_mu;
std::vector<EyeRes>                          g_recommended_views;
std::atomic<int64_t>                         g_res_seq{0};

// ──────────────────────────────────────────────────────────────────────────────
// Socket publisher — one Unix domain datagram-style stream.
// Accepts a single client at a time (Python consumer). Newline-delimited JSON.
// ──────────────────────────────────────────────────────────────────────────────

void socket_init() {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    if (g_server_fd >= 0) return;

    unlink(SOCKET_PATH);  // remove stale

    g_server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        fprintf(stderr, "[xr-layer] socket() failed: %s\n", strerror(errno));
        return;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(g_server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "[xr-layer] bind(%s) failed: %s\n", SOCKET_PATH, strerror(errno));
        close(g_server_fd);
        g_server_fd = -1;
        return;
    }
    if (listen(g_server_fd, 1) < 0) {
        fprintf(stderr, "[xr-layer] listen() failed: %s\n", strerror(errno));
        close(g_server_fd);
        g_server_fd = -1;
        return;
    }

    // Non-blocking so accept() in the hot path doesn't stall rendering
    int flags = fcntl(g_server_fd, F_GETFL, 0);
    fcntl(g_server_fd, F_SETFL, flags | O_NONBLOCK);

    fprintf(stderr, "[xr-layer] listening on %s\n", SOCKET_PATH);
}

void socket_try_accept() {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    if (g_server_fd < 0 || g_client_fd >= 0) return;
    int fd = accept(g_server_fd, nullptr, nullptr);
    if (fd >= 0) {
        g_client_fd = fd;
        g_client_connected = true;
        fprintf(stderr, "[xr-layer] consumer connected\n");
    }
}

void socket_send_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    if (g_client_fd < 0) return;
    ssize_t n = send(g_client_fd, line.data(), line.size(), MSG_NOSIGNAL);
    if (n < 0) {
        fprintf(stderr, "[xr-layer] consumer disconnected (send errno=%d)\n", errno);
        close(g_client_fd);
        g_client_fd = -1;
        g_client_connected = false;
    }
}

void socket_close_all() {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    if (g_client_fd >= 0) { close(g_client_fd); g_client_fd = -1; }
    if (g_server_fd >= 0) { close(g_server_fd); g_server_fd = -1; }
    unlink(SOCKET_PATH);
}

// ──────────────────────────────────────────────────────────────────────────────
// Intercepted OpenXR calls.
// ──────────────────────────────────────────────────────────────────────────────

XrResult XRAPI_CALL layer_EnumerateViewConfigurationViews(
    XrInstance                  instance,
    XrSystemId                  systemId,
    XrViewConfigurationType     viewConfigurationType,
    uint32_t                    viewCapacityInput,
    uint32_t*                   viewCountOutput,
    XrViewConfigurationView*    views)
{
    NextDispatch nd;
    {
        std::lock_guard<std::mutex> lk(g_instances_mu);
        auto it = g_instances.find(instance);
        if (it == g_instances.end() || !it->second.EnumerateViewConfigurationViews)
            return XR_ERROR_FUNCTION_UNSUPPORTED;
        nd = it->second;
    }

    XrResult r = nd.EnumerateViewConfigurationViews(
        instance, systemId, viewConfigurationType,
        viewCapacityInput, viewCountOutput, views);

    // Capture when the runtime actually fills the array
    if (r == XR_SUCCESS && views && viewCapacityInput > 0 && viewCountOutput) {
        std::lock_guard<std::mutex> lk(g_res_mu);
        g_recommended_views.clear();
        g_recommended_views.reserve(*viewCountOutput);
        for (uint32_t i = 0; i < *viewCountOutput; ++i) {
            EyeRes e;
            e.width  = views[i].recommendedImageRectWidth;
            e.height = views[i].recommendedImageRectHeight;
            g_recommended_views.push_back(e);
        }
        g_res_seq.fetch_add(1);
        fprintf(stderr, "[xr-layer] recommended view sizes captured: %u views\n",
                *viewCountOutput);
        for (uint32_t i = 0; i < *viewCountOutput; ++i) {
            fprintf(stderr, "[xr-layer]   view[%u] %ux%u\n",
                    i, views[i].recommendedImageRectWidth,
                    views[i].recommendedImageRectHeight);
        }
    }
    return r;
}

XrResult XRAPI_CALL layer_LocateViews(
    XrSession                   session,
    const XrViewLocateInfo*     viewLocateInfo,
    XrViewState*                viewState,
    uint32_t                    viewCapacityInput,
    uint32_t*                   viewCountOutput,
    XrView*                     views)
{
    // Figure out which instance owns this session by asking the runtime.
    // Simplest path: walk our instance map (only one XR instance per process
    // in practice). Store the dispatch under the first (and only) instance.
    NextDispatch nd;
    {
        std::lock_guard<std::mutex> lk(g_instances_mu);
        if (g_instances.empty()) return XR_ERROR_HANDLE_INVALID;
        nd = g_instances.begin()->second;
    }
    if (!nd.LocateViews) return XR_ERROR_FUNCTION_UNSUPPORTED;

    XrResult r = nd.LocateViews(session, viewLocateInfo, viewState,
                                viewCapacityInput, viewCountOutput, views);

    if (r != XR_SUCCESS || !views || viewCapacityInput == 0 || !viewCountOutput)
        return r;
    uint32_t n = *viewCountOutput;
    if (n == 0) return r;

    // Lazy init socket + opportunistic accept (non-blocking; doesn't stall).
    socket_init();
    socket_try_accept();
    if (!g_client_connected.load()) return r;

    // Build a single JSON line for this frame.
    std::string out = "{\"seq\":";
    out += std::to_string(g_res_seq.load());
    out += ",\"time\":";
    out += std::to_string(viewLocateInfo ? (long long)viewLocateInfo->displayTime : 0LL);
    out += ",\"views\":[";
    for (uint32_t i = 0; i < n; ++i) {
        if (i) out += ",";
        const XrView& v = views[i];
        const XrFovf&   f = v.fov;
        const XrPosef&  p = v.pose;

        uint32_t w = 0, h = 0;
        {
            std::lock_guard<std::mutex> lk(g_res_mu);
            if (i < g_recommended_views.size()) {
                w = g_recommended_views[i].width;
                h = g_recommended_views[i].height;
            }
        }

        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\"eye\":%u,\"w\":%u,\"h\":%u,"
            "\"angleLeft\":%.9f,\"angleRight\":%.9f,"
            "\"angleUp\":%.9f,\"angleDown\":%.9f,"
            "\"px\":%.9f,\"py\":%.9f,\"pz\":%.9f,"
            "\"qx\":%.9f,\"qy\":%.9f,\"qz\":%.9f,\"qw\":%.9f}",
            i, w, h,
            f.angleLeft, f.angleRight, f.angleUp, f.angleDown,
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
        out += buf;
    }
    out += "]}\n";
    socket_send_line(out);
    return r;
}

XrResult XRAPI_CALL layer_DestroyInstance(XrInstance instance)
{
    PFN_xrDestroyInstance next = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_instances_mu);
        auto it = g_instances.find(instance);
        if (it != g_instances.end()) {
            next = it->second.DestroyInstance;
            g_instances.erase(it);
        }
    }
    XrResult r = next ? next(instance) : XR_SUCCESS;
    if (g_instances.empty()) socket_close_all();
    return r;
}

// ──────────────────────────────────────────────────────────────────────────────
// Dispatcher — returns our wrappers for intercepted calls, otherwise forwards.
// ──────────────────────────────────────────────────────────────────────────────

XrResult XRAPI_CALL layer_GetInstanceProcAddr(
    XrInstance instance, const char* name, PFN_xrVoidFunction* function)
{
    if (!function) return XR_ERROR_VALIDATION_FAILURE;

    // Our overrides
    if (std::strcmp(name, "xrEnumerateViewConfigurationViews") == 0) {
        *function = reinterpret_cast<PFN_xrVoidFunction>(&layer_EnumerateViewConfigurationViews);
        return XR_SUCCESS;
    }
    if (std::strcmp(name, "xrLocateViews") == 0) {
        *function = reinterpret_cast<PFN_xrVoidFunction>(&layer_LocateViews);
        return XR_SUCCESS;
    }
    if (std::strcmp(name, "xrDestroyInstance") == 0) {
        *function = reinterpret_cast<PFN_xrVoidFunction>(&layer_DestroyInstance);
        return XR_SUCCESS;
    }

    // Everything else: forward to runtime
    PFN_xrGetInstanceProcAddr nextGetProc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_instances_mu);
        auto it = g_instances.find(instance);
        if (it == g_instances.end()) return XR_ERROR_HANDLE_INVALID;
        nextGetProc = it->second.GetInstanceProcAddr;
    }
    if (!nextGetProc) return XR_ERROR_HANDLE_INVALID;
    return nextGetProc(instance, name, function);
}

// ──────────────────────────────────────────────────────────────────────────────
// xrCreateApiLayerInstance: called by the loader when the app creates an
// XrInstance. We forward to the next link and capture its proc-addr table.
// ──────────────────────────────────────────────────────────────────────────────

XrResult XRAPI_CALL layer_CreateApiLayerInstance(
    const XrInstanceCreateInfo*        info,
    const XrApiLayerCreateInfo*        apiLayerInfo,
    XrInstance*                        instance)
{
    if (!apiLayerInfo || !apiLayerInfo->nextInfo) return XR_ERROR_INITIALIZATION_FAILED;

    // Forward CreateInstance through the rest of the chain → runtime.
    PFN_xrGetInstanceProcAddr nextGetProc = apiLayerInfo->nextInfo->nextGetInstanceProcAddr;
    PFN_xrCreateApiLayerInstance nextCreate = apiLayerInfo->nextInfo->nextCreateApiLayerInstance;
    if (!nextGetProc || !nextCreate) return XR_ERROR_INITIALIZATION_FAILED;

    // Advance the chain pointer for the next link.
    XrApiLayerCreateInfo nextLayerInfo = *apiLayerInfo;
    nextLayerInfo.nextInfo = apiLayerInfo->nextInfo->next;

    XrResult r = nextCreate(info, &nextLayerInfo, instance);
    if (r != XR_SUCCESS) return r;

    // Cache the next-link dispatch for this XrInstance.
    NextDispatch nd;
    nd.GetInstanceProcAddr = nextGetProc;

#define LOAD_NEXT(NAME)                                                              \
    do {                                                                             \
        PFN_xrVoidFunction fn = nullptr;                                             \
        if (nextGetProc(*instance, "xr" #NAME, &fn) == XR_SUCCESS) {                 \
            nd.NAME = reinterpret_cast<PFN_xr##NAME>(fn);                            \
        }                                                                            \
    } while (0)

    LOAD_NEXT(EnumerateViewConfigurationViews);
    LOAD_NEXT(LocateViews);
    LOAD_NEXT(DestroyInstance);

#undef LOAD_NEXT

    {
        std::lock_guard<std::mutex> lk(g_instances_mu);
        g_instances[*instance] = nd;
    }
    fprintf(stderr, "[xr-layer] XrInstance created; interceptors armed\n");

    // ── DIAGNOSTIC: dump enabled + available extensions ─────────────────────
    // Helps us understand whether foveation / lens-correction / warp
    // extensions are active in the CloudXR + Isaac Sim session, which would
    // explain visible curvature in captured swapchain frames.
    auto is_interesting = [](const char* name) -> const char* {
        const struct { const char* sub; const char* tag; } pats[] = {
            {"foveat",    "FOVEATION"},
            {"Foveat",    "FOVEATION"},
            {"FOVEAT",    "FOVEATION"},
            {"distort",   "DISTORT"},
            {"Distort",   "DISTORT"},
            {"warp",      "WARP"},
            {"Warp",      "WARP"},
            {"WARP",      "WARP"},
            {"lens",      "LENS"},
            {"Lens",      "LENS"},
            {"NVX",       "NVX"},
            {"_NV_",      "NV"},
            {"_NV1",      "NV"},
            {"_FB_",      "FB"},
            {"_VARJO_",   "VARJO"},
            {"_META_",    "META"},
        };
        for (auto& p : pats) {
            if (std::strstr(name, p.sub)) return p.tag;
        }
        return nullptr;
    };

    // 1) What did Isaac Sim REQUEST?
    fprintf(stderr, "[xr-layer] app-requested extensions (%u):\n",
            (unsigned)info->enabledExtensionCount);
    for (uint32_t i = 0; i < info->enabledExtensionCount; ++i) {
        const char* name = info->enabledExtensionNames[i];
        const char* tag = is_interesting(name);
        if (tag) fprintf(stderr, "  [%s] %s\n", tag, name);
        else      fprintf(stderr, "        %s\n", name);
    }

    // 2) What does the runtime OFFER? (asks the loader, not the layer chain)
    PFN_xrVoidFunction fn = nullptr;
    if (nextGetProc(*instance, "xrEnumerateInstanceExtensionProperties", &fn)
        == XR_SUCCESS && fn) {
        auto pEnum = reinterpret_cast<PFN_xrEnumerateInstanceExtensionProperties>(fn);
        uint32_t count = 0;
        if (pEnum(nullptr, 0, &count, nullptr) == XR_SUCCESS && count > 0) {
            std::vector<XrExtensionProperties> props(count,
                {XR_TYPE_EXTENSION_PROPERTIES, nullptr, {0}, 0});
            if (pEnum(nullptr, count, &count, props.data()) == XR_SUCCESS) {
                fprintf(stderr, "[xr-layer] runtime-available extensions (%u):\n",
                        (unsigned)count);
                // Print interesting ones first (so they're visible in a big list)
                for (auto& p : props) {
                    const char* tag = is_interesting(p.extensionName);
                    if (tag) fprintf(stderr, "  [%s] %s (v%u)\n", tag,
                                     p.extensionName, p.extensionVersion);
                }
                for (auto& p : props) {
                    if (!is_interesting(p.extensionName))
                        fprintf(stderr, "        %s (v%u)\n",
                                p.extensionName, p.extensionVersion);
                }
            }
        }
    }
    fflush(stderr);

    // Init socket early so consumer can connect even before first xrLocateViews.
    socket_init();

    return XR_SUCCESS;
}

} // anon namespace

// ──────────────────────────────────────────────────────────────────────────────
// Loader negotiation entry point — exported with default visibility.
// ──────────────────────────────────────────────────────────────────────────────

extern "C" __attribute__((visibility("default")))
XrResult XRAPI_CALL xrNegotiateLoaderApiLayerInterface(
    const XrNegotiateLoaderInfo* loaderInfo,
    const char*                  layerName,
    XrNegotiateApiLayerRequest*  apiLayerRequest)
{
    if (!loaderInfo || !apiLayerRequest || !layerName) return XR_ERROR_INITIALIZATION_FAILED;
    if (loaderInfo->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO) return XR_ERROR_INITIALIZATION_FAILED;
    if (apiLayerRequest->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST) return XR_ERROR_INITIALIZATION_FAILED;
    if (std::strcmp(layerName, LAYER_NAME) != 0) return XR_ERROR_INITIALIZATION_FAILED;

    apiLayerRequest->layerInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
    apiLayerRequest->layerApiVersion       = XR_CURRENT_API_VERSION;
    apiLayerRequest->getInstanceProcAddr       = &layer_GetInstanceProcAddr;
    apiLayerRequest->createApiLayerInstance    = &layer_CreateApiLayerInstance;
    return XR_SUCCESS;
}
