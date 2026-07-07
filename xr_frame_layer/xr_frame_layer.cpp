// xr_frame_layer.cpp — OpenXR API layer that streams stereo color
// swapchain pixels to a UNIX socket on demand. Hooks xrEndFrame to
// run a continuous capture into a 3-slot staging ring; a dedicated
// worker thread ships the freshest ready slot in response to each
// GET on /tmp/xr_frames.sock.
//
// Architecture, wire protocol, concurrency model, HOST_CACHED memory
// rationale, and build/deploy instructions: see
// XR_FRAME_LAYER_ARCHITECTURE.md.

#include <openxr/openxr.h>
#include <vulkan/vulkan_core.h>
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr_platform.h>
#include <openxr/openxr_loader_negotiation.h>
#include <XR_NV_opaque_data_channel.h>   // XR_NV_OPAQUE_DATA_CHANNEL_EXTENSION_NAME, NV types
#include "opaque_data_sender.h"          // OpaqueSender_* (channel create / send)

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ioctl.h>

#include <unistd.h>
#include <errno.h>

namespace {

constexpr const char* LAYER_NAME    = "XR_APILAYER_FRAME_GRAB";
constexpr const char* SOCKET_PATH   = "/tmp/xr_frames.sock";
constexpr uint32_t    WIRE_MAGIC    = 0x52465258u;   // "XRFR" LE
constexpr uint32_t    WIRE_VERSION  = 2u;
constexpr uint32_t    WIRE_FMT_SRGB = 1u;
constexpr uint32_t    EXPORT_MAGIC  = 0x52465845u;   // "EXFR" LE — FD-export tail

// Overlay receive (Piece 2): pipeline → layer. RGBA per-eye images that the
// layer uploads into its own swapchain and injects as a composition layer.
constexpr const char* OVERLAY_SOCKET_PATH = "/tmp/xr_overlay.sock";
constexpr uint32_t    OVERLAY_MAGIC       = 0x314C564Fu;  // "OVL1" LE
constexpr uint8_t     OVERLAY_FMT_RGBA8   = 0u;
constexpr uint8_t     OVERLAY_FMT_MONO8   = 1u;
#pragma pack(push, 1)
struct OverlayHeader {
    uint32_t magic;
    uint32_t seq;
    uint16_t w, h;
    uint8_t  eye_count;
    uint8_t  format;
    uint64_t capture_ts_us;
    float    fov[2][4];   // [eye][angleLeft, angleRight, angleUp, angleDown]
};
#pragma pack(pop)


void log(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[xr-frame] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(ap);
}

// ───── OpenXR dispatch ─────────────────────────────────────────────────────
struct XrDispatch {
    PFN_xrGetInstanceProcAddr           GetInstanceProcAddr           = nullptr;
    PFN_xrGetSystem                     GetSystem                     = nullptr;
    PFN_xrDestroyInstance               DestroyInstance               = nullptr;
    PFN_xrCreateSession                 CreateSession                 = nullptr;
    PFN_xrDestroySession                DestroySession                = nullptr;
    PFN_xrCreateSwapchain               CreateSwapchain               = nullptr;
    PFN_xrDestroySwapchain              DestroySwapchain              = nullptr;
    PFN_xrEnumerateSwapchainImages      EnumerateSwapchainImages      = nullptr;
    PFN_xrAcquireSwapchainImage         AcquireSwapchainImage         = nullptr;
    PFN_xrReleaseSwapchainImage         ReleaseSwapchainImage         = nullptr;
    PFN_xrEndFrame                      EndFrame                      = nullptr;
    PFN_xrBeginSession                  BeginSession                  = nullptr;
    PFN_xrEndSession                    EndSession                    = nullptr;
    PFN_xrPollEvent                     PollEvent                     = nullptr;
    PFN_xrWaitSwapchainImage            WaitSwapchainImage            = nullptr;
};
std::mutex                                   g_mu;
std::unordered_map<XrInstance, XrDispatch>   g_inst;
std::atomic<bool>                            g_opaque_enabled{false};  // XR_OPAQUE_DATA_CHANNEL=1
std::atomic<bool>                            g_opaque_created{false};  // channel created once
std::atomic<int>                             g_endframe_count{0};

struct SessionInfo {
    bool             handles_captured = false;
    VkInstance       vkInstance       = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice         vkDevice         = VK_NULL_HANDLE;
    uint32_t         queueFamily      = 0;
    uint32_t         queueIndex       = 0;
    // The transfer queue + command pool live here; per-slot fences and
    // command buffers live on g_staging[] below.
    VkQueue          vkQueue          = VK_NULL_HANDLE;
    VkCommandPool    cmdPool          = VK_NULL_HANDLE;
};
std::mutex   g_sess_mu;
SessionInfo  g_sess;

// ───── Per-eye pose ───────────────────────────────────────────────────────
// Embedded in each StagingSlot (below) and cached globally (further down)
// from xrEndFrame's XrCompositionLayerProjection views. `valid` is false
// until the first frame supplies a projection layer.
struct EyePose {
    bool     valid = false;
    float    px = 0, py = 0, pz = 0;
    float    qw = 1, qx = 0, qy = 0, qz = 0;
};

// Persisted projection-layer parameters, captured whenever the app submits a
// XR_TYPE_COMPOSITION_LAYER_PROJECTION. The overlay is injected EVERY frame
// from this state — even on frames where the app exposes no projection layer
// (e.g. foveation-warped frames) — so the overlay never blinks out. On frames
// that DO carry a projection layer this is refreshed, so it is at worst one
// frame stale (no visible swim).
struct PersistProj {
    bool     seen  = false;
    XrSpace  space = XR_NULL_HANDLE;
    uint32_t nv    = 0;
    XrPosef  pose[2];
    XrFovf   fov[2];
};

// ───── Staging slot ring (N=3, continuous capture) ────────────────────────
// Render thread overwrites the stalest slot every xrEndFrame; send worker
// picks the freshest signalled slot per GET. See architecture doc for the
// race-protection / read_lock semantics.
struct StagingSlot {
    VkBuffer          buf       = VK_NULL_HANDLE;
    VkDeviceMemory    mem       = VK_NULL_HANDLE;
    void*             mapped    = nullptr;     // host-visible + coherent + cached
    // ── FD-export path (XR_FRAME_EXPORT_FD=1): a parallel DEVICE_LOCAL buffer
    // whose memory is exported as an OpaqueFd for zero-copy CUDA import. The
    // host buffer above is left fully intact; this is additive.
    VkBuffer          exp_buf   = VK_NULL_HANDLE;
    VkDeviceMemory    exp_mem   = VK_NULL_HANDLE;
    int               mem_fd    = -1;          // exported OpaqueFd (owned)
    VkDeviceSize      size      = 0;           // bytes (= bpe * 2)
    VkFence           fence     = VK_NULL_HANDLE;
    VkCommandBuffer   cmdBuf    = VK_NULL_HANDLE;

    // seq: monotonic frame number; 0 = never filled.
    // read_lock: held briefly during render-side write or worker-side send.
    std::atomic<uint64_t> seq{0};
    std::atomic<bool>     read_lock{false};
    uint32_t              w[2]   = {0, 0};
    uint32_t              h[2]   = {0, 0};
    size_t                bpe    = 0;          // bytes per eye
    EyePose               pose[2]{};
};
constexpr size_t          N_STAGING = 3;
StagingSlot               g_staging[N_STAGING];
std::atomic<uint64_t>     g_frame_seq{0};      // monotonic seq for slot assignment
// ── FD-export globals (XR_FRAME_EXPORT_FD=1) ────────────────────────────
bool                      g_export_fd     = false;          // set from env once
VkSemaphore               g_export_sem    = VK_NULL_HANDLE;  // exported timeline
int                       g_export_sem_fd = -1;

// ───── Vulkan ──────────────────────────────────────────────────────────────
struct VkFns {
    void*  libvulkan = nullptr;
    PFN_vkGetInstanceProcAddr            GetInstanceProcAddr            = nullptr;
    PFN_vkGetDeviceProcAddr              GetDeviceProcAddr              = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetDeviceQueue                 GetDeviceQueue                 = nullptr;
    PFN_vkCreateCommandPool              CreateCommandPool              = nullptr;
    PFN_vkDestroyCommandPool             DestroyCommandPool             = nullptr;
    PFN_vkAllocateCommandBuffers         AllocateCommandBuffers         = nullptr;
    PFN_vkBeginCommandBuffer             BeginCommandBuffer             = nullptr;
    PFN_vkEndCommandBuffer               EndCommandBuffer               = nullptr;
    PFN_vkResetCommandBuffer             ResetCommandBuffer             = nullptr;
    PFN_vkCmdPipelineBarrier             CmdPipelineBarrier             = nullptr;
    PFN_vkCmdCopyImageToBuffer           CmdCopyImageToBuffer           = nullptr;
    PFN_vkCmdCopyBufferToImage           CmdCopyBufferToImage           = nullptr;
    PFN_vkQueueSubmit                    QueueSubmit                    = nullptr;
    PFN_vkCreateFence                    CreateFence                    = nullptr;
    PFN_vkDestroyFence                   DestroyFence                   = nullptr;
    PFN_vkWaitForFences                  WaitForFences                  = nullptr;
    PFN_vkResetFences                    ResetFences                    = nullptr;
    PFN_vkCreateBuffer                   CreateBuffer                   = nullptr;
    PFN_vkDestroyBuffer                  DestroyBuffer                  = nullptr;
    PFN_vkGetBufferMemoryRequirements    GetBufferMemoryRequirements    = nullptr;
    PFN_vkAllocateMemory                 AllocateMemory                 = nullptr;
    PFN_vkFreeMemory                     FreeMemory                     = nullptr;
    PFN_vkBindBufferMemory               BindBufferMemory               = nullptr;
    PFN_vkMapMemory                      MapMemory                      = nullptr;
    PFN_vkUnmapMemory                    UnmapMemory                    = nullptr;
    PFN_vkDeviceWaitIdle                 DeviceWaitIdle                 = nullptr;
    // FD-export path (XR_FRAME_EXPORT_FD=1). Loaded non-fatally.
    PFN_vkGetMemoryFdKHR                 GetMemoryFdKHR                 = nullptr;
    PFN_vkGetSemaphoreFdKHR              GetSemaphoreFdKHR              = nullptr;
    PFN_vkCreateSemaphore                CreateSemaphore                = nullptr;
    PFN_vkDestroySemaphore               DestroySemaphore               = nullptr;
    bool loaded = false;
};
VkFns g_vk;
std::atomic<bool> g_vk_load_attempted{false};

// ───── Color swapchain tracking ────────────────────────────────────────────
struct ColorSwapchain {
    XrSwapchain          handle = XR_NULL_HANDLE;
    int64_t              format = 0;
    uint32_t             width  = 0;
    uint32_t             height = 0;
    std::vector<VkImage> images;
    int                  last_acquired = -1;
    int                  last_released = -1;
};
std::mutex                              g_sc_mu;
std::vector<ColorSwapchain>             g_color_scs;
std::unordered_map<XrSwapchain, size_t> g_sc_idx;

// ───── Per-eye pose tracking ───────────────────────────────────────────────
// EyePose itself is defined above (StagingSlot embeds it). These globals
// are updated every xrEndFrame from the XrCompositionLayerProjection's
// views; `valid` is false until the first frame supplies a projection
// layer.
std::mutex            g_pose_mu;
std::vector<EyePose>  g_eye_poses(2);   // index 0 = left, 1 = right (by view order)
PersistProj           g_pproj;           // last-known projection layer params (g_pose_mu)

// ── Capture-time eye-pose history (overlay reprojection / timewarp) ────────
// The overlay arrives tagged with capture_ts_us — CLOCK_REALTIME µs stamped by
// test_cloudxr at encode time (int(time.time()*1e6)) — the same wall clock we
// read here. To let the runtime timewarp the overlay correctly we must inject
// it with the eye pose its content was RENDERED for (capture time), not the
// current pose; otherwise the ghost swims under head motion. We keep a ring of
// (realtime_us → eye pose) recorded each xrEndFrame and pick the entry nearest
// the overlay's capture_ts_us. Frame spacing (≥33 ms) ≫ the layer→encoder
// offset (~few ms), so nearest-match reliably lands on the right frame.
// Guarded by g_pose_mu (same lock as g_eye_poses).
struct PoseStamp { uint64_t ts_us = 0; EyePose pose[2]{}; };
constexpr size_t POSE_RING_N = 90;          // ~3 s @30fps, ~6 s @15fps
PoseStamp        g_pose_ring[POSE_RING_N];
size_t           g_pose_ring_head = 0;      // next write slot (under g_pose_mu)

static uint64_t now_realtime_us() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

// Caller must hold g_pose_mu. Fills out[2] with the eye pose nearest ts_us.
static bool pose_ring_lookup(uint64_t ts_us, EyePose out[2]) {
    uint64_t best_d = ~0ull; int best = -1;
    for (size_t i = 0; i < POSE_RING_N; ++i) {
        if (g_pose_ring[i].ts_us == 0 || !g_pose_ring[i].pose[0].valid) continue;
        uint64_t t = g_pose_ring[i].ts_us;
        uint64_t d = (t > ts_us) ? (t - ts_us) : (ts_us - t);
        if (d < best_d) { best_d = d; best = (int)i; }
    }
    if (best < 0) return false;
    out[0] = g_pose_ring[best].pose[0];
    out[1] = g_pose_ring[best].pose[1];
    return true;
}

static XrPosef eyepose_to_xrposef(const EyePose& e) {
    XrPosef p{};
    p.position.x = e.px; p.position.y = e.py; p.position.z = e.pz;
    p.orientation.x = e.qx; p.orientation.y = e.qy;
    p.orientation.z = e.qz; p.orientation.w = e.qw;
    return p;
}

// ───── Socket ──────────────────────────────────────────────────────────────
std::mutex        g_sock_mu;
int               g_listen_fd  = -1;
int               g_client_fd  = -1;
std::atomic<bool> g_socket_up{false};

XrDispatch get_dispatch(XrInstance instance) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_inst.find(instance);
    return (it != g_inst.end()) ? it->second : XrDispatch{};
}
XrDispatch get_first_dispatch() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_inst.empty()) return XrDispatch{};
    return g_inst.begin()->second;
}

// ───── Vulkan loader ───────────────────────────────────────────────────────
bool load_vulkan_fns() {
    log("STEP vulkan_load begin");
    g_vk.libvulkan = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!g_vk.libvulkan) g_vk.libvulkan = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!g_vk.libvulkan) { log("  dlopen FAILED: %s", dlerror()); return false; }
    g_vk.GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        dlsym(g_vk.libvulkan, "vkGetInstanceProcAddr"));
    if (!g_vk.GetInstanceProcAddr) { log("  dlsym(vkGetInstanceProcAddr) FAILED"); return false; }

    auto inst = g_sess.vkInstance;
    auto dev  = g_sess.vkDevice;
#define LI(NAME)                                                                \
    g_vk.NAME = reinterpret_cast<PFN_vk##NAME>(g_vk.GetInstanceProcAddr(inst, "vk" #NAME)); \
    if (!g_vk.NAME) { log("  MISSING vk" #NAME); return false; }
    LI(GetDeviceProcAddr);
    LI(GetPhysicalDeviceMemoryProperties);
#undef LI
#define LD(NAME)                                                                \
    g_vk.NAME = reinterpret_cast<PFN_vk##NAME>(g_vk.GetDeviceProcAddr(dev, "vk" #NAME)); \
    if (!g_vk.NAME) { log("  MISSING vk" #NAME); return false; }
    LD(GetDeviceQueue);
    LD(CreateCommandPool); LD(DestroyCommandPool);
    LD(AllocateCommandBuffers);
    LD(BeginCommandBuffer); LD(EndCommandBuffer); LD(ResetCommandBuffer);
    LD(CmdPipelineBarrier); LD(CmdCopyImageToBuffer);
    LD(CmdCopyBufferToImage);
    LD(QueueSubmit);
    LD(CreateFence); LD(DestroyFence); LD(WaitForFences); LD(ResetFences);
    LD(CreateBuffer); LD(DestroyBuffer);
    LD(GetBufferMemoryRequirements);
    LD(AllocateMemory); LD(FreeMemory);
    LD(BindBufferMemory); LD(MapMemory); LD(UnmapMemory);
    LD(DeviceWaitIdle);
#undef LD
    // FD-export PFNs: only needed when XR_FRAME_EXPORT_FD=1, so load them
    // non-fatally — a build/runtime without them still runs the host path.
    g_vk.GetMemoryFdKHR    = reinterpret_cast<PFN_vkGetMemoryFdKHR>(g_vk.GetDeviceProcAddr(dev, "vkGetMemoryFdKHR"));
    g_vk.GetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(g_vk.GetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR"));
    g_vk.CreateSemaphore   = reinterpret_cast<PFN_vkCreateSemaphore>(g_vk.GetDeviceProcAddr(dev, "vkCreateSemaphore"));
    g_vk.DestroySemaphore  = reinterpret_cast<PFN_vkDestroySemaphore>(g_vk.GetDeviceProcAddr(dev, "vkDestroySemaphore"));
    g_vk.loaded = true;
    log("STEP vulkan_load done");
    return true;
}

// ───── Socket helpers ──────────────────────────────────────────────────────
// The send worker owns the listen/client fds for both accept and recv —
// see send_worker_run() below. socket_send_all is the only function
// callable from outside the worker thread (it's used to flush headers,
// poses, and pixel payloads to the connected client).
void send_worker_start();
void send_worker_stop();

void socket_init_if_needed() {
    if (g_socket_up.load()) return;
    std::lock_guard<std::mutex> lk(g_sock_mu);
    if (g_listen_fd >= 0) return;
    log("STEP socket_init begin");
    unlink(SOCKET_PATH);
    g_listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_listen_fd < 0) { log("socket(): %s", strerror(errno)); return; }
    sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    if (bind(g_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        log("bind: %s", strerror(errno));
        close(g_listen_fd); g_listen_fd = -1; return;
    }
    if (listen(g_listen_fd, 1) < 0) {
        log("listen: %s", strerror(errno));
        close(g_listen_fd); g_listen_fd = -1; return;
    }
    int fl = fcntl(g_listen_fd, F_GETFL, 0);
    fcntl(g_listen_fd, F_SETFL, fl | O_NONBLOCK);
    g_socket_up = true;
    log("STEP socket_init done: %s", SOCKET_PATH);
    send_worker_start();
}

bool socket_send_all(const void* data, size_t n) {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    if (g_client_fd < 0) return false;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    size_t left = n;
    while (left > 0) {
        ssize_t sent = send(g_client_fd, p, left, MSG_NOSIGNAL);
        if (sent > 0) { p += sent; left -= sent; continue; }
        if (sent == 0) {
            log("send: peer closed");
            close(g_client_fd); g_client_fd = -1; return false;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            pollfd pfd; pfd.fd = g_client_fd; pfd.events = POLLOUT;
            int pr = poll(&pfd, 1, 5000);
            if (pr <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
                log("send: poll timeout");
                close(g_client_fd); g_client_fd = -1; return false;
            }
            continue;
        }
        log("send: %s", strerror(errno));
        close(g_client_fd); g_client_fd = -1; return false;
    }
    return true;
}

void socket_close_all() {
    send_worker_stop();   // join worker before tearing down the socket
    std::lock_guard<std::mutex> lk(g_sock_mu);
    if (g_client_fd >= 0) { close(g_client_fd); g_client_fd = -1; }
    if (g_listen_fd >= 0) { close(g_listen_fd); g_listen_fd = -1; }
    unlink(SOCKET_PATH);
    g_socket_up = false;
}

// ───── Send worker ────────────────────────────────────────────────────────
// Owns the UNIX socket. Polls listen_fd for accept and client_fd for GET;
// on each GET, ships the freshest signalled staging slot via
// socket_send_all(). Race protection: per-slot read_lock. See arch doc.

std::atomic<bool>           g_send_running{false};
std::thread                 g_send_thread;

// Find the freshest signalled slot, claim its read_lock, and ship it.
// Returns false if no slot was ready in time or the send failed.
bool send_latest_ready_slot();

void send_worker_run() {
    log("STEP send_worker started");
    char rbuf[64];
    while (g_send_running.load()) {
        // Snapshot socket state under the lock.
        int listen_fd, client_fd;
        {
            std::lock_guard<std::mutex> lk(g_sock_mu);
            listen_fd = g_listen_fd;
            client_fd = g_client_fd;
        }
        if (listen_fd < 0) {
            // Socket not initialised yet (Vulkan handles not captured).
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // Wait for either a new connection (if no client) or a GET
        // from the existing client. 100 ms timeout keeps the loop
        // responsive to g_send_running going false during shutdown.
        pollfd pfd{};
        pfd.fd     = (client_fd < 0) ? listen_fd : client_fd;
        pfd.events = POLLIN;
        int pr = poll(&pfd, 1, 100);
        if (pr <= 0) continue;        // timeout or error → recheck running

        if (client_fd < 0) {
            // accept-arm
            int fd = accept(listen_fd, nullptr, nullptr);
            if (fd < 0) continue;
            // Worker uses blocking semantics on the client fd — we want
            // recv to block on GET arrival, not spin.
            int fl = fcntl(fd, F_GETFL, 0);
            fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
            {
                std::lock_guard<std::mutex> lk(g_sock_mu);
                g_client_fd = fd;
            }
            log("STEP consumer connected (fd=%d)", fd);
            continue;
        }

        // GET-arm: read whatever's available. Coalesce multiple
        // newlines into multiple sends (one frame per '\n').
        ssize_t n = recv(client_fd, rbuf, sizeof(rbuf), 0);
        if (n <= 0) {
            // Disconnect or error.
            std::lock_guard<std::mutex> lk(g_sock_mu);
            if (g_client_fd == client_fd) {
                close(g_client_fd);
                g_client_fd = -1;
                log("consumer disconnected (recv n=%zd, errno=%d)",
                    n, (int)errno);
            }
            continue;
        }
        int n_gets = 0;
        for (ssize_t i = 0; i < n; ++i) if (rbuf[i] == '\n') ++n_gets;
        for (int g = 0; g < n_gets && g_send_running.load(); ++g) {
            send_latest_ready_slot();
        }
    }
    log("STEP send_worker exiting");
}

bool send_latest_ready_slot() {
    // Wait briefly for a ready slot if none is available yet (typical
    // on the very first capture after client connect — the first
    // EndFrame's Vulkan copy may not yet have completed).
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(50);

    StagingSlot* picked = nullptr;
    uint64_t     picked_seq = 0;
    int          picked_idx = -1;
    while (std::chrono::steady_clock::now() < deadline) {
        uint64_t best_seq = 0;
        int      best_idx = -1;
        for (size_t i = 0; i < N_STAGING; ++i) {
            StagingSlot& s = g_staging[i];
            if (s.read_lock.load(std::memory_order_acquire)) continue;
            uint64_t sq = s.seq.load(std::memory_order_acquire);
            if (sq == 0) continue;            // never filled
            // Non-blocking fence poll: timeout 0 ns = just check status
            if (g_vk.WaitForFences(g_sess.vkDevice, 1, &s.fence,
                                   VK_TRUE, 0) != VK_SUCCESS) continue;
            if (sq > best_seq) { best_seq = sq; best_idx = (int)i; }
        }
        if (best_idx < 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }
        // Try to atomically claim this slot. If render thread takes
        // it first, loop and pick another candidate.
        StagingSlot& s = g_staging[best_idx];
        bool expected = false;
        if (!s.read_lock.compare_exchange_strong(
                expected, true,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }
        picked     = &s;
        picked_seq = best_seq;
        picked_idx = best_idx;
        break;
    }
    if (!picked) {
        log("send_latest_ready_slot: no ready slot within 50 ms");
        return false;
    }

    // Sanity log: send loop timing on first few + every 30 frames.
    // Expect ~5 ms; a regression to ~120 ms means HOST_CACHED fell back.
    auto t_send_begin = std::chrono::steady_clock::now();
    static std::atomic<int> send_log_count{0};

    StagingSlot& slot = *picked;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(slot.mapped);
    // FD-export: pixels live in the exportable GPU buffer (the host buffer is
    // no longer copied), so advertise 0 host bytes and skip the payload.
    const uint32_t px_bytes = g_export_fd ? 0u : (uint32_t)slot.bpe;
    bool ok = true;
    for (uint32_t eye = 0; eye < 2 && ok; ++eye) {
        uint32_t hdr[7] = {
            WIRE_MAGIC, WIRE_VERSION, eye,
            slot.w[eye], slot.h[eye], WIRE_FMT_SRGB,
            px_bytes,
        };
        float pose_arr[7] = {
            slot.pose[eye].px, slot.pose[eye].py, slot.pose[eye].pz,
            slot.pose[eye].qw, slot.pose[eye].qx,
            slot.pose[eye].qy, slot.pose[eye].qz,
        };
        if (!socket_send_all(hdr,       sizeof(hdr)))       { ok = false; break; }
        if (!socket_send_all(pose_arr,  sizeof(pose_arr)))  { ok = false; break; }
        if (px_bytes &&
            !socket_send_all(base + slot.bpe * eye, slot.bpe)) { ok = false; break; }
    }
    // FD-export tail (Stage 2 validation): after the host pixels, append the
    // slot index + the three per-slot export fds + the timeline-semaphore fd,
    // so the consumer can CUDA-import the exportable buffers and byte-compare
    // against the host pixels it just received. Sent while read_lock is still
    // held, so picked_idx is valid for the consumer's read window.
    if (ok && g_export_fd) {
        uint64_t sq = slot.seq.load(std::memory_order_acquire);
        int32_t etail[8] = {
            (int32_t)EXPORT_MAGIC, picked_idx,
            (int32_t)(sq & 0xffffffffu), (int32_t)(sq >> 32),
            g_staging[0].mem_fd, g_staging[1].mem_fd, g_staging[2].mem_fd,
            g_export_sem_fd,
        };
        if (!socket_send_all(etail, sizeof(etail))) ok = false;
    }

    // Release the slot back to render-thread rotation regardless of
    // outcome. seq stays unchanged — the data on disk is still valid
    // and can be re-sent if needed.
    slot.read_lock.store(false, std::memory_order_release);

    if (ok) {
        auto t_send_end = std::chrono::steady_clock::now();
        double send_ms = std::chrono::duration<double, std::milli>(
            t_send_end - t_send_begin).count();
        int n = send_log_count.fetch_add(1);
        if (n < 8 || n % 30 == 0)
            log("send_worker: socket write loop took %.2f ms (frame %d, seq=%llu)",
                send_ms, n, (unsigned long long)picked_seq);
    }
    return ok;
}

void send_worker_start() {
    bool expected = false;
    if (!g_send_running.compare_exchange_strong(expected, true)) return;
    g_send_thread = std::thread(send_worker_run);
}

void send_worker_stop() {
    if (!g_send_running.exchange(false)) return;
    if (g_send_thread.joinable()) g_send_thread.join();
}


// ───── Capture pipeline ────────────────────────────────────────────────────
int find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties memProps{};
    g_vk.GetPhysicalDeviceMemoryProperties(g_sess.vkPhysicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & required) == required)
            return (int)i;
    }
    return -1;
}

// ── FD-export helpers (Stage 1: validate Vulkan-side export on this device) ──
// alloc_export_buffer: a DEVICE_LOCAL buffer whose memory is exportable as an
// OpaqueFd, exported via vkGetMemoryFdKHR. GPU-resident twin of the host
// staging buffer; CUDA imports the fd (cudaImportExternalMemory) for a
// zero-copy view. Additive — the host buffer/map are untouched.
bool alloc_export_buffer(StagingSlot& s, VkDeviceSize need) {
    if (!g_vk.GetMemoryFdKHR) { log("export: vkGetMemoryFdKHR not loaded"); return false; }
    if (s.exp_buf != VK_NULL_HANDLE) return true;            // already allocated

    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext       = &ext;
    bci.size        = need;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (g_vk.CreateBuffer(g_sess.vkDevice, &bci, nullptr, &s.exp_buf) != VK_SUCCESS) {
        log("export: CreateBuffer failed"); return false;
    }

    VkMemoryRequirements mr{};
    g_vk.GetBufferMemoryRequirements(g_sess.vkDevice, s.exp_buf, &mr);
    int idx = find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (idx < 0) { log("export: no DEVICE_LOCAL memory type"); return false; }

    // Dedicated allocation is recommended for exportable resources (some
    // drivers require it); harmless when merely optional.
    VkMemoryDedicatedAllocateInfo ded{};
    ded.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    ded.buffer = s.exp_buf;
    VkExportMemoryAllocateInfo exp{};
    exp.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exp.pNext       = &ded;
    exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext           = &exp;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = (uint32_t)idx;
    if (g_vk.AllocateMemory(g_sess.vkDevice, &ai, nullptr, &s.exp_mem) != VK_SUCCESS) {
        log("export: AllocateMemory failed"); return false;
    }
    if (g_vk.BindBufferMemory(g_sess.vkDevice, s.exp_buf, s.exp_mem, 0) != VK_SUCCESS) {
        log("export: BindBufferMemory failed"); return false;
    }

    VkMemoryGetFdInfoKHR gfi{};
    gfi.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gfi.memory     = s.exp_mem;
    gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (g_vk.GetMemoryFdKHR(g_sess.vkDevice, &gfi, &s.mem_fd) != VK_SUCCESS) {
        log("export: vkGetMemoryFdKHR failed"); return false;
    }
    return true;
}

// ensure_export_semaphore: one exported TIMELINE semaphore for the whole
// capture stream. The copy submit will later signal value == frame seq;
// CUDA waits on that value (cudaWaitExternalSemaphoresAsync).
bool ensure_export_semaphore() {
    if (g_export_sem != VK_NULL_HANDLE) return true;
    if (!g_vk.CreateSemaphore || !g_vk.GetSemaphoreFdKHR) {
        log("export: semaphore PFNs not loaded"); return false;
    }
    VkSemaphoreTypeCreateInfo tci{};
    tci.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue  = 0;
    VkExportSemaphoreCreateInfo esci{};
    esci.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    esci.pNext       = &tci;
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &esci;
    if (g_vk.CreateSemaphore(g_sess.vkDevice, &sci, nullptr, &g_export_sem) != VK_SUCCESS) {
        log("export: CreateSemaphore (timeline) failed"); return false;
    }
    VkSemaphoreGetFdInfoKHR gfi{};
    gfi.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    gfi.semaphore  = g_export_sem;
    gfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (g_vk.GetSemaphoreFdKHR(g_sess.vkDevice, &gfi, &g_export_sem_fd) != VK_SUCCESS) {
        log("export: vkGetSemaphoreFdKHR failed"); return false;
    }
    return true;
}

// Allocate / re-allocate all N_STAGING staging slots. Per-slot resources:
// buffer + host-coherent + cached memory + persistent map + fence +
// command buffer. Called lazily on the first capture, and re-called if
// the per-eye byte count grows (e.g. a panel-resolution change).
bool ensure_staging_slots(VkDeviceSize bytes_per_eye) {
    static bool s_export_checked = false;
    if (!s_export_checked) {
        const char* e = getenv("XR_FRAME_EXPORT_FD");
        g_export_fd = (e && e[0] == '1');
        s_export_checked = true;
        log("FD-export mode: %s", g_export_fd ? "ON (XR_FRAME_EXPORT_FD=1)" : "off");
    }
    const VkDeviceSize need = bytes_per_eye * 2;
    // Fast path: all slots already sized large enough.
    bool all_ready = true;
    for (size_t i = 0; i < N_STAGING; ++i) {
        if (g_staging[i].buf == VK_NULL_HANDLE || g_staging[i].size < need) {
            all_ready = false; break;
        }
    }
    if (all_ready) return true;

    // Slow path: (re)allocate every slot. Drain GPU first so we don't
    // pull resources out from under in-flight work.
    g_vk.DeviceWaitIdle(g_sess.vkDevice);
    for (size_t i = 0; i < N_STAGING; ++i) {
        StagingSlot& s = g_staging[i];
        if (s.mapped) { g_vk.UnmapMemory(g_sess.vkDevice, s.mem); s.mapped = nullptr; }
        if (s.buf != VK_NULL_HANDLE)   { g_vk.DestroyBuffer(g_sess.vkDevice, s.buf, nullptr);   s.buf = VK_NULL_HANDLE; }
        if (s.mem != VK_NULL_HANDLE)   { g_vk.FreeMemory(g_sess.vkDevice, s.mem, nullptr);     s.mem = VK_NULL_HANDLE; }
        if (s.fence != VK_NULL_HANDLE) { g_vk.DestroyFence(g_sess.vkDevice, s.fence, nullptr); s.fence = VK_NULL_HANDLE; }
        if (s.mem_fd >= 0)               { close(s.mem_fd); s.mem_fd = -1; }
        if (s.exp_buf != VK_NULL_HANDLE) { g_vk.DestroyBuffer(g_sess.vkDevice, s.exp_buf, nullptr); s.exp_buf = VK_NULL_HANDLE; }
        if (s.exp_mem != VK_NULL_HANDLE) { g_vk.FreeMemory(g_sess.vkDevice, s.exp_mem, nullptr);    s.exp_mem = VK_NULL_HANDLE; }
        // cmdBuf is freed when cmdPool is destroyed; clear handle.
        s.cmdBuf = VK_NULL_HANDLE;
        s.size = 0;
        s.read_lock.store(false);
        s.seq.store(0);
    }
    for (size_t i = 0; i < N_STAGING; ++i) {
        StagingSlot& s = g_staging[i];

        // Host-visible / host-coherent staging buffer (mapped persistently).
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = need;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (g_vk.CreateBuffer(g_sess.vkDevice, &bci, nullptr, &s.buf) != VK_SUCCESS) {
            log("ensure_staging_slots: CreateBuffer failed (slot %zu)", i); return false;
        }
        VkMemoryRequirements mr{};
        g_vk.GetBufferMemoryRequirements(g_sess.vkDevice, s.buf, &mr);
        // Prefer HOST_CACHED so CPU reads from slot.mapped (which the
        // send worker has to do every frame) run at cache speed
        // (~10 GB/s) instead of write-combined speed (~250 MB/s).
        int idx = find_memory_type(mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
          | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        bool got_cached = (idx >= 0);
        if (!got_cached) {
            idx = find_memory_type(mr.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        if (idx < 0) { log("no HOST_VISIBLE|COHERENT memory type"); return false; }
        if (i == 0) {
            log("staging memory: type idx=%d, HOST_CACHED %s", idx,
                got_cached ? "yes (fast send)"
                           : "NO — falling back to write-combined (slow send)");
        }
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = (uint32_t)idx;
        if (g_vk.AllocateMemory(g_sess.vkDevice, &ai, nullptr, &s.mem) != VK_SUCCESS) {
            log("ensure_staging_slots: AllocateMemory failed (slot %zu)", i); return false;
        }
        if (g_vk.BindBufferMemory(g_sess.vkDevice, s.buf, s.mem, 0) != VK_SUCCESS) {
            log("ensure_staging_slots: BindBufferMemory failed (slot %zu)", i); return false;
        }
        if (g_vk.MapMemory(g_sess.vkDevice, s.mem, 0, need, 0, &s.mapped) != VK_SUCCESS) {
            log("ensure_staging_slots: MapMemory failed (slot %zu)", i); return false;
        }

        // Fence created SIGNALED so the first do_continuous_capture()
        // that picks this slot doesn't need a special-case for "never
        // submitted yet" — ResetFences just clears whatever previous-
        // or-initial signal.
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (g_vk.CreateFence(g_sess.vkDevice, &fci, nullptr, &s.fence) != VK_SUCCESS) {
            log("ensure_staging_slots: CreateFence failed (slot %zu)", i); return false;
        }

        // Per-slot command buffer from the shared pool. Each slot's
        // cmdBuf can be in flight on the GPU independently of the others.
        VkCommandBufferAllocateInfo cbi{};
        cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool        = g_sess.cmdPool;
        cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        if (g_vk.AllocateCommandBuffers(g_sess.vkDevice, &cbi, &s.cmdBuf) != VK_SUCCESS) {
            log("ensure_staging_slots: AllocateCommandBuffers failed (slot %zu)", i); return false;
        }

        s.size = need;
        s.read_lock.store(false);
        s.seq.store(0);
    }
    log("staging: %zu slots × %llu bytes each (%llu per eye)",
        N_STAGING, (unsigned long long)need, (unsigned long long)bytes_per_eye);

    // ── Stage 1: FD-export validation (additive; host path above untouched) ──
    if (g_export_fd) {
        bool sem_ok = ensure_export_semaphore();
        int  n_ok   = 0;
        for (size_t i = 0; i < N_STAGING; ++i)
            if (alloc_export_buffer(g_staging[i], need)) ++n_ok;
        log("export: %d/%zu exportable buffers ready; sem=%s; "
            "mem_fds={%d,%d,%d} sem_fd=%d",
            n_ok, (size_t)N_STAGING, sem_ok ? "ok" : "FAIL",
            g_staging[0].mem_fd, g_staging[1].mem_fd, g_staging[2].mem_fd,
            g_export_sem_fd);
    }
    return true;
}

bool ensure_resources() {
    // Only the queue handle + command pool live in SessionInfo now;
    // fence and per-slot command buffer are owned by g_staging[].
    if (g_sess.cmdPool != VK_NULL_HANDLE) return true;
    g_vk.GetDeviceQueue(g_sess.vkDevice, g_sess.queueFamily, g_sess.queueIndex, &g_sess.vkQueue);
    if (g_sess.vkQueue == VK_NULL_HANDLE) { log("GetDeviceQueue → null"); return false; }

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g_sess.queueFamily;
    if (g_vk.CreateCommandPool(g_sess.vkDevice, &pci, nullptr, &g_sess.cmdPool) != VK_SUCCESS) {
        log("CreateCommandPool failed"); return false;
    }
    log("session cmdPool ready");
    return true;
}

// Forward decl — defined after do_continuous_capture.
bool ensure_swapchain_images();

// Run every xrEndFrame to keep the staging ring fresh. Send worker picks
// the freshest ready slot on GET, so capture latency drops to "just send".
void do_continuous_capture() {
    // Lazy enumerate VkImage handles if we haven't yet
    if (!ensure_swapchain_images()) { log("capture: image enumeration failed"); return; }

    struct Tgt { VkImage img; uint32_t w, h; size_t bytes; };
    std::vector<Tgt> targets;
    {
        std::lock_guard<std::mutex> lk(g_sc_mu);
        for (auto& sc : g_color_scs) {
            if (sc.last_released < 0 || sc.images.empty()) {
                // Swapchain not yet released by app — silent skip
                // (this is normal for the first few frames).
                return;
            }
            targets.push_back({ sc.images[(size_t)sc.last_released],
                                sc.width, sc.height,
                                (size_t)sc.width * sc.height * 4 });
        }
    }
    if (targets.size() != 2) return;
    const size_t bpe = targets[0].bytes;

    std::lock_guard<std::mutex> lk(g_sess_mu);
    if (!g_vk.loaded)                return;
    if (!ensure_resources())         return;
    if (!ensure_staging_slots(bpe))  return;

    // Pick the slot with the lowest seq (or seq=0 = never filled).
    // Skip slots with read_lock held (worker is mid-send).
    size_t   slot_idx  = SIZE_MAX;
    uint64_t oldest_sq = UINT64_MAX;
    for (size_t i = 0; i < N_STAGING; ++i) {
        if (g_staging[i].read_lock.load(std::memory_order_acquire)) continue;
        uint64_t sq = g_staging[i].seq.load(std::memory_order_acquire);
        if (sq < oldest_sq) { oldest_sq = sq; slot_idx = i; }
    }
    if (slot_idx == SIZE_MAX) {
        // All slots locked by worker — extremely rare with N=3.
        // Skip this frame; next EndFrame will retry.
        return;
    }
    StagingSlot& slot = g_staging[slot_idx];
    bool expected = false;
    if (!slot.read_lock.compare_exchange_strong(
            expected, true,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;   // worker grabbed it between scan and CAS
    }

    // Hold read_lock from here until after seq is updated, so the worker
    // never sees a slot with a fresh seq but a stale fence / metadata.

    g_vk.ResetFences(g_sess.vkDevice, 1, &slot.fence);
    g_vk.ResetCommandBuffer(slot.cmdBuf, 0);

    VkCommandBufferBeginInfo bbi{};
    bbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (g_vk.BeginCommandBuffer(slot.cmdBuf, &bbi) != VK_SUCCESS) {
        log("BeginCommandBuffer failed");
        slot.read_lock.store(false, std::memory_order_release);
        return;
    }

    for (size_t eye = 0; eye < 2; ++eye) {
        const auto& t = targets[eye];
        VkImageMemoryBarrier toSrc{};
        toSrc.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toSrc.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toSrc.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image               = t.img;
        toSrc.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        toSrc.subresourceRange.baseMipLevel   = 0;
        toSrc.subresourceRange.levelCount     = 1;
        toSrc.subresourceRange.baseArrayLayer = 0;
        toSrc.subresourceRange.layerCount     = 1;
        g_vk.CmdPipelineBarrier(slot.cmdBuf,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy reg{};
        reg.bufferOffset                    = bpe * eye;
        reg.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        reg.imageSubresource.layerCount     = 1;
        reg.imageExtent                     = {t.w, t.h, 1};
        // Host readback copy — skipped in FD-export mode; the DEVICE_LOCAL
        // exportable buffer below replaces it, so there is no GPU→CPU readback.
        if (!g_export_fd)
            g_vk.CmdCopyImageToBuffer(slot.cmdBuf, t.img,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.buf, 1, &reg);
        // FD-export: same image, same region, into the exportable buffer.
        if (g_export_fd && slot.exp_buf != VK_NULL_HANDLE)
            g_vk.CmdCopyImageToBuffer(slot.cmdBuf, t.img,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.exp_buf, 1, &reg);

        VkImageMemoryBarrier toColor = toSrc;
        toColor.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toColor.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toColor.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        g_vk.CmdPipelineBarrier(slot.cmdBuf,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toColor);
    }
    if (g_vk.EndCommandBuffer(slot.cmdBuf) != VK_SUCCESS) {
        log("EndCommandBuffer failed");
        slot.read_lock.store(false, std::memory_order_release);
        return;
    }
    // Monotonic seq for this frame; also the timeline value the CUDA
    // consumer waits on (Stage 3). Computed before submit so it can be the
    // signalled value; stored into slot.seq after (replacing the old
    // post-submit fetch_add).
    const uint64_t this_seq = g_frame_seq.fetch_add(1) + 1;

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &slot.cmdBuf;
    // FD-export: signal the exported timeline semaphore to this_seq when the
    // copy completes, so CUDA can wait on it (cudaWaitExternalSemaphoresAsync)
    // instead of relying on the host fence.
    VkTimelineSemaphoreSubmitInfo tssi{};
    if (g_export_fd && g_export_sem != VK_NULL_HANDLE) {
        tssi.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tssi.signalSemaphoreValueCount = 1;
        tssi.pSignalSemaphoreValues    = &this_seq;
        si.pNext                = &tssi;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores    = &g_export_sem;
    }
    if (g_vk.QueueSubmit(g_sess.vkQueue, 1, &si, slot.fence) != VK_SUCCESS) {
        log("QueueSubmit failed");
        slot.read_lock.store(false, std::memory_order_release);
        return;
    }

    // Now write the per-slot metadata. The worker reads these after
    // confirming fence-signalled + acquiring read_lock; ordering is
    // covered by the release on read_lock below.
    slot.bpe = bpe;
    for (uint32_t eye = 0; eye < 2; ++eye) {
        slot.w[eye] = targets[eye].w;
        slot.h[eye] = targets[eye].h;
    }
    {
        std::lock_guard<std::mutex> pose_lk(g_pose_mu);
        for (uint32_t eye = 0; eye < 2; ++eye) {
            if (eye < g_eye_poses.size())
                slot.pose[eye] = g_eye_poses[eye];
        }
    }
    // Assign the precomputed monotonic sequence number — this is what the
    // worker uses to pick the freshest slot, and the timeline value CUDA waits on.
    slot.seq.store(this_seq, std::memory_order_release);
    // Release the slot: GPU copy is in flight, fence will signal when
    // the data is valid. Worker may now read this slot.
    slot.read_lock.store(false, std::memory_order_release);
}

// ───── Hooks ───────────────────────────────────────────────────────────────

XrResult XRAPI_CALL h_CreateSession(
    XrInstance instance, const XrSessionCreateInfo* ci, XrSession* sess)
{
    log("STEP CreateSession enter");
    auto nd = get_dispatch(instance);
    if (!nd.CreateSession) return XR_ERROR_FUNCTION_UNSUPPORTED;
    const void* next = ci ? ci->next : nullptr;
    while (next) {
        const XrBaseInStructure* base = reinterpret_cast<const XrBaseInStructure*>(next);
        if (base->type == XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR) {
            const uint8_t* raw = reinterpret_cast<const uint8_t*>(base) + 16;
            std::lock_guard<std::mutex> lk(g_sess_mu);
            g_sess.vkInstance       = *reinterpret_cast<const VkInstance*>(raw + 0);
            g_sess.vkPhysicalDevice = *reinterpret_cast<const VkPhysicalDevice*>(raw + 8);
            g_sess.vkDevice         = *reinterpret_cast<const VkDevice*>(raw + 16);
            g_sess.queueFamily      = *reinterpret_cast<const uint32_t*>(raw + 24);
            g_sess.queueIndex       = *reinterpret_cast<const uint32_t*>(raw + 28);
            g_sess.handles_captured = true;
            log("  captured handles: vkInstance=%p vkDevice=%p qFam=%u",
                (void*)g_sess.vkInstance, (void*)g_sess.vkDevice, g_sess.queueFamily);
            break;
        }
        next = base->next;
    }
    XrResult r = nd.CreateSession(instance, ci, sess);
    log("STEP CreateSession exit (r=%d)", (int)r);
    return r;
}

XrResult XRAPI_CALL h_DestroySession(XrSession s) {
    log("STEP DestroySession enter");
    socket_close_all();   // joins send worker before we touch g_staging
    {
        std::lock_guard<std::mutex> lk(g_sess_mu);
        if (g_vk.loaded) {
            g_vk.DeviceWaitIdle(g_sess.vkDevice);
            // Per-slot teardown. cmdBuf handles are freed implicitly when
            // we DestroyCommandPool below, so we only need to clear the
            // handle field here.
            for (size_t i = 0; i < N_STAGING; ++i) {
                StagingSlot& sl = g_staging[i];
                if (sl.mapped) { g_vk.UnmapMemory(g_sess.vkDevice, sl.mem); sl.mapped = nullptr; }
                if (sl.buf != VK_NULL_HANDLE)   { g_vk.DestroyBuffer(g_sess.vkDevice, sl.buf, nullptr);   sl.buf = VK_NULL_HANDLE; }
                if (sl.mem != VK_NULL_HANDLE)   { g_vk.FreeMemory(g_sess.vkDevice, sl.mem, nullptr);     sl.mem = VK_NULL_HANDLE; }
                if (sl.fence != VK_NULL_HANDLE) { g_vk.DestroyFence(g_sess.vkDevice, sl.fence, nullptr); sl.fence = VK_NULL_HANDLE; }
                sl.cmdBuf = VK_NULL_HANDLE;
                sl.size = 0;
                sl.read_lock.store(false);
                sl.seq.store(0);
            }
            if (g_sess.cmdPool != VK_NULL_HANDLE) {
                g_vk.DestroyCommandPool(g_sess.vkDevice, g_sess.cmdPool, nullptr);
                g_sess.cmdPool = VK_NULL_HANDLE;
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_sc_mu);
        g_color_scs.clear();
        g_sc_idx.clear();
    }
    auto nd = get_first_dispatch();
    XrResult r = nd.DestroySession ? nd.DestroySession(s) : XR_SUCCESS;
    log("STEP DestroySession exit");
    return r;
}

XrResult XRAPI_CALL h_BeginSession(XrSession s, const XrSessionBeginInfo* i) {
    log("STEP BeginSession enter");
    auto nd = get_first_dispatch();
    XrResult r = nd.BeginSession ? nd.BeginSession(s, i) : XR_ERROR_FUNCTION_UNSUPPORTED;
    log("STEP BeginSession exit (r=%d)", (int)r);
    return r;
}

XrResult XRAPI_CALL h_EndSession(XrSession s) {
    auto nd = get_first_dispatch();
    return nd.EndSession ? nd.EndSession(s) : XR_SUCCESS;
}

XrResult XRAPI_CALL h_CreateSwapchain(
    XrSession s, const XrSwapchainCreateInfo* ci, XrSwapchain* sc)
{
    auto nd = get_first_dispatch();
    if (!nd.CreateSwapchain) return XR_ERROR_FUNCTION_UNSUPPORTED;

    XrSwapchainCreateInfo patched = *ci;
    const bool is_color = (ci->usageFlags & XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT) != 0;
    if (is_color) patched.usageFlags |= XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT;

    XrResult r = nd.CreateSwapchain(s, &patched, sc);
    log("STEP CreateSwapchain (fmt=%lld, %ux%u, color=%d) → r=%d",
        (long long)ci->format, ci->width, ci->height, (int)is_color, (int)r);
    if (r != XR_SUCCESS || !sc || !is_color) return r;

    // RECORD the handle + dims here, but DEFER image enumeration to a safer
    // moment (first capture). Calling xrEnumerateSwapchainImages from inside
    // xrCreateSwapchain — while the loader/runtime is still in its session-
    // setup critical section — has been observed to hang the runtime.
    ColorSwapchain cs;
    cs.handle = *sc;
    cs.format = ci->format;
    cs.width  = ci->width;
    cs.height = ci->height;
    // cs.images left empty; will be filled by ensure_swapchain_images()
    {
        std::lock_guard<std::mutex> lk(g_sc_mu);
        g_sc_idx[*sc] = g_color_scs.size();
        g_color_scs.push_back(std::move(cs));
    }
    log("  recorded color swapchain #%zu (images will be enumerated lazily)",
        g_color_scs.size() - 1);
    return r;
}

// Lazy: enumerate VkImage handles for each registered color swapchain.
// Called from do_continuous_capture() before we try to use them. Idempotent.
bool ensure_swapchain_images() {
    auto nd = get_first_dispatch();
    if (!nd.EnumerateSwapchainImages) {
        log("ensure_swapchain_images: no EnumerateSwapchainImages dispatch");
        return false;
    }
    std::lock_guard<std::mutex> lk(g_sc_mu);
    for (auto& cs : g_color_scs) {
        if (!cs.images.empty()) continue;
        uint32_t imgN = 0;
        nd.EnumerateSwapchainImages(cs.handle, 0, &imgN, nullptr);
        std::vector<XrSwapchainImageVulkan2KHR> imgs(imgN);
        for (auto& im : imgs) im.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
        nd.EnumerateSwapchainImages(cs.handle, imgN, &imgN,
            reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data()));
        cs.images.reserve(imgN);
        for (auto& im : imgs) cs.images.push_back(im.image);
        log("  enumerated swapchain images: %u images for swapchain %p",
            imgN, (void*)cs.handle);
    }
    return true;
}

XrResult XRAPI_CALL h_DestroySwapchain(XrSwapchain sc) {
    auto nd = get_first_dispatch();
    return nd.DestroySwapchain ? nd.DestroySwapchain(sc) : XR_SUCCESS;
}

XrResult XRAPI_CALL h_AcquireSwapchainImage(
    XrSwapchain sc, const XrSwapchainImageAcquireInfo* ai, uint32_t* idx)
{
    auto nd = get_first_dispatch();
    XrResult r = nd.AcquireSwapchainImage(sc, ai, idx);
    if (r == XR_SUCCESS && idx) {
        std::lock_guard<std::mutex> lk(g_sc_mu);
        auto it = g_sc_idx.find(sc);
        if (it != g_sc_idx.end()) g_color_scs[it->second].last_acquired = (int)*idx;
    }
    return r;
}

XrResult XRAPI_CALL h_ReleaseSwapchainImage(
    XrSwapchain sc, const XrSwapchainImageReleaseInfo* ri)
{
    {
        std::lock_guard<std::mutex> lk(g_sc_mu);
        auto it = g_sc_idx.find(sc);
        if (it != g_sc_idx.end()) {
            auto& s = g_color_scs[it->second];
            s.last_released = s.last_acquired;
        }
    }
    auto nd = get_first_dispatch();
    return nd.ReleaseSwapchainImage(sc, ri);
}

// ───── Overlay swapchain (Piece 1: layer-owned, layer-filled) ──────────────
// A dedicated color swapchain the LAYER owns and fills itself, injected as a
// composition layer. Test mode fills it with a solid color; the real path
// (Piece 2/3) will vkCmdCopyBufferToImage the pipeline's rendered overlay.
struct OverlaySwapchain {
    XrSwapchain          handle = XR_NULL_HANDLE;
    std::vector<VkImage> images;
    uint32_t             w = 0, h = 0;       // PER-EYE dimensions
    uint32_t             array_size = 1;     // single layer; eyes laid out
                                             // side-by-side in a 2w-wide image
                                             // (L in x[0,w), R in x[w,2w)).
                                             // Avoids imageArrayIndex=1, which
                                             // some runtimes mis-composite on
                                             // injected layers (right-eye flicker).
    int64_t              format = 43;   // VK_FORMAT_R8G8B8A8_SRGB (matches app)
    VkCommandBuffer      cmd = VK_NULL_HANDLE;
    VkFence              fence = VK_NULL_HANDLE;
    bool                 ready = false;
    bool                 has_content = false;   // set after first upload
    // Persistent host-visible staging buffer for RGBA uploads (Piece 2).
    // Sized for both eyes: eye 0 at offset 0, eye 1 at offset w*h*4.
    VkBuffer             upbuf  = VK_NULL_HANDLE;
    VkDeviceMemory       upmem  = VK_NULL_HANDLE;
    void*                upmap  = nullptr;
    VkDeviceSize         upsize = 0;
};
OverlaySwapchain g_overlay;

// ───── Overlay receive (Piece 2): pipeline → layer ─────────────────────────
// A second listening socket. A worker accepts one client (the handoff),
// reads OverlayHeader + RGBA payload, and stashes the freshest frame. The
// render thread (h_EndFrame) consumes the latest into the swapchain.
std::mutex            g_ovl_mu;
std::vector<uint8_t>  g_ovl_pix[2];     // latest RGBA per eye (0=L, 1=R)
uint32_t              g_ovl_w = 0, g_ovl_h = 0;
uint32_t              g_ovl_eyes = 0;   // how many eyes the last frame carried
uint32_t              g_ovl_seq = 0;
uint64_t              g_ovl_capture_ts_us = 0;  // wall-clock µs of mask content (g_ovl_mu)
float                 g_ovl_fov[2][4] = {{0,0,0,0},{0,0,0,0}};  // per-eye L,R,U,D
bool                  g_ovl_have_fov = false;
bool                  g_ovl_fresh = false;
std::atomic<unsigned long long> g_ovl_drained{0};  // stale frames dropped (latest-wins)
std::atomic<bool>     g_ovl_run{false};
std::thread           g_ovl_thread;
int                   g_ovl_listen_fd = -1;
int                   g_ovl_client_fd = -1;

static bool recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, p + got, n - got, 0);
        if (r == 0) return false;            // peer closed
        if (r < 0) { if (errno == EINTR) continue; return false; }
        got += (size_t)r;
    }
    return true;
}

void overlay_recv_run() {
    log("overlay_recv: worker started, listening %s", OVERLAY_SOCKET_PATH);
    while (g_ovl_run.load()) {
        if (g_ovl_client_fd < 0) {
            int c = accept(g_ovl_listen_fd, nullptr, nullptr);
            if (c < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            g_ovl_client_fd = c;
            log("overlay_recv: client connected (fd=%d)", c);
        }
        OverlayHeader hd{};
        if (!recv_all(g_ovl_client_fd, &hd, sizeof(hd))) {
            log("overlay_recv: client disconnected");
            close(g_ovl_client_fd); g_ovl_client_fd = -1; continue;
        }
        if (hd.magic != OVERLAY_MAGIC) {
            log("overlay_recv: bad magic 0x%x — dropping client", hd.magic);
            close(g_ovl_client_fd); g_ovl_client_fd = -1; continue;
        }
        const uint32_t bpp = (hd.format == OVERLAY_FMT_RGBA8) ? 4u : 1u;
        const size_t   eye_bytes = (size_t)hd.w * hd.h * bpp;
        const uint32_t neyes = (hd.eye_count >= 2) ? 2u : 1u;

        std::vector<uint8_t> raw[2];
        bool ok = true;
        for (uint32_t e = 0; e < neyes; ++e) {
            raw[e].resize(eye_bytes);
            if (!recv_all(g_ovl_client_fd, raw[e].data(), eye_bytes)) { ok = false; break; }
        }
        if (!ok) {
            log("overlay_recv: short read on payload — dropping client");
            close(g_ovl_client_fd); g_ovl_client_fd = -1; continue;
        }

        // LATEST-WINS DRAIN. The producer sends a full mask every optimiser
        // frame (~22 Hz). If frames ever queue in the socket buffer, reading
        // them FIFO would display an ever-older mask (the latency the optimiser
        // doesn't have). So if a COMPLETE newer frame is already buffered, drop
        // this one (we only paid the cheap recv, not the expand/upload) and go
        // straight to the freshest. Bounds socket latency to ≤1 frame.
        {
            int avail = 0;
            const size_t frame_bytes =
                sizeof(OverlayHeader) + (size_t)neyes * eye_bytes;
            if (ioctl(g_ovl_client_fd, FIONREAD, &avail) == 0 &&
                (size_t)avail >= frame_bytes) {
                ++g_ovl_drained;
                continue;   // stale — skip expand/publish, read the newer one
            }
        }

        // Expand mono8 → RGBA so the upload path is uniform.
        std::vector<uint8_t> rgba[2];
        for (uint32_t e = 0; e < neyes; ++e) {
            if (hd.format == OVERLAY_FMT_RGBA8) {
                rgba[e] = std::move(raw[e]);
            } else {  // mono8 → green-tinted, premultiplied RGBA
                rgba[e].resize((size_t)hd.w * hd.h * 4);
                for (size_t i = 0; i < (size_t)hd.w * hd.h; ++i) {
                    uint8_t m = raw[e][i];
                    rgba[e][i*4+0] = 0; rgba[e][i*4+1] = m;
                    rgba[e][i*4+2] = 0; rgba[e][i*4+3] = m;
                }
            }
        }
        {
            std::lock_guard<std::mutex> lk(g_ovl_mu);
            g_ovl_pix[0].swap(rgba[0]);
            if (neyes >= 2) g_ovl_pix[1].swap(rgba[1]);
            g_ovl_w = hd.w; g_ovl_h = hd.h; g_ovl_eyes = neyes;
            g_ovl_seq = hd.seq; g_ovl_fresh = true;
            g_ovl_capture_ts_us = hd.capture_ts_us;
            for (int e = 0; e < 2; ++e)
                for (int k = 0; k < 4; ++k) g_ovl_fov[e][k] = hd.fov[e][k];
            // Treat fov as valid if any angle is non-zero.
            g_ovl_have_fov = (hd.fov[0][0] != 0.f || hd.fov[0][1] != 0.f);
        }

        // Delivery-latency telemetry: capture_ts_us is stamped by the producer
        // when the mask became ready (wall clock, same machine). now - that is
        // the producer→publish age (streamer 1-frame + socket + recv). If this
        // stays ~1 frame, residual lag is upstream (optimiser / CloudXR); if it
        // climbs, the socket was backlogged (the drain above should prevent it).
        if (hd.capture_ts_us != 0) {
            uint64_t now_us = (uint64_t)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            double age_ms = (now_us >= hd.capture_ts_us)
                ? (double)(now_us - hd.capture_ts_us) / 1000.0 : 0.0;
            static double s_age_acc = 0.0; static int s_age_n = 0;
            s_age_acc += age_ms; ++s_age_n;
            auto tnow = std::chrono::steady_clock::now();
            static auto s_last = tnow;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    tnow - s_last).count() >= 2000) {
                log("overlay_recv: delivery age avg %.1f ms over %d frames, "
                    "drained %llu stale", s_age_acc / s_age_n,
                    s_age_n, (unsigned long long)g_ovl_drained.load());
                s_age_acc = 0.0; s_age_n = 0; s_last = tnow;
            }
        }
    }
    log("overlay_recv: worker exiting");
}

void overlay_recv_start() {
    if (g_ovl_listen_fd >= 0) return;
    unlink(OVERLAY_SOCKET_PATH);
    g_ovl_listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_ovl_listen_fd < 0) { log("overlay_recv: socket(): %s", strerror(errno)); return; }
    sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, OVERLAY_SOCKET_PATH, sizeof(addr.sun_path) - 1);
    if (bind(g_ovl_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        log("overlay_recv: bind(): %s", strerror(errno));
        close(g_ovl_listen_fd); g_ovl_listen_fd = -1; return;
    }
    if (listen(g_ovl_listen_fd, 1) < 0) {
        log("overlay_recv: listen(): %s", strerror(errno));
        close(g_ovl_listen_fd); g_ovl_listen_fd = -1; return;
    }
    g_ovl_run = true;
    g_ovl_thread = std::thread(overlay_recv_run);
    log("overlay_recv: started on %s", OVERLAY_SOCKET_PATH);
}



bool overlay_ensure(XrSession session, uint32_t w, uint32_t h) {
    if (g_overlay.ready) return true;
    auto nd = get_first_dispatch();
    if (!nd.CreateSwapchain || !nd.EnumerateSwapchainImages ||
        !nd.AcquireSwapchainImage || !nd.WaitSwapchainImage ||
        !nd.ReleaseSwapchainImage) {
        log("overlay_ensure: missing swapchain dispatch fns");
        return false;
    }
    if (!g_vk.loaded || !ensure_resources()) {
        log("overlay_ensure: vulkan/resources not ready");
        return false;
    }

    XrSwapchainCreateInfo ci{};
    ci.type        = XR_TYPE_SWAPCHAIN_CREATE_INFO;
    ci.usageFlags  = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                     XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    ci.format      = g_overlay.format;
    ci.sampleCount = 1;
    ci.width       = w * 2;                  // double-wide: L | R side by side
    ci.height      = h;
    ci.faceCount   = 1;
    ci.arraySize   = 1;                      // single layer (no imageArrayIndex)
    ci.mipCount    = 1;
    XrResult r = nd.CreateSwapchain(session, &ci, &g_overlay.handle);
    if (r != XR_SUCCESS) { log("overlay_ensure: CreateSwapchain → %d", (int)r); return false; }
    g_overlay.w = w; g_overlay.h = h;

    uint32_t imgN = 0;
    nd.EnumerateSwapchainImages(g_overlay.handle, 0, &imgN, nullptr);
    std::vector<XrSwapchainImageVulkan2KHR> imgs(imgN);
    for (auto& im : imgs) im.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
    nd.EnumerateSwapchainImages(g_overlay.handle, imgN, &imgN,
        reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data()));
    for (auto& im : imgs) g_overlay.images.push_back(im.image);

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool        = g_sess.cmdPool;
    cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    if (g_vk.AllocateCommandBuffers(g_sess.vkDevice, &cbi, &g_overlay.cmd) != VK_SUCCESS) {
        log("overlay_ensure: AllocateCommandBuffers failed"); return false;
    }
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (g_vk.CreateFence(g_sess.vkDevice, &fci, nullptr, &g_overlay.fence) != VK_SUCCESS) {
        log("overlay_ensure: CreateFence failed"); return false;
    }

    g_overlay.ready = true;
    log("overlay_ensure: swapchain %ux%u (2x%ux%u per-eye), %zu images, fmt=%lld",
        w * 2, h, w, h, g_overlay.images.size(), (long long)g_overlay.format);
    return true;
}

// Ensure the persistent host-visible upload buffer is at least `bytes`.
bool overlay_ensure_upbuf(VkDeviceSize bytes) {
    if (g_overlay.upbuf != VK_NULL_HANDLE && g_overlay.upsize >= bytes) return true;
    if (g_overlay.upbuf != VK_NULL_HANDLE) {
        g_vk.DeviceWaitIdle(g_sess.vkDevice);
        if (g_overlay.upmap) { g_vk.UnmapMemory(g_sess.vkDevice, g_overlay.upmem); g_overlay.upmap = nullptr; }
        g_vk.DestroyBuffer(g_sess.vkDevice, g_overlay.upbuf, nullptr);
        g_vk.FreeMemory(g_sess.vkDevice, g_overlay.upmem, nullptr);
        g_overlay.upbuf = VK_NULL_HANDLE; g_overlay.upmem = VK_NULL_HANDLE;
    }
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (g_vk.CreateBuffer(g_sess.vkDevice, &bci, nullptr, &g_overlay.upbuf) != VK_SUCCESS) {
        log("overlay_upbuf: CreateBuffer failed"); return false;
    }
    VkMemoryRequirements mr{};
    g_vk.GetBufferMemoryRequirements(g_sess.vkDevice, g_overlay.upbuf, &mr);
    int idx = find_memory_type(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (idx < 0) { log("overlay_upbuf: no host-visible mem type"); return false; }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size; mai.memoryTypeIndex = (uint32_t)idx;
    if (g_vk.AllocateMemory(g_sess.vkDevice, &mai, nullptr, &g_overlay.upmem) != VK_SUCCESS) {
        log("overlay_upbuf: AllocateMemory failed"); return false;
    }
    g_vk.BindBufferMemory(g_sess.vkDevice, g_overlay.upbuf, g_overlay.upmem, 0);
    if (g_vk.MapMemory(g_sess.vkDevice, g_overlay.upmem, 0, mr.size, 0, &g_overlay.upmap) != VK_SUCCESS) {
        log("overlay_upbuf: MapMemory failed"); return false;
    }
    g_overlay.upsize = mr.size;
    return true;
}

// Upload one or two RGBA host buffers into the overlay swapchain's array
// layers (0=left, 1=right). If pxR is null, pxL is uploaded to both layers.
bool overlay_upload_stereo(const uint8_t* pxL, const uint8_t* pxR,
                           uint32_t w, uint32_t h) {
    auto nd = get_first_dispatch();
    const VkDeviceSize eye = (VkDeviceSize)w * h * 4;
    if (!overlay_ensure_upbuf(eye * 2)) return false;
    // eye 0 at offset 0, eye 1 at offset `eye`.
    memcpy(g_overlay.upmap, pxL, (size_t)eye);
    const uint8_t* src1 = pxR ? pxR : pxL;
    memcpy(static_cast<uint8_t*>(g_overlay.upmap) + eye, src1, (size_t)eye);

    uint32_t idx = 0;
    XrSwapchainImageAcquireInfo ai{}; ai.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO;
    if (nd.AcquireSwapchainImage(g_overlay.handle, &ai, &idx) != XR_SUCCESS) return false;
    XrSwapchainImageWaitInfo wi{}; wi.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO;
    wi.timeout = XR_INFINITE_DURATION;
    if (nd.WaitSwapchainImage(g_overlay.handle, &wi) != XR_SUCCESS) return false;

    VkImage img = g_overlay.images[idx];
    g_vk.ResetFences(g_sess.vkDevice, 1, &g_overlay.fence);
    g_vk.ResetCommandBuffer(g_overlay.cmd, 0);
    VkCommandBufferBeginInfo bbi{};
    bbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    g_vk.BeginCommandBuffer(g_overlay.cmd, &bbi);

    // Barrier covers BOTH array layers.
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1; range.baseArrayLayer = 0;
    range.layerCount = g_overlay.array_size;

    VkImageMemoryBarrier toDst{};
    toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toDst.srcAccessMask = 0; toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image = img; toDst.subresourceRange = range;
    g_vk.CmdPipelineBarrier(g_overlay.cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toDst);

    // Two copy regions, both into array layer 0: left eye at x=0, right eye
    // at x=w (double-wide image). The runtime composites each eye by selecting
    // its half via imageRect (no array index), which avoids the right-eye
    // flicker seen with imageArrayIndex=1 on injected layers.
    VkBufferImageCopy regs[2]{};
    for (uint32_t e = 0; e < 2; ++e) {
        regs[e].bufferOffset = (VkDeviceSize)e * eye;
        regs[e].bufferRowLength = 0; regs[e].bufferImageHeight = 0;
        regs[e].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regs[e].imageSubresource.mipLevel = 0;
        regs[e].imageSubresource.baseArrayLayer = 0;
        regs[e].imageSubresource.layerCount = 1;
        regs[e].imageOffset = {(int32_t)(e * w), 0, 0};   // L at 0, R at w
        regs[e].imageExtent = {w, h, 1};
    }
    g_vk.CmdCopyBufferToImage(g_overlay.cmd, g_overlay.upbuf, img,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, regs);

    VkImageMemoryBarrier toCol = toDst;
    toCol.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toCol.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toCol.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toCol.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    g_vk.CmdPipelineBarrier(g_overlay.cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toCol);

    g_vk.EndCommandBuffer(g_overlay.cmd);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &g_overlay.cmd;
    g_vk.QueueSubmit(g_sess.vkQueue, 1, &si, g_overlay.fence);
    g_vk.WaitForFences(g_sess.vkDevice, 1, &g_overlay.fence, VK_TRUE, UINT64_MAX);

    XrSwapchainImageReleaseInfo ri{}; ri.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO;
    nd.ReleaseSwapchainImage(g_overlay.handle, &ri);
    return true;
}


XrResult XRAPI_CALL h_EndFrame(XrSession s, const XrFrameEndInfo* info) {
    auto nd = get_first_dispatch();

    int n = ++g_endframe_count;

    // Lazy init order: socket, then vulkan (only after a few frames so we
    // don't race the runtime's own init).
    if (g_sess.handles_captured && !g_socket_up.load()) socket_init_if_needed();
    if (g_sess.handles_captured && n >= 10 && !g_vk_load_attempted.exchange(true)) {
        load_vulkan_fns();
    }
    // Start the overlay receive worker once (only when the overlay is
    // enabled; harmless otherwise — it would just listen for a client).
    {
        static std::atomic<bool> ovl_started{false};
        const char* oe = getenv("XR_OVERLAY");
        if (g_sess.handles_captured && oe && oe[0] && oe[0] != '0' &&
            !ovl_started.exchange(true))
            overlay_recv_start();
    }

    // ── Cache per-eye poses from the projection layer ────────────────────
    // Walk info->layers[]. The first XrCompositionLayerProjection we find
    // tells us the per-eye poses the runtime is actually using to compose
    // this frame. We grab up to 2 views (stereo); any extra are ignored.
    //
    // DIAGNOSTIC: report the layer inventory once so we can see whether
    // Isaac/Omniverse submits a single projection layer or several. Layer
    // type codes: PROJECTION=35000, QUAD=35001, CUBE=35004, CYLINDER=35005,
    // EQUIRECT=35006/35009, PASSTHROUGH_FB=1000118000 (approx). The exact
    // numbers print so we don't have to guess.
    if (info && info->layers) {
        static std::atomic<bool> layers_announced{false};
        if (!layers_announced.exchange(true)) {
            log("LAYER INVENTORY: layerCount=%u", info->layerCount);
            for (uint32_t li = 0; li < info->layerCount; ++li) {
                const XrCompositionLayerBaseHeader* lb = info->layers[li];
                if (!lb) { log("  layer[%u] = NULL", li); continue; }
                uint32_t vc = 0;
                if (lb->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION) {
                    const auto* pj =
                        reinterpret_cast<const XrCompositionLayerProjection*>(lb);
                    vc = pj->viewCount;
                }
                log("  layer[%u] type=%d layerFlags=0x%x viewCount=%u",
                    li, (int)lb->type, (unsigned)lb->layerFlags, vc);
            }
        }
    }
    const XrCompositionLayerProjection* g_proj_for_inject = nullptr;
    if (info && info->layers && info->layerCount > 0) {
        for (uint32_t li = 0; li < info->layerCount; ++li) {
            const XrCompositionLayerBaseHeader* lb = info->layers[li];
            if (!lb) continue;
            if (lb->type != XR_TYPE_COMPOSITION_LAYER_PROJECTION) continue;
            const auto* proj = reinterpret_cast<const XrCompositionLayerProjection*>(lb);
            if (!proj->views || proj->viewCount == 0) continue;
            g_proj_for_inject = proj;   // remember for the optional test injection

            std::lock_guard<std::mutex> lk(g_pose_mu);
            const uint32_t nv = (proj->viewCount > 2u) ? 2u : proj->viewCount;
            // Persist the full projection params so the overlay can inject
            // every frame, even when a later frame lacks a projection layer.
            g_pproj.seen  = true;
            g_pproj.space = proj->space;
            g_pproj.nv    = nv;
            for (uint32_t v = 0; v < nv; ++v) {
                g_pproj.pose[v] = proj->views[v].pose;
                g_pproj.fov[v]  = proj->views[v].fov;
            }
            for (uint32_t v = 0; v < nv; ++v) {
                const XrPosef& p = proj->views[v].pose;
                g_eye_poses[v].valid = true;
                g_eye_poses[v].px = p.position.x;
                g_eye_poses[v].py = p.position.y;
                g_eye_poses[v].pz = p.position.z;
                g_eye_poses[v].qw = p.orientation.w;
                g_eye_poses[v].qx = p.orientation.x;
                g_eye_poses[v].qy = p.orientation.y;
                g_eye_poses[v].qz = p.orientation.z;
            }
            // Record (capture wall-clock, eye poses) for overlay reprojection.
            // ts uses the same CLOCK_REALTIME wall clock test_cloudxr stamps
            // the frame with (int(time.time()*1e6)), so the overlay's
            // capture_ts_us indexes straight back into this ring.
            g_pose_ring[g_pose_ring_head].ts_us   = now_realtime_us();
            g_pose_ring[g_pose_ring_head].pose[0] = g_eye_poses[0];
            g_pose_ring[g_pose_ring_head].pose[1] = g_eye_poses[1];
            g_pose_ring_head = (g_pose_ring_head + 1) % POSE_RING_N;
            // Log once after we first capture stereo poses, so we can
            // verify visually in terminal output that they look sensible.
            static std::atomic<bool> announced{false};
            if (!announced.exchange(true)) {
                log("first stereo poses cached: "
                    "L pos=(%.3f, %.3f, %.3f) R pos=(%.3f, %.3f, %.3f)",
                    g_eye_poses[0].px, g_eye_poses[0].py, g_eye_poses[0].pz,
                    g_eye_poses[1].px, g_eye_poses[1].py, g_eye_poses[1].pz);
                const float dx = g_eye_poses[1].px - g_eye_poses[0].px;
                const float dy = g_eye_poses[1].py - g_eye_poses[0].py;
                const float dz = g_eye_poses[1].pz - g_eye_poses[0].pz;
                const float baseline = std::sqrt(dx*dx + dy*dy + dz*dz);
                log("  → measured stereo baseline = %.1f mm", baseline * 1000.f);
            }
            break;   // first projection layer only
        }
    }

    // ── Overlay injection ────────────────────────────────────────────────
    // Enabled by env XR_OVERLAY=1. Pulls the latest stereo overlay received
    // from the pipeline, uploads it into the layer-owned array swapchain
    // (layer 0 = left eye, 1 = right), and appends an XrCompositionLayer
    // Projection reusing the captured eye pose+fov so the overlay composites
    // at the full eye FoV — true size, aligned with the scene, feedback-free
    // (the capture path only reads the app's projection swapchain).
    static int s_overlay = -1;            // -1 unknown, 0 off, 1 on
    if (s_overlay < 0) {
        const char* e = getenv("XR_OVERLAY");
        s_overlay = (e && e[0] && e[0] != '0') ? 1 : 0;
    }

    bool injected = false;
    XrResult r;

    if (s_overlay) {
        // Re-upload the most recent mask EVERY composited frame. We keep a
        // persistent copy and acquire→copy→release the overlay swapchain each
        // frame, because this runtime does NOT retain a released swapchain
        // image across frames: re-submitting a stale one made the overlay
        // blink at the send-vs-display beat (~20 Hz content vs ~90 Hz display),
        // seen as a both-eye flicker. Content (mono8, ~1 MB) is cheap to re-copy.
        static std::vector<uint8_t> s_lastL, s_lastR;
        static uint64_t s_cap_ts = 0;   // capture_ts_us of the held overlay content
        static uint32_t s_lw = 0, s_lh = 0, s_leyes = 0;
        float fov[2][4]; bool have_fov = false;
        {
            std::lock_guard<std::mutex> lk(g_ovl_mu);
            if (g_ovl_fresh && !g_ovl_pix[0].empty()) {
                s_lastL = g_ovl_pix[0];
                s_leyes = g_ovl_eyes;
                if (s_leyes >= 2 && !g_ovl_pix[1].empty()) s_lastR = g_ovl_pix[1];
                else s_lastR.clear();
                s_lw = g_ovl_w; s_lh = g_ovl_h;
                s_cap_ts = g_ovl_capture_ts_us;   // anchor pose lookup to this
                g_ovl_fresh = false;
            }
            // Always snapshot the latest fov (constant across frames).
            have_fov = g_ovl_have_fov;
            for (int e = 0; e < 2; ++e)
                for (int k = 0; k < 4; ++k) fov[e][k] = g_ovl_fov[e][k];
        }
        // Persist the fov across frames (constant once received).
        static float s_fov[2][4]; static bool s_have_fov = false;
        if (have_fov) {
            for (int e = 0; e < 2; ++e)
                for (int k = 0; k < 4; ++k) s_fov[e][k] = fov[e][k];
            s_have_fov = true;
        }
        if (s_lw > 0 && s_lh > 0 && !s_lastL.empty()) {
            const uint8_t* pR = (s_leyes >= 2 && !s_lastR.empty()) ? s_lastR.data() : nullptr;
            if (overlay_ensure(s, s_lw, s_lh) &&
                overlay_upload_stereo(s_lastL.data(), pR, s_lw, s_lh)) {
                g_overlay.has_content = true;
                static std::atomic<bool> first_content{false};
                if (!first_content.exchange(true))
                    log("overlay: first frame uploaded (%ux%u, eyes=%u, seq=%u, fov=%d)",
                        s_lw, s_lh, s_leyes, g_ovl_seq, (int)s_have_fov);
            }
        }

        // Snapshot the persisted projection params (updated whenever the app
        // last submitted a projection layer). Injecting from this — rather than
        // requiring a projection layer THIS frame — means the overlay never
        // blinks out on frames that lack one.
        PersistProj pp;
        EyePose  cap_pose[2];
        bool     have_cap_pose = false;
        {
            std::lock_guard<std::mutex> lk(g_pose_mu);
            pp = g_pproj;
            if (s_cap_ts != 0)
                have_cap_pose = pose_ring_lookup(s_cap_ts, cap_pose);
        }

        if (g_overlay.has_content && g_overlay.handle != XR_NULL_HANDLE
                && pp.seen && pp.space != XR_NULL_HANDLE) {
            const uint32_t nv = (pp.nv >= 2) ? 2u : 1u;
            static XrCompositionLayerProjectionView pv[2];
            for (uint32_t v = 0; v < nv; ++v) {
                pv[v].type     = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
                pv[v].next     = nullptr;
                // Anchor to the eye pose at CAPTURE time (when the mask was
                // rendered) so the runtime reprojects/timewarps the overlay to
                // the current head pose — removes head-motion swim. Fall back
                // to the current projection pose if the ring had no match
                // (no worse than before).
                pv[v].pose     = (have_cap_pose && cap_pose[v].valid)
                                 ? eyepose_to_xrposef(cap_pose[v])
                                 : pp.pose[v];
                // Use the pipeline-camera fov the overlay was rendered with
                // (matches the mask's projection → aligned). Fall back to the
                // captured native eye fov if none was sent.
                if (s_have_fov) {
                    pv[v].fov.angleLeft  = s_fov[v][0];
                    pv[v].fov.angleRight = s_fov[v][1];
                    pv[v].fov.angleUp    = s_fov[v][2];
                    pv[v].fov.angleDown  = s_fov[v][3];
                } else {
                    pv[v].fov = pp.fov[v];
                }
                pv[v].subImage.swapchain = g_overlay.handle;
                pv[v].subImage.imageRect.offset = {(int32_t)(v * g_overlay.w), 0};
                pv[v].subImage.imageRect.extent = {(int32_t)g_overlay.w,
                                                   (int32_t)g_overlay.h};
                pv[v].subImage.imageArrayIndex = 0;   // single-layer double-wide
            }
            static XrCompositionLayerProjection ovl_proj{};
            ovl_proj.type       = XR_TYPE_COMPOSITION_LAYER_PROJECTION;
            ovl_proj.next       = nullptr;
            ovl_proj.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            ovl_proj.space      = pp.space;
            ovl_proj.viewCount  = nv;
            ovl_proj.views      = pv;

            std::vector<const XrCompositionLayerBaseHeader*> nl;
            nl.reserve(info->layerCount + 1);
            for (uint32_t i = 0; i < info->layerCount; ++i)
                nl.push_back(info->layers[i]);
            nl.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&ovl_proj));

            XrFrameEndInfo ni = *info;
            ni.layerCount = (uint32_t)nl.size();
            ni.layers     = nl.data();

            static std::atomic<bool> once{false};
            if (!once.exchange(true))
                log("overlay: injecting projection layer, %u views", nv);
            {
                static std::atomic<int> anchlog{0};
                int al = anchlog.fetch_add(1);
                if (al < 3 || al % 300 == 0) {
                    double age_ms = (s_cap_ts && have_cap_pose)
                        ? (double)(now_realtime_us() - s_cap_ts) / 1000.0 : 0.0;
                    log("overlay: pose-anchor=%s, overlay age %.0f ms",
                        have_cap_pose ? "CAPTURE-TIME" : "current(fallback)",
                        age_ms);
                }
            }
            r = nd.EndFrame ? nd.EndFrame(s, &ni) : XR_SUCCESS;
            injected = true;
            static std::atomic<bool> rl{false};
            if (!rl.exchange(true))
                log("overlay: EndFrame(with overlay) returned %d", (int)r);
        }

        // Frame accounting: how many frames inject vs skip, and why. If we
        // inject ~every frame but it still flickers, the runtime is dropping
        // the secondary projection layer (not us skipping it).
        {
            static std::atomic<int> c_inj{0}, c_nocontent{0}, c_noproj{0},
                                    c_thisframe_noproj{0}, c_tot{0};
            ++c_tot;
            if (injected) ++c_inj;
            else if (!g_overlay.has_content) ++c_nocontent;
            else if (!pp.seen) ++c_noproj;
            if (g_proj_for_inject == nullptr) ++c_thisframe_noproj;
            int t = c_tot.load();
            if (t % 300 == 0)
                log("overlay-acct: tot=%d inj=%d skip(nocontent)=%d "
                    "skip(neverproj)=%d frames-without-proj-layer=%d",
                    t, c_inj.load(), c_nocontent.load(), c_noproj.load(),
                    c_thisframe_noproj.load());
        }
    }

    // Socket polling and GET handling live entirely in the send worker
    // thread (see send_worker_run). The render thread's only job here
    // is to forward EndFrame to the runtime and then refresh the
    // staging ring.

    // Forward EndFrame FIRST. The runtime releases the image and the
    // GPU queue drains before we submit the staging copy. (We tested
    // capturing BEFORE forwarding — same result, since the foveation
    // warp is applied by the app pre-submit.)
    if (!injected)
        r = nd.EndFrame ? nd.EndFrame(s, info) : XR_SUCCESS;

    // Gated on a connected client so we don't burn GPU+PCIe when idle.
    if (g_client_fd >= 0) do_continuous_capture();

    if (n == 1 || n == 30 || n == 100 || n % 600 == 0)
        log("STEP EndFrame #%d (socket=%d, vk=%d, client=%d, scs=%zu)",
            n, (int)g_socket_up.load(), (int)g_vk.loaded, g_client_fd, g_color_scs.size());
    return r;
}

XrResult XRAPI_CALL h_PollEvent(XrInstance i, XrEventDataBuffer* eb) {
    auto nd = get_first_dispatch();
    return nd.PollEvent ? nd.PollEvent(i, eb) : XR_EVENT_UNAVAILABLE;
}

XrResult XRAPI_CALL h_DestroyInstance(XrInstance instance) {
    log("STEP DestroyInstance enter");
    socket_close_all();
    if (g_opaque_enabled.load()) { OpaqueSender_Destroy(); g_opaque_created.store(false); }
    if (g_vk.libvulkan) { dlclose(g_vk.libvulkan); g_vk.libvulkan = nullptr; }
    g_vk.loaded = false;
    PFN_xrDestroyInstance next = nullptr;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_inst.find(instance);
      if (it != g_inst.end()) { next = it->second.DestroyInstance; g_inst.erase(it); } }
    return next ? next(instance) : XR_SUCCESS;
}

// Forward xrGetSystem; when XR_OPAQUE_DATA_CHANNEL=1, use the systemId the app
// obtains to create the opaque data channel exactly once. The channel starts in
// CONNECTING and flips to CONNECTED when the CloudXR client opens it; sends
// (driven by the app via OpaqueSender_SendExtrinsic) no-op until then.
XrResult XRAPI_CALL h_GetSystem(
    XrInstance instance, const XrSystemGetInfo* getInfo, XrSystemId* systemId)
{
    PFN_xrGetSystem next = nullptr;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_inst.find(instance);
      if (it == g_inst.end()) return XR_ERROR_HANDLE_INVALID;
      next = it->second.GetSystem; }
    if (!next) return XR_ERROR_FUNCTION_UNSUPPORTED;
    XrResult r = next(instance, getInfo, systemId);
    if (r == XR_SUCCESS && systemId &&
        g_opaque_enabled.load() && !g_opaque_created.exchange(true)) {
        if (OpaqueSender_Create(instance, *systemId))
            log("  [opaque] channel created (systemId=%llu)",
                (unsigned long long)*systemId);
        else
            g_opaque_created.store(false);   // creation failed; retry next xrGetSystem
    }
    return r;
}

// ── Vulkan-enable2 device-creation hooks (diagnostic → injection site) ──
// Confirms the app drives device creation through OpenXR (enable2) and dumps
// the requested device extensions + queue-create-infos. Safe under enable1:
// the loader never queries these, so the hooks are simply never installed.
XrResult XRAPI_CALL h_GetVulkanDeviceExtensionsKHR(
    XrInstance instance, XrSystemId systemId,
    uint32_t bufferCapacityInput, uint32_t* bufferCountOutput, char* buffer)
{
    PFN_xrGetInstanceProcAddr nextGetProc = nullptr;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_inst.find(instance);
      if (it == g_inst.end()) return XR_ERROR_HANDLE_INVALID;
      nextGetProc = it->second.GetInstanceProcAddr; }
    PFN_xrGetVulkanDeviceExtensionsKHR next = nullptr;
    if (nextGetProc) nextGetProc(instance, "xrGetVulkanDeviceExtensionsKHR",
                                 reinterpret_cast<PFN_xrVoidFunction*>(&next));
    if (!next) return XR_ERROR_FUNCTION_UNSUPPORTED;
    XrResult r = next(instance, systemId, bufferCapacityInput,
                      bufferCountOutput, buffer);
    if (buffer && bufferCountOutput && *bufferCountOutput)
        log("STEP GetVulkanDeviceExtensionsKHR — runtime-required dev exts: %s", buffer);
    else
        log("STEP GetVulkanDeviceExtensionsKHR (size query, cap=%u)", bufferCapacityInput);
    return r;
}

XrResult XRAPI_CALL h_CreateVulkanDeviceKHR(
    XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo,
    VkDevice* vulkanDevice, VkResult* vulkanResult)
{
    log("STEP CreateVulkanDeviceKHR enter — enable2 CONFIRMED (this is the injection site)");
    PFN_xrGetInstanceProcAddr nextGetProc = nullptr;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_inst.find(instance);
      if (it == g_inst.end()) return XR_ERROR_HANDLE_INVALID;
      nextGetProc = it->second.GetInstanceProcAddr; }
    PFN_xrCreateVulkanDeviceKHR next = nullptr;
    if (nextGetProc) nextGetProc(instance, "xrCreateVulkanDeviceKHR",
                                 reinterpret_cast<PFN_xrVoidFunction*>(&next));
    if (!next) { log("  no next xrCreateVulkanDeviceKHR"); return XR_ERROR_FUNCTION_UNSUPPORTED; }

    if (createInfo && createInfo->vulkanCreateInfo) {
        const VkDeviceCreateInfo* dci = createInfo->vulkanCreateInfo;
        log("  app requested %u device extensions:", dci->enabledExtensionCount);
        for (uint32_t i = 0; i < dci->enabledExtensionCount; ++i)
            log("    [%u] %s", i, dci->ppEnabledExtensionNames[i]);
        log("  app requested %u queue-create-info(s)", dci->queueCreateInfoCount);
        for (uint32_t i = 0; i < dci->queueCreateInfoCount; ++i)
            log("    queue[%u] family=%u count=%u", i,
                dci->pQueueCreateInfos[i].queueFamilyIndex,
                dci->pQueueCreateInfos[i].queueCount);
    }
    // Diagnostic: forward unchanged. Injection (exts + transfer queue) lands here next.
    return next(instance, createInfo, vulkanDevice, vulkanResult);
}

XrResult XRAPI_CALL h_GetInstanceProcAddr(
    XrInstance instance, const char* name, PFN_xrVoidFunction* function)
{
    if (!function) return XR_ERROR_VALIDATION_FAILURE;
#define HOOK(N, FN)                                                              \
    if (std::strcmp(name, "xr" #N) == 0) {                                       \
        *function = reinterpret_cast<PFN_xrVoidFunction>(&FN);                   \
        return XR_SUCCESS;                                                       \
    }
    HOOK(GetSystem,             h_GetSystem);
    HOOK(CreateSession,         h_CreateSession);
    HOOK(DestroySession,        h_DestroySession);
    HOOK(BeginSession,          h_BeginSession);
    HOOK(EndSession,            h_EndSession);
    HOOK(CreateSwapchain,       h_CreateSwapchain);
    HOOK(DestroySwapchain,      h_DestroySwapchain);
    HOOK(AcquireSwapchainImage, h_AcquireSwapchainImage);
    HOOK(ReleaseSwapchainImage, h_ReleaseSwapchainImage);
    HOOK(EndFrame,              h_EndFrame);
    HOOK(PollEvent,             h_PollEvent);
    HOOK(DestroyInstance,       h_DestroyInstance);
    HOOK(CreateVulkanDeviceKHR,        h_CreateVulkanDeviceKHR);
    HOOK(GetVulkanDeviceExtensionsKHR, h_GetVulkanDeviceExtensionsKHR);
#undef HOOK
    PFN_xrGetInstanceProcAddr nextGetProc = nullptr;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_inst.find(instance);
      if (it == g_inst.end()) return XR_ERROR_HANDLE_INVALID;
      nextGetProc = it->second.GetInstanceProcAddr; }
    return nextGetProc ? nextGetProc(instance, name, function) : XR_ERROR_HANDLE_INVALID;
}

XrResult XRAPI_CALL h_CreateApiLayerInstance(
    const XrInstanceCreateInfo*  info,
    const XrApiLayerCreateInfo*  apiLayerInfo,
    XrInstance*                  instance)
{
    log("STEP CreateApiLayerInstance enter");
    if (!apiLayerInfo || !apiLayerInfo->nextInfo) return XR_ERROR_INITIALIZATION_FAILED;
    PFN_xrGetInstanceProcAddr     nextGetProc = apiLayerInfo->nextInfo->nextGetInstanceProcAddr;
    PFN_xrCreateApiLayerInstance  nextCreate  = apiLayerInfo->nextInfo->nextCreateApiLayerInstance;
    XrApiLayerCreateInfo nextLayer = *apiLayerInfo;
    nextLayer.nextInfo = apiLayerInfo->nextInfo->next;

    // ── Optional foveation extension strip ─────────────────────────────
    // XR_FRAME_STRIP_FOVEATION=1 filters out the NVX1 foveation extension
    // so the swapchain holds pinhole content (required for geometric
    // work). See XR_FRAME_LAYER_ARCHITECTURE.md.
    const XrInstanceCreateInfo* info_to_pass = info;
    XrInstanceCreateInfo info_patched{};
    std::vector<const char*> exts;
    const char* strip_env  = getenv("XR_FRAME_STRIP_FOVEATION");
    const char* opaque_env = getenv("XR_OPAQUE_DATA_CHANNEL");
    const bool  strip_fov   = strip_env  && strip_env[0]  == '1';
    const bool  want_opaque = opaque_env && opaque_env[0] == '1';
    g_opaque_enabled.store(want_opaque);
    if (info && (strip_fov || want_opaque)) {
        bool changed = false;
        if (info->enabledExtensionNames) {
            exts.reserve(info->enabledExtensionCount + 1);
            for (uint32_t i = 0; i < info->enabledExtensionCount; ++i) {
                const char* name = info->enabledExtensionNames[i];
                if (strip_fov && name &&
                    std::strcmp(name, "XR_NVX1_foveation_piecewise_quadratic_warp") == 0) {
                    log("  STRIPPING extension from app request: %s", name);
                    changed = true;
                    continue;
                }
                exts.push_back(name);
            }
        }
        if (want_opaque) {
            bool present = false;
            for (const char* n : exts)
                if (n && std::strcmp(n, XR_NV_OPAQUE_DATA_CHANNEL_EXTENSION_NAME) == 0)
                    present = true;
            if (!present) {
                exts.push_back(XR_NV_OPAQUE_DATA_CHANNEL_EXTENSION_NAME);
                changed = true;
                log("  ADDING extension: %s", XR_NV_OPAQUE_DATA_CHANNEL_EXTENSION_NAME);
            }
        }
        if (changed) {
            info_patched = *info;
            info_patched.enabledExtensionCount = (uint32_t)exts.size();
            info_patched.enabledExtensionNames = exts.data();
            info_to_pass = &info_patched;
            log("  enabledExtensionCount: %u → %u",
                info->enabledExtensionCount, info_patched.enabledExtensionCount);
        }
    }

    XrResult r = nextCreate(info_to_pass, &nextLayer, instance);
    if (r != XR_SUCCESS) return r;

    XrDispatch nd;
    nd.GetInstanceProcAddr = nextGetProc;
#define LOAD(NAME)                                                              \
    do { PFN_xrVoidFunction fn = nullptr;                                       \
         if (nextGetProc(*instance, "xr" #NAME, &fn) == XR_SUCCESS)             \
             nd.NAME = reinterpret_cast<PFN_xr##NAME>(fn); } while (0)
    LOAD(DestroyInstance);
    LOAD(GetSystem);
    LOAD(CreateSession);  LOAD(DestroySession);
    LOAD(BeginSession);   LOAD(EndSession);
    LOAD(CreateSwapchain); LOAD(DestroySwapchain);
    LOAD(EnumerateSwapchainImages);
    LOAD(AcquireSwapchainImage); LOAD(ReleaseSwapchainImage);
    LOAD(EndFrame);
    LOAD(PollEvent);
    LOAD(WaitSwapchainImage);
#undef LOAD
    { std::lock_guard<std::mutex> lk(g_mu); g_inst[*instance] = nd; }
    if (g_opaque_enabled.load()) {
        OpaqueSender_ResolveProcs(*instance, nextGetProc);
        log("  [opaque] procs resolved; channel created at first xrGetSystem");
    }
    log("STEP CreateApiLayerInstance exit");
    return XR_SUCCESS;
}

} // anon namespace

extern "C" __attribute__((visibility("default")))
XrResult XRAPI_CALL xrNegotiateLoaderApiLayerInterface(
    const XrNegotiateLoaderInfo* loaderInfo,
    const char*                  layerName,
    XrNegotiateApiLayerRequest*  apiLayerRequest)
{
    if (!loaderInfo || !apiLayerRequest || !layerName) return XR_ERROR_INITIALIZATION_FAILED;
    if (loaderInfo->structType    != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO)         return XR_ERROR_INITIALIZATION_FAILED;
    if (apiLayerRequest->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST) return XR_ERROR_INITIALIZATION_FAILED;
    if (std::strcmp(layerName, LAYER_NAME) != 0) return XR_ERROR_INITIALIZATION_FAILED;

    apiLayerRequest->layerInterfaceVersion  = XR_CURRENT_LOADER_API_LAYER_VERSION;
    apiLayerRequest->layerApiVersion        = XR_CURRENT_API_VERSION;
    apiLayerRequest->getInstanceProcAddr    = &h_GetInstanceProcAddr;
    apiLayerRequest->createApiLayerInstance = &h_CreateApiLayerInstance;
    return XR_SUCCESS;
}