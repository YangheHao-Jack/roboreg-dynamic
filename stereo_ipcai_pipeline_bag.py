#!/usr/bin/env python3
"""
Stereo IPCAI per-frame processing — shared library.

The per-frame seg + opt core plus rosbag2 reading helpers. NOT a runnable
script. The single entry point for offline / paced-bag / live ROS input is
`stereo_pipeline_live.py`, which imports from here.

Public API used by `stereo_pipeline_live.py`:
    process_one_frame_dual_stream     per-frame seg + opt core
    init_seg_model_and_graph          load seg model, capture CUDA graph
    init_processing_state             init dual-stream manager + optimizer cache
    BagStreamReader, load_bag_data    synchronised triplet reader
    _decode_image_msg_to_numpy        Image | CompressedImage -> RGB numpy (CPU)
                                       — CompressedImage paths route through GPU
                                       (nvJPEG for JPEG, PyAV NVDEC for H.264)
                                       and bounce back to numpy; falls back to
                                       cv2 only when CUDA isn't available.
    _decode_image_msg_to_gpu_chw      Image | CompressedImage -> CHW uint8 CUDA
                                       — supports both JPEG (nvJPEG) and H.264
                                       (PyAV h264_cuvid). H.264 decoder state
                                       is cached per msg.header.frame_id.
    _stamp_to_ns, _detect_bag_storage_id
    _compute_event_stats              CUDA-event timing helper
    DualStreamManager, OptimizerType  classes
    dual_stream_mgr                   module-level singleton
    _cache                            module-level optimizer cache
"""
import os
import time
import yaml
import numpy as np
import cv2
import torch
import pytorch_kinematics as pk
from enum import Enum
import pathlib
import gdown
import torch.nn.functional as F
import kornia
from torchvision.utils import draw_segmentation_masks
from collections import deque

# ── Lazy H.264 decoder (PyAV / NVDEC) ────────────────────────────────
# Only imported on first H.264 CompressedImage encounter, so JPEG-only
# bags still work in envs without PyAV.
_AV_MODULE = None
_H264_DECODERS = {}        # frame_id -> av.CodecContext, persistent
                           # across frames within a stream (inter-frame deps).
_H264_BACKEND_LOGGED = False


def _ensure_av():
    """Import PyAV on first use. Raises with an install hint if missing."""
    global _AV_MODULE
    if _AV_MODULE is None:
        try:
            import av as _av  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Decoding H.264 CompressedImage requires PyAV. "
                "Install with: pip install av")
        _AV_MODULE = _av
    return _AV_MODULE


def _get_h264_decoder(frame_id: str):
    """Return a persistent H.264 decoder for a given stream id.

    H.264 has inter-frame dependencies (P-frames reference earlier
    frames within the GOP) and a stateful parser (SPS/PPS are parsed
    on the first IDR and cached internally). We must reuse the same
    CodecContext across frames of one stream.

    `frame_id` is `msg.header.frame_id` — distinct between left and
    right cameras when capturing into a bag. If the bag has empty
    frame_ids, we fall back to a single "default" decoder, which
    will conflate left and right streams (warning emitted once).
    """
    global _H264_BACKEND_LOGGED
    av = _ensure_av()
    key = frame_id or "default"
    if key not in _H264_DECODERS:
        # Prefer NVDEC (h264_cuvid) — falls back to CPU h264 if
        # cuvid isn't compiled into the linked ffmpeg.
        try:
            codec = av.CodecContext.create("h264_cuvid", "r")
            backend = "h264_cuvid (NVDEC)"
        except Exception as e:
            codec = av.CodecContext.create("h264", "r")
            backend = f"h264 (CPU fallback: {e})"
        _H264_DECODERS[key] = codec
        if not _H264_BACKEND_LOGGED:
            print(f"[stereo_ipcai_bag] H.264 decoder backend: {backend}")
            _H264_BACKEND_LOGGED = True
        if not frame_id:
            print(f"[stereo_ipcai_bag] WARN: empty msg.header.frame_id; "
                  f"left and right streams may share one decoder, "
                  f"which corrupts inter-frame decode. Set distinct "
                  f"frame_ids upstream.")
    return _H264_DECODERS[key]


# Raised when an H.264 decoder consumes a packet but the resulting
# frame isn't ready yet — most often because the bag starts mid-GOP
# (the first packet is a P-frame; the decoder needs an IDR + SPS/PPS
# before it can emit anything). Callers that read synchronised
# triplets should catch this and skip to the next triplet.
class H264NeedsMorePackets(Exception):
    pass


def _reset_h264_decoders():
    """Drop all cached decoders. Useful between bag plays / unit
    tests where the SPS/PPS may change."""
    _H264_DECODERS.clear()


class REGISTRATION_MODE(Enum):
    DISTANCE_FUNCTION = "distance-function"
    SEGMENTATION = "segmentation"

# === PYTORCH OPTIMIZATIONS ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    pass
torch.set_grad_enabled(True)
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except AttributeError:
    pass

# GLOBAL STATE
t_small_static: torch.Tensor = None
seg_graph: torch.cuda.CUDAGraph = None
prob_small_static: torch.Tensor = None
dual_stream_mgr = None
coarse_model = None
device = None
args = None


class OptimizerType(Enum):
    IPCAI = "ipcai"
    LBFGS = "lbfgs"  # stub for save_results compatibility


# === DUAL STREAM ARCHITECTURE ===
class DualStreamManager:
    """Two-stream CUDA manager for overlapped seg + opt.

    Streams have fixed roles:
        stream_opt   — opt's inner loop runs here (high priority).
        stream_seg   — seg-model prefetch runs here (default priority).
        stream_display — visualization / video write (default priority).

    Earlier versions of this class swapped the role of two physical streams
    each iteration via `swap_streams()`. That made the cross-stream tensor
    handoff implicit (the prefetched seg's stream became the new "current"
    stream so opt could consume the tensor without a sync), but it also
    meant CUDA stream priorities couldn't be assigned meaningfully — both
    physical streams had to be equal-priority because their roles rotated.
    With fixed roles we can set `stream_opt` to high priority so the GPU
    scheduler prefers opt's many small kernels over seg's larger ones when
    they collide on SMs (seg-prefetch on `stream_seg` overlaps with opt
    on `stream_opt`). The cross-stream tensor handoff is now made explicit
    by `current_stream.wait_stream(next_stream)` in
    `process_one_frame_dual_stream`.

    The legacy `current/next` accessors and `swap_streams()` are preserved
    so callers don't need to change. They now return the fixed roles and
    swap is a no-op.
    """
    def __init__(self):
        # CUDA stream priorities: lower number = higher priority. Most
        # NVIDIA GPUs (Ampere/Ada/Hopper) support [-1, 0]; some support more.
        # We use the implementation-reported high-priority value so this
        # works on any supported card.
        try:
            high_pri, _low_pri = torch.cuda.get_stream_priority_range()
        except Exception:
            high_pri = -1
        self.stream_opt = torch.cuda.Stream(priority=high_pri)
        self.stream_seg = torch.cuda.Stream(priority=0)
        self.stream_display = torch.cuda.Stream(priority=0)
        # Backwards-compat aliases — old code path still uses these names.
        self.stream_current = self.stream_opt
        self.stream_next = self.stream_seg

    def get_current_stream(self):
        return self.stream_opt

    def get_next_stream(self):
        return self.stream_seg

    def swap_streams(self):
        # No-op. Kept so call sites don't need to change.
        pass

    def synchronize_current(self):
        self.stream_opt.synchronize()

    def synchronize_all(self):
        self.stream_opt.synchronize()
        self.stream_seg.synchronize()
        self.stream_display.synchronize()


# === SEGMENTATION ===
@torch.no_grad()
def generate_wireframe_overlay(pred_mask, gt_mask, base_image=None,
                               pred_color=(0, 255, 255), gt_color=(255, 255, 0), thickness=2):
    """Generate wireframe contours from prediction and ground truth masks."""
    H, W = pred_mask.shape
    canvas_np = base_image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8).copy() if base_image is not None else np.zeros((H, W, 3), dtype=np.uint8)
    pred_contours, _ = cv2.findContours(pred_mask.cpu().numpy().astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask.cpu().numpy().astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas_np, gt_contours, -1, gt_color, thickness)
    cv2.drawContours(canvas_np, pred_contours, -1, pred_color, thickness)
    return torch.from_numpy(canvas_np).permute(2, 0, 1).to(pred_mask.device)


@torch.no_grad()
def extract_mask(img_batch, stream):
    """Extract segmentation mask at the model's 576x960 input, then upsample to
    the original image size.

    Default: stretch the input to 576x960 (distorts aspect ratio when the input
    isn't ~1.667:1 — e.g. 16:9 1080p gets a horizontal squeeze).

    With ``args.preserve_aspect`` (``--preserve-aspect``): letterbox instead —
    resize preserving aspect ratio to fit inside 576x960, zero-pad to fill the
    buffer, run the (graph-captured, fixed-shape) model, then crop the mask back
    to the valid (unpadded) region before upsampling to the original size. The
    576x960 graph input shape is unchanged, so the CUDA graph still applies.
    """
    global t_small_static, prob_small_static, seg_graph, args
    TARGET_H, TARGET_W = 576, 960
    with torch.cuda.stream(stream):
        H0, W0 = int(img_batch.shape[2]), int(img_batch.shape[3])
        if getattr(args, 'preserve_aspect', False):
            scale = min(TARGET_H / H0, TARGET_W / W0)
            new_h = min(TARGET_H, max(1, int(round(H0 * scale))))
            new_w = min(TARGET_W, max(1, int(round(W0 * scale))))
            pad_top = (TARGET_H - new_h) // 2
            pad_bottom = TARGET_H - new_h - pad_top
            pad_left = (TARGET_W - new_w) // 2
            pad_right = TARGET_W - new_w - pad_left
            resized = F.interpolate(img_batch, (new_h, new_w), mode='bilinear',
                                    align_corners=False, antialias=False)
            tmp = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            t_small_static.copy_(tmp)
            seg_graph.replay()
            prob_small = prob_small_static[:, :, pad_top:pad_top + new_h,
                                           pad_left:pad_left + new_w].float()
        else:
            tmp = F.interpolate(img_batch, (TARGET_H, TARGET_W), mode='bilinear',
                                align_corners=False, antialias=False)
            t_small_static.copy_(tmp)
            seg_graph.replay()
            prob_small = prob_small_static.float()
        # Threshold on the small (model-resolution) prob, THEN upsample. Same
        # information as upsample-then-threshold (the mask is only ~576x960 real
        # resolution; the upsample adds no true detail), but the snap runs on
        # ~0.55 M elements instead of the full-res 2*H*W, and torch.where avoids
        # the separate full-res clone. Soft target semantics are preserved: soft
        # prob below pth, snapped to 1.0 at/above pth. NOTE: not bit-identical to
        # the old order at the boundary band (coarse snap then upsample gives a
        # slightly softer edge than upsample then snap).
        thresh_small = torch.where(prob_small > args.pth,
                                   torch.ones_like(prob_small), prob_small)
        prob_out = F.interpolate(thresh_small, size=(H0, W0), mode='bilinear',
                                 align_corners=False, antialias=False)
        return prob_out


# === MASK AND OPTIMIZATION UTILITIES ===
def extract_mask_region(mask, render_mask, margin=5):
    """Extract ROI using dilated rendered mask shape."""
    if render_mask is None:
        return mask, None
    mask_2d = mask
    while mask_2d.dim() > 2:
        mask_2d = mask_2d.squeeze(0) if mask_2d.shape[0] == 1 else (mask_2d.squeeze(-1) if mask_2d.shape[-1] == 1 else mask_2d[0])
    render_mask_2d = render_mask
    while render_mask_2d.dim() > 2:
        render_mask_2d = render_mask_2d.squeeze(0) if render_mask_2d.shape[0] == 1 else (render_mask_2d.squeeze(-1) if render_mask_2d.shape[-1] == 1 else render_mask_2d[0])
    if mask_2d.dim() != 2 or render_mask_2d.dim() != 2:
        return mask, None

    render_binary = (render_mask_2d > 0.5).float().unsqueeze(0).unsqueeze(0)
    kernel_size = 2 * margin + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=render_binary.device, dtype=render_binary.dtype)
    dilated_roi = (F.conv2d(render_binary, kernel, padding=margin).squeeze() > 0.5).float()

    nonzero = torch.nonzero(dilated_roi > 0.5, as_tuple=False)
    if nonzero.shape[0] == 0:
        return mask, None
    y_min, y_max = nonzero[:, 0].min().item(), nonzero[:, 0].max().item()
    x_min, x_max = nonzero[:, 1].min().item(), nonzero[:, 1].max().item()

    crop_coords = {
        'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max,
        'original_shape': mask.shape,
        'roi_mask': dilated_roi[y_min:y_max+1, x_min:x_max+1].contiguous()
    }
    return mask_2d[y_min:y_max+1, x_min:x_max+1].contiguous(), crop_coords


def gpu_gaussian_blur(mask, kernel_size=65, sigma=15.0):
    """Apply Gaussian blur to binary mask to create gradient field."""
    mask_bchw = mask.permute(0, 3, 1, 2).contiguous()
    binary_mask = (mask_bchw > 0.5).float()
    blurred = kornia.filters.gaussian_blur2d(binary_mask, (kernel_size, kernel_size), (sigma, sigma))
    result = torch.where(binary_mask > 0.0, torch.ones_like(blurred), blurred)
    return result.permute(0, 2, 3, 1).contiguous()


def compute_mask_moments_cropped(mask, crop_coords=None):
    """Compute image moments for orientation estimation on cropped mask."""
    if mask.dim() != 2:
        return None
    masked_prob = mask * crop_coords['roi_mask'] if (crop_coords is not None and 'roi_mask' in crop_coords) else mask
    m00 = masked_prob.sum()
    if m00 < 1e-6:
        return None
    h, w = masked_prob.shape
    y_coords = torch.arange(h, device=masked_prob.device, dtype=torch.float32).view(-1, 1)
    x_coords = torch.arange(w, device=masked_prob.device, dtype=torch.float32).view(1, -1)
    cy_local = (y_coords * masked_prob).sum() / m00
    cx_local = (x_coords * masked_prob).sum() / m00
    cy_global = cy_local + crop_coords['y_min'] if crop_coords else cy_local
    cx_global = cx_local + crop_coords['x_min'] if crop_coords else cx_local
    dy, dx = y_coords - cy_local, x_coords - cx_local
    mu20 = (dx * dx * masked_prob).sum() / m00
    mu02 = (dy * dy * masked_prob).sum() / m00
    mu11 = (dx * dy * masked_prob).sum() / m00
    angle = 0.0 if abs(mu20 - mu02) < 1e-6 else 0.5 * torch.atan2(2 * mu11, mu20 - mu02)
    return {'centroid': (cx_global, cy_global), 'angle': angle, 'mu20': mu20, 'mu02': mu02, 'mu11': mu11, 'area': m00}


def compute_cropped_soft_iou(prob1, prob2, crop1, crop2):
    """Compute soft IoU between two cropped probability maps."""
    if crop1 is None or crop2 is None:
        intersection = torch.minimum(prob1, prob2).sum()
        union = torch.maximum(prob1, prob2).sum()
        return intersection / (union + 1e-8)
    global_y_min = min(crop1['y_min'], crop2['y_min'])
    global_y_max = max(crop1['y_max'], crop2['y_max'])
    global_x_min = min(crop1['x_min'], crop2['x_min'])
    global_x_max = max(crop1['x_max'], crop2['x_max'])
    global_h = global_y_max - global_y_min + 1
    global_w = global_x_max - global_x_min + 1

    aligned1 = torch.zeros((global_h, global_w), device=prob1.device, dtype=prob1.dtype)
    aligned2 = torch.zeros((global_h, global_w), device=prob2.device, dtype=prob2.dtype)
    aligned_roi1 = torch.zeros((global_h, global_w), device=prob1.device, dtype=torch.float32)
    aligned_roi2 = torch.zeros((global_h, global_w), device=prob2.device, dtype=torch.float32)

    for prob, crop, aligned, aligned_roi in [(prob1, crop1, aligned1, aligned_roi1), (prob2, crop2, aligned2, aligned_roi2)]:
        ys = crop['y_min'] - global_y_min
        xs = crop['x_min'] - global_x_min
        aligned[ys:ys+prob.shape[0], xs:xs+prob.shape[1]] = prob
        if 'roi_mask' in crop:
            aligned_roi[ys:ys+prob.shape[0], xs:xs+prob.shape[1]] = crop['roi_mask']
        else:
            aligned_roi[ys:ys+prob.shape[0], xs:xs+prob.shape[1]] = 1.0

    common_roi = torch.minimum(aligned_roi1, aligned_roi2)
    masked1, masked2 = aligned1 * common_roi, aligned2 * common_roi
    return torch.minimum(masked1, masked2).sum() / (torch.maximum(masked1, masked2).sum() + 1e-8)


@torch.jit.script
def fast_stereo_tversky_loss(pL: torch.Tensor, pR: torch.Tensor,
                              tgtL: torch.Tensor, tgtR: torch.Tensor,
                              alpha: float = 0.7, beta: float = 0.3,
                              smooth: float = 1e-6) -> torch.Tensor:
    """JIT-compiled stereo Tversky loss (sum of L+R)."""
    pL_flat, tgtL_flat = pL.view(-1), tgtL.view(-1)
    TP_L = (pL_flat * tgtL_flat).sum()
    FP_L = (pL_flat * (1.0 - tgtL_flat)).sum()
    FN_L = ((1.0 - pL_flat) * tgtL_flat).sum()
    tversky_L = 1.0 - (TP_L + smooth) / (TP_L + alpha * FP_L + beta * FN_L + smooth)
    pR_flat, tgtR_flat = pR.view(-1), tgtR.view(-1)
    TP_R = (pR_flat * tgtR_flat).sum()
    FP_R = (pR_flat * (1.0 - tgtR_flat)).sum()
    FN_R = ((1.0 - pR_flat) * tgtR_flat).sum()
    tversky_R = 1.0 - (TP_R + smooth) / (TP_R + alpha * FP_R + beta * FN_R + smooth)
    return tversky_L + tversky_R


# === OPTIMIZER STATE ===
class OptimizerCache:
    """Cache for IPCAI optimizer state management."""
    def __init__(self):
        self.cached_fk_vertices = None
        self.cached_js = None
        self.fk_changed = True
        self.prev_probL_cropped = None
        self.prev_crop_coords = None
        self.prev_moments = None
        self.prev_best_pL = None
        self.frame_counter = 0
        self.smooth_lr = None
        self.smooth_betas = (0.9, 0.999)
        self.smooth_weight_decay = None

    def reset(self, base_lr, weight_decay):
        self.cached_fk_vertices = None
        self.cached_js = None
        self.fk_changed = True
        self.prev_probL_cropped = None
        self.prev_crop_coords = None
        self.prev_moments = None
        self.prev_best_pL = None
        self.frame_counter = 0
        self.smooth_lr = base_lr
        self.smooth_betas = (0.9, 0.999)
        self.smooth_weight_decay = weight_decay


_cache = OptimizerCache()

# Latest optimiser-rendered masks, stashed for the overlay to reuse (pipeline
# resolution / pipeline-camera K). None until the first frame is processed.
_overlay_pL = None
_overlay_pR = None
# Target-pose mask rendered once per frame ON THE OPT STREAM (inside the
# opt_end_event bracket, so it overlaps the seg prefetch instead of fencing the
# device from inside the viz). The viz reuses this as data — no in-viz nvdiffrast
# rasterisation, so the seg∥opt overlap is preserved under --viz-light-sync.
_overlay_target_maskL = None


# === INITIALISATION HELPERS (callable from wrapper scripts) ===
def init_seg_model_and_graph(args_obj, device_obj):
    """Load the segmentation model and capture its CUDA graph.

    Sets the module-level globals (`coarse_model`, `t_small_static`,
    `prob_small_static`, `seg_graph`) that `extract_mask` reads. Wrapper
    scripts (e.g. `stereo_pipeline_live.py`) call this so they can reuse
    `process_one_frame_dual_stream` / `extract_mask` without duplicating
    model loading.
    """
    global coarse_model, t_small_static, seg_graph, prob_small_static, args, device
    args = args_obj
    device = device_obj

    model_file = pathlib.Path(args_obj.model_path) / args_obj.model_name
    model_file.parent.mkdir(parents=True, exist_ok=True)
    if not model_file.exists():
        gdown.download(args_obj.model_url, str(model_file), quiet=False)
    coarse_model = torch.jit.load(str(model_file), map_location=device_obj).eval().to(device_obj)

    # (1) FP16 seg model (default): a transformer-class backbone at 576x960 batch-2
    # runs ~2x faster in half precision on the 5090's tensor cores than in FP32.
    # Inference-only fp16 rarely shifts the mask given the --pth threshold. Use
    # --seg-fp32 to revert if a TorchScript op errors in fp16 or accuracy drops.
    seg_dtype = torch.float32 if getattr(args_obj, 'seg_fp32', False) else torch.float16
    if seg_dtype == torch.float16:
        coarse_model = coarse_model.half()

    t_small_static = torch.empty((2, 3, 576, 960), device=device_obj, dtype=seg_dtype)
    logits_static = torch.empty((2, 1, 576, 960), device=device_obj, dtype=seg_dtype)
    prob_small_static = torch.empty_like(logits_static)
    seg_graph = torch.cuda.CUDAGraph()

    warmup_stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(warmup_stream):
        _ = coarse_model(t_small_static)
    torch.cuda.synchronize()
    with torch.no_grad(), torch.cuda.stream(warmup_stream), torch.cuda.graph(seg_graph):
        logits_static.copy_(coarse_model(t_small_static))
        prob_small_static.copy_(torch.sigmoid(logits_static))
    torch.cuda.synchronize()


def init_processing_state(args_obj, device_obj):
    """Initialise the dual-stream manager and reset the optimizer cache.

    Both are module-level singletons used by `process_one_frame_dual_stream`.
    """
    global dual_stream_mgr
    dual_stream_mgr = DualStreamManager()
    _cache.reset(args_obj.ipcai_lr, args_obj.weight_decay)


def update_fk_cache(js_buf, scene):
    """Configure robot from the current joints and return FK vertices.

    configure() re-bakes the robot mesh (_configured_vertices) every call —
    required because render_with_pose overwrites it. The cached FK clone used for
    rendering is refreshed whenever the joint state changes (live tracking: the
    robot articulates frame to frame), not cloned once. So a joint change updates
    BOTH the FK vertices and the baked mesh. _cache.fk_changed records whether
    this frame's joints differ, so the caller can invalidate the reused baseline
    render. (Offline bag with constant joints: clones on the first frame, then
    fk_changed stays False and the cached clone is reused — the original
    behaviour.)
    """
    with torch.no_grad():
        identity = torch.eye(4, device=js_buf.device, dtype=torch.float32).unsqueeze(0)
        scene.robot.configure(js_buf, identity)
        changed = (
            _cache.cached_fk_vertices is None
            or _cache.cached_js is None
            or _cache.cached_js.shape != js_buf.shape
            or not torch.equal(js_buf, _cache.cached_js))
        if changed:
            _cache.cached_fk_vertices = scene.robot._configured_vertices.clone()
            _cache.cached_js = js_buf.clone()
        _cache.fk_changed = changed
    return _cache.cached_fk_vertices


def compute_registration_loss(pL, pR, tgtL, tgtR, mode, args):
    """Compute loss: MSE (distance-function) or Tversky (segmentation)."""
    if mode == REGISTRATION_MODE.DISTANCE_FUNCTION:
        return F.mse_loss(pL, tgtL) + F.mse_loss(pR, tgtR)
    return fast_stereo_tversky_loss(pL, pR, tgtL, tgtR, args.tversky_alpha, args.tversky_beta)


def detect_motion(probL_t, render_maskL, imgL_t, img_diagonal):
    """Detect motion between frames for adaptive parameter tuning."""
    cache = _cache
    is_large_displacement, is_rotation, iou = False, False, 1.0

    probL_for_analysis = probL_t if probL_t is not None else (imgL_t.unsqueeze(0) if imgL_t.dim() == 3 else imgL_t)
    probL_cropped, crop_coords = extract_mask_region(probL_for_analysis, render_maskL, margin=5)
    if crop_coords is None:
        probL_cropped = probL_for_analysis.squeeze() if probL_for_analysis.dim() > 2 else probL_for_analysis

    current_moments = compute_mask_moments_cropped(probL_cropped, crop_coords)

    if cache.frame_counter <= 2:
        if current_moments is not None:
            cache.prev_probL_cropped = probL_cropped.detach()
            cache.prev_crop_coords = crop_coords
            cache.prev_moments = current_moments
    elif cache.prev_moments is not None and current_moments is not None:
        iou = compute_cropped_soft_iou(cache.prev_probL_cropped, probL_cropped, cache.prev_crop_coords, crop_coords)
        prev_cx, prev_cy = cache.prev_moments['centroid']
        curr_cx, curr_cy = current_moments['centroid']
        centroid_dist = torch.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
        normalized_centroid_dist = (centroid_dist / img_diagonal).item() if img_diagonal else 0
        angle_diff_raw = current_moments['angle'] - cache.prev_moments['angle']
        angle_diff = torch.abs(torch.atan2(torch.sin(angle_diff_raw), torch.cos(angle_diff_raw)))
        angle_diff_deg = angle_diff.item() * 180.0 / np.pi if isinstance(angle_diff, torch.Tensor) else angle_diff * 180.0 / np.pi

        if iou < 0.99:
            is_large_rotation = angle_diff_deg > 0.1
            is_large_translation = normalized_centroid_dist > 0.001
            is_large_displacement = is_large_rotation or is_large_translation
            is_rotation = is_large_rotation

        cache.prev_probL_cropped = probL_cropped.detach()
        cache.prev_crop_coords = crop_coords
        cache.prev_moments = current_moments

    return is_large_displacement, is_rotation, iou


def compute_adaptive_params(base_lr, base_weight_decay, is_large_displacement, is_rotation, iou):
    """Compute motion-adaptive optimizer parameters (always adaptive)."""
    if not is_large_displacement:
        return base_lr * 0.1, (0.9, 0.999), 0.1, base_weight_decay * 2.0
    elif is_rotation:
        if iou < 0.92:
            lr_mult, betas = 3.0, (0.8, 0.9)
        elif iou < 0.96:
            lr_mult, betas = 2.0, (0.85, 0.95)
        else:
            lr_mult, betas = 1.5, (0.9, 0.98)
        return base_lr * lr_mult, betas, 0.08, base_weight_decay * 0.3
    else:
        if iou < 0.93:
            lr_mult, betas = 3.0, (0.85, 0.95)
        elif iou < 0.97:
            lr_mult, betas = 2.5, (0.9, 0.98)
        else:
            lr_mult, betas = 2.0, (0.9, 0.99)
        return base_lr * lr_mult, betas, 0.15, base_weight_decay * 0.6


def smooth_adaptive_params(target_lr, target_betas, target_weight_decay, smooth_factor=0.3):
    """Apply EMA smoothing to adaptive parameters."""
    cache = _cache
    if cache.smooth_lr is None:
        cache.smooth_lr = target_lr
        cache.smooth_weight_decay = target_weight_decay
        return
    cache.smooth_lr = smooth_factor * target_lr + (1 - smooth_factor) * cache.smooth_lr
    cache.smooth_weight_decay = smooth_factor * target_weight_decay + (1 - smooth_factor) * cache.smooth_weight_decay
    old_betas = cache.smooth_betas
    cache.smooth_betas = (
        smooth_factor * target_betas[0] + (1 - smooth_factor) * old_betas[0],
        smooth_factor * target_betas[1] + (1 - smooth_factor) * old_betas[1])


def render_with_pose(scene, cached_fk_vertices, extr_9d):
    """Render scene with given pose using cached FK vertices."""
    H_base2cam = pk.se3_9d_to_matrix44(extr_9d)
    scene.robot._configured_vertices = torch.matmul(cached_fk_vertices, H_base2cam.transpose(-1, -2))
    return scene.observe_from("left"), scene.observe_from("right")


def cache_target_overlay(args, scene, js_buf, target_9d):
    """Render the target-pose mask ONCE per frame and stash it in the module
    global `_overlay_target_maskL`. Call this on the opt stream, inside the
    opt_end_event bracket — then the rasterisation overlaps the seg prefetch
    (exactly like opt's own renders), instead of running in the viz after the
    sync point where the nvdiffrast GL fence would serialise the pipeline. The
    light-sync viz then reuses this mask as data (no in-viz render). No-op when
    display/overlay is inactive or there is no target pose."""
    global _overlay_target_maskL
    _overlay_target_maskL = None
    # Only the full composite (display / save-video) draws the target wireframe;
    # the --publish-overlay mask is the optimised silhouette only, so skip this
    # render for publish-only runs.
    if not (args.display_progress or args.save_video):
        return
    if scene is None or target_9d is None:
        return
    with torch.no_grad():
        cfk = update_fk_cache(js_buf, scene)
        tgt_pL, _ = render_with_pose(scene, cfk, target_9d)
        _overlay_target_maskL = (tgt_pL[0, ..., 0] > 0.5)


# ── GPU-native visualisation helpers (used when args.gpu_viz) ────────────────
# Keep the whole overlay on the GPU: rasterise vector primitives with torch
# index-assignment (no cv2 host round-trip) and present the composited tensor
# through OpenCV's OpenGL window via a zero-copy GpuMat wrap (no D2H).

def _draw_line_gpu(img, p0, p1, color, thickness=2):
    """Rasterise a line on a CHW uint8 CUDA tensor, fully on-GPU. p0/p1 are
    (x, y) pixel coords; color is a (3,) tensor matching img's channel order.
    Samples len+1 points along the segment and stamps a (2r+1)^2 square for
    thickness — cheap for the handful of axis/arrow segments per frame."""
    H, W = int(img.shape[1]), int(img.shape[2])
    x0, y0, x1, y1 = float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
    n = max(2, int(np.hypot(x1 - x0, y1 - y0)) + 1)
    tt = torch.linspace(0.0, 1.0, n, device=img.device)
    xs = (x0 + (x1 - x0) * tt).round().long()
    ys = (y0 + (y1 - y0) * tt).round().long()
    r = max(0, int(thickness) // 2)
    col = color.view(3, 1)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            xx = (xs + dx).clamp(0, W - 1)
            yy = (ys + dy).clamp(0, H - 1)
            img[:, yy, xx] = col


_GPU_DISPLAY_DISABLED = False
_viz_stream = None


def _get_viz_stream(device):
    """Dedicated CUDA stream for the light-sync visualisation, isolated from the
    dual-stream manager's seg/opt streams so the viz never shares a stream with
    loop work."""
    global _viz_stream
    if _viz_stream is None:
        _viz_stream = torch.cuda.Stream(device=device)
    return _viz_stream


def _build_publish_mask_bgr(best_pL, best_pR, args):
    """Cheap mask layer for the overlay topic: HWC BGR uint8 on GPU, black
    background with the optimised-pose silhouette in red. No camera image, no
    wireframe/axes, no full composite — just the mask, so the GPU nvJPEG encode
    and the build are both far cheaper than the composite. Layout follows
    --video-layout (overlay_lr -> [L|R] side by side; otherwise left only).
    Used by the opaque (JPEG) ROS publish path."""
    predL = (best_pL[0, ..., 0] > 0.5)
    if getattr(args, 'video_layout', '2x2') == 'overlay_lr' and best_pR is not None:
        predR = (best_pR[0, ..., 0] > 0.5)
        m = torch.cat([predL, predR], dim=1)
    else:
        m = predL
    H, W = m.shape
    bgr = torch.zeros((H, W, 3), dtype=torch.uint8, device=m.device)
    bgr[..., 2] = m.to(torch.uint8) * 255   # red where the mask is (BGR: R=idx2)
    return bgr


def _build_publish_mask_u8(best_pL, best_pR, args):
    """1-channel uint8 mask (255=robot, 0=bg) for the transparent-PNG ROS publish.
    Layout follows --video-layout (overlay_lr -> [L|R] side by side; else left)."""
    predL = (best_pL[0, ..., 0] > 0.5)
    if getattr(args, 'video_layout', '2x2') == 'overlay_lr' and best_pR is not None:
        predR = (best_pR[0, ..., 0] > 0.5)
        m = torch.cat([predL, predR], dim=1)
    else:
        m = predL
    return (m.to(torch.uint8) * 255)


def _emit_overlay_sinks(best_pL, best_pR, args, frame_data):
    """Publish the optimised-pose mask to the ROS overlay topic
    (--publish-overlay): transparent PNG by default, or --overlay-opaque JPEG.
    Cheap mask only (no composite). (The in-headset overlay is handled
    separately by overlay_sender.AsyncMonoStreamer in the run loop.)"""
    _pub = getattr(args, '_overlay_pub', None)
    if _pub is None:
        return
    if getattr(args, 'overlay_opaque', False):
        _pub.publish_jpeg(_build_publish_mask_bgr(best_pL, best_pR, args))
    else:
        _pub.publish_png(_build_publish_mask_u8(best_pL, best_pR, args).cpu().numpy())


def _route_overlay(arr, pose_label, args, win, video_writer):
    """Send the composited overlay to its display sink: GPU OpenGL window
    (--gpu-viz) or host cv2.imshow; plus optional video write. (The ROS publish
    is handled separately by the mask path, not here.) A host copy is made only
    for video or the imshow fallback."""
    if arr is None:
        return
    displayed_gpu = False
    if args.display_progress and getattr(args, 'gpu_viz', False):
        displayed_gpu = _gpu_display(win, arr, pose_label)
    need_cpu = ((args.save_video and video_writer is not None)
                or (args.display_progress and not displayed_gpu))
    if need_cpu:
        frame_display = arr.cpu().numpy() if hasattr(arr, 'cpu') else np.asarray(arr)
        if frame_display.dtype != np.uint8:
            frame_display = frame_display.astype(np.uint8)
        if args.save_video and video_writer is not None:
            tgt = getattr(args, "_video_target_size", None)
            if tgt is None or tgt[0] <= 0 or tgt[1] <= 0:
                video_writer.write(frame_display)
            else:
                tgt_w, tgt_h = tgt
                if (frame_display.shape[1] != tgt_w or frame_display.shape[0] != tgt_h):
                    frame_display = cv2.resize(frame_display, (tgt_w, tgt_h),
                                               interpolation=cv2.INTER_AREA)
                video_writer.write(frame_display)
        if args.display_progress and not displayed_gpu:
            cv2.imshow(win, frame_display)
            cv2.waitKey(1)

def _gpu_display(win, arr, title=None):
    """Show a HWC uint8 CUDA tensor in an OpenGL window with no host copy:
    wrap the tensor's device memory in a cv2.cuda_GpuMat (zero-copy) and
    imshow it (OpenCV uploads GpuMat->GL texture via CUDA-GL interop). Returns
    True on success; on any failure disables itself and returns False so the
    caller falls back to host cv2.imshow. Requires OpenCV built WITH_CUDA +
    WITH_OPENGL and the window created with WINDOW_OPENGL."""
    global _GPU_DISPLAY_DISABLED
    if _GPU_DISPLAY_DISABLED:
        return False
    try:
        a = arr if arr.is_contiguous() else arr.contiguous()
        h, w = int(a.shape[0]), int(a.shape[1])
        gm = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3, a.data_ptr(), int(a.stride(0)))
        if title:
            try:
                cv2.setWindowTitle(win, title)
            except Exception:
                pass
        cv2.imshow(win, gm)
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f"[Viz] GPU display unavailable ({e}); falling back to host "
              "cv2.imshow. For the zero-copy path rebuild OpenCV with "
              "WITH_CUDA + WITH_OPENGL.")
        _GPU_DISPLAY_DISABLED = True
        return False


def create_optimizer_visualization(final_pL, final_pR, probL_t, probR_t, imgL_t, imgR_t, args, frame_i=None,
                                   scene=None, cached_fk_vertices=None, target_9d=None,
                                   H_current=None, H_target=None, K_left=None, pose_label=None,
                                   crop_coords=None, target_mask_cached=None):
    """Create 2x2 visualization: [overlay_L | seg_L] / [overlay_R | seg_R].
    Draws target wireframe, pose axes, and error label on overlay_L.

    If `target_mask_cached` is given it is used directly for the target
    wireframe (rendered earlier on the opt stream by cache_target_overlay), so
    the viz does NO nvdiffrast rasterisation — required under --viz-light-sync to
    keep the seg∥opt overlap. Otherwise the target is rendered in-place from
    scene/cached_fk_vertices/target_9d (full-sync / legacy path).
    """
    if not (args.display_progress or args.save_frames or args.save_video
            or getattr(args, 'publish_overlay', False)
            or getattr(args, 'publish_composite', False)):
        return None
    predL, predR = (final_pL[0, ..., 0] > 0.5), (final_pR[0, ..., 0] > 0.5)
    gtL, gtR = (probL_t[0, 0] > 0.5), (probR_t[0, 0] > 0.5)
    origL, origR = (imgL_t * 255).byte(), (imgR_t * 255).byte()

    # Target wireframe mask: reuse the opt-stream cache if provided, else render.
    target_maskL = None
    if target_mask_cached is not None:
        target_maskL = target_mask_cached
    elif scene is not None and cached_fk_vertices is not None and target_9d is not None:
        with torch.no_grad():
            tgt_pL, _ = render_with_pose(scene, cached_fk_vertices, target_9d)
            target_maskL = (tgt_pL[0, ..., 0] > 0.5)

    ovL = draw_segmentation_masks(origL, predL, alpha=args.alpha, colors=['red'])
    ovR = draw_segmentation_masks(origR, predR, alpha=args.alpha, colors=['red'])

    # Draw all overlays on ovL
    needs_overlay = target_maskL is not None or (H_current is not None and K_left is not None) or pose_label
    if needs_overlay and getattr(args, 'gpu_viz', False):
        # Fully-GPU overlay: no D2H round-trip. Wireframe via Sobel edge,
        # axes/arrow via GPU line raster. The pose label is NOT drawn on the
        # image (no GPU text rasteriser) — the display path sets it as the
        # window title instead. Colours match the CPU path's on-array values so
        # the final BGR swap renders identically.
        dev = ovL.device

        if target_maskL is not None:
            edge = (kornia.filters.sobel(target_maskL.float()[None, None])[0, 0] > 0.1)
            ovL[:, edge] = torch.tensor([0, 255, 0], device=dev, dtype=ovL.dtype).view(3, 1)

        if K_left is not None and H_current is not None and scene is not None:
            ht_optical = scene.cameras['left'].ht_optical.squeeze(0).cpu().numpy()
            fx, fy = float(K_left[0, 0]), float(K_left[1, 1])
            cx, cy = float(K_left[0, 2]), float(K_left[1, 2])
            axis_pts = np.array([[0., 0., 0.], [0.08, 0., 0.],
                                 [0., 0.08, 0.], [0., 0., 0.08]], np.float64)
            # X/Y/Z colours in the same on-array (BGR-convention) order
            # cv2.drawFrameAxes uses, so the final [[2,1,0]] swap matches.
            axis_cols = {1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0]}
            origins = {}
            for H, thick, key in [(H_target, 2, 'tgt'), (H_current, 4, 'cur')]:
                if H is None:
                    continue
                H_b2c = np.linalg.inv(H @ ht_optical)
                Rm, tv = H_b2c[:3, :3], H_b2c[:3, 3]
                pc = (Rm @ axis_pts.T).T + tv                      # (4,3) camera frame
                uv = np.stack([fx * pc[:, 0] / pc[:, 2] + cx,
                               fy * pc[:, 1] / pc[:, 2] + cy], 1)   # (4,2) px
                for ax_i, col in axis_cols.items():
                    _draw_line_gpu(ovL, uv[0], uv[ax_i],
                                   torch.tensor(col, device=dev, dtype=ovL.dtype), thick)
                origins[key] = uv[0]
            if 'cur' in origins and 'tgt' in origins:
                d = float(np.hypot(*(origins['cur'] - origins['tgt'])))
                if d > 3:
                    _draw_line_gpu(ovL, origins['cur'], origins['tgt'],
                                   torch.tensor([0, 255, 255], device=dev, dtype=ovL.dtype), 3)
    elif needs_overlay:
        ovL_np = ovL.permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

        # Green wireframe for target pose
        if target_maskL is not None:
            tgt_np = target_maskL.cpu().numpy().astype(np.uint8) * 255
            contours, _ = cv2.findContours(tgt_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ovL_np, contours, -1, (0, 255, 0), 3)

        # Pose axes at base using ht_optical for correct projection
        if K_left is not None and H_current is not None and scene is not None:
            dist_coeffs = np.zeros(5)
            ht_optical = scene.cameras['left'].ht_optical.squeeze(0).cpu().numpy()
            if hasattr(cv2, 'setLogLevel'): cv2.setLogLevel(0)
            origins = {}
            for H, thickness, key in [(H_target, 2, 'tgt'), (H_current, 4, 'cur')]:
                if H is not None:
                    H_b2c = np.linalg.inv(H @ ht_optical)
                    rvec, _ = cv2.Rodrigues(H_b2c[:3,:3])
                    tvec = H_b2c[:3,3].reshape(3,1)
                    cv2.drawFrameAxes(ovL_np, K_left, dist_coeffs, rvec, tvec, 0.08, thickness)
                    px, _ = cv2.projectPoints(np.array([[0.,0.,0.]]), rvec, tvec, K_left, dist_coeffs)
                    origins[key] = tuple(px.reshape(2).astype(int))
            if hasattr(cv2, 'setLogLevel'): cv2.setLogLevel(3)
            if 'cur' in origins and 'tgt' in origins:
                d = np.sqrt(sum((a-b)**2 for a,b in zip(origins['cur'], origins['tgt'])))
                if d > 3: cv2.arrowedLine(ovL_np, origins['cur'], origins['tgt'], (0,255,255), 3, tipLength=0.3)

        # Error label
        if pose_label:
            (tw, th), _ = cv2.getTextSize(pose_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(ovL_np, (8, 8), (16+tw, 16+th), (0,0,0), -1)
            cv2.putText(ovL_np, pose_label, (12, 12+th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # Adaptive tracking ROI contour — the dilated rendered mask used for motion detection
        # Removed OpenCV approach — using kornia Sobel on msL below

        ovL = torch.from_numpy(ovL_np).permute(2, 0, 1).to(origL.device)

    layout = getattr(args, "video_layout", "2x2")
    overlay_only = (layout == "overlay_lr")

    # Segmentation (ground-truth) panels. Skipped entirely in the overlay-only
    # layout — saves two draw_segmentation_masks calls + the ROI Sobel — unless
    # save_frames needs msL for the per-frame seg dump.
    msL = msR = None
    if not overlay_only or args.save_frames:
        msL = draw_segmentation_masks(origL, gtL, alpha=args.alpha, colors=['blue'])
        msR = draw_segmentation_masks(origR, gtR, alpha=args.alpha, colors=['blue'])

        # Draw ROI contour on segmentation panel (msL) using kornia Sobel edge detection
        if crop_coords is not None and 'roi_mask' in crop_coords:
            y_min_crop, y_max_crop = crop_coords['y_min'], crop_coords['y_max']
            x_min_crop, x_max_crop = crop_coords['x_min'], crop_coords['x_max']
            # Reconstruct full-size ROI mask from cropped roi_mask + coordinates
            roi_mask_full = torch.zeros_like(msL[0], dtype=torch.float32)
            roi_mask_full[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1] = crop_coords['roi_mask']
            # Sobel edge detection to find ROI boundary
            roi_mask_4d = roi_mask_full.unsqueeze(0).unsqueeze(0)
            roi_boundary = kornia.filters.sobel(roi_mask_4d)[0, 0]
            roi_boundary = (roi_boundary > 0.1)
            # Overlay boundary in yellow on the segmentation panel
            yellow = torch.tensor([255, 255, 0], device=msL.device, dtype=msL.dtype).view(3, 1)
            msL[:, roi_boundary] = yellow

    # Compose final frame according to selected layout.
    # 'overlay_lr': 1x2 strip [overlay_L | overlay_R] (rendered overlays only,
    #               no segmentation panels generated).
    # 'left_only':  1x2 strip [overlay_L | seg_L] (no right cam panels).
    # '2x2' (default): both cams + their seg side-by-side and stacked.
    if overlay_only:
        combo = torch.cat([ovL, ovR], dim=2)
    elif layout == "left_only":
        combo = torch.cat([ovL, msL], dim=2)
    else:
        combo = torch.cat([torch.cat([ovL, msL], dim=2),
                           torch.cat([ovR, msR], dim=2)], dim=1)
    arr = combo[[2, 1, 0]].permute(1, 2, 0).contiguous()
    if args.save_frames and frame_i is not None:
        diff_baseL = ((predL.float() - gtL.float()).abs() * 255).to(torch.uint8).unsqueeze(0).repeat(3, 1, 1)
        dfL = generate_wireframe_overlay(predL, gtL, base_image=diff_baseL, thickness=args.wireframe_thickness)
        _save_individual_frames(ovL, dfL, msL, frame_i, args)
    return arr



def _save_individual_frames(overlay_left, diff_left, seg_left, frame_idx, args):
    """Save individual visualization frames."""
    frames_dir = args.save_frames_dir or os.path.join(args.output_dir, "frames")
    for subdir, filename, tensor in [
        ("optimization", f"optimization_left_{frame_idx:06d}.png", overlay_left),
        ("difference_wireframe", f"wireframe_diff_left_{frame_idx:06d}.png", diff_left),
        ("segmentation", f"segmentation_left_{frame_idx:06d}.png", seg_left),
    ]:
        dir_path = os.path.join(frames_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(os.path.join(dir_path, filename), tensor[[2, 1, 0]].permute(1, 2, 0).cpu().numpy())


# === IPCAI POSE OPTIMIZER ===
def optimize_frame(js_buf, prev_9d, scene, tgtL_soft, tgtR_soft, imgL_t, imgR_t,
                   opt, frame_i, args, img_diagonal=None, probL_t=None, probR_t=None):
    """IPCAI (AdamW) pose optimizer with motion-adaptive parameters."""
    cache = _cache
    cache.frame_counter += 1

    if prev_9d.dim() == 2:
        prev_9d = prev_9d.squeeze(0)
    extr_9d = prev_9d.unsqueeze(0).clone().detach().requires_grad_(True)
    mode = REGISTRATION_MODE(args.mode)
    cached_fk_vertices = update_fk_cache(js_buf, scene)

    # Motion detection. The baseline render is the robot at the CURRENT start
    # pose (extr_9d == prev_9d). When FK is unchanged and prev_9d is exactly last
    # frame's converged pose, render(extr_9d) == last frame's best_pL — so reuse
    # that cached render instead of paying a 6th render here. But if the joints
    # changed this frame (cache.fk_changed), last frame's render is at the old
    # articulation and must NOT be reused — render fresh with the new FK.
    # (First frame / after a reset has no cached render, so render once.)
    with torch.no_grad():
        if cache.prev_best_pL is not None and not cache.fk_changed:
            pL_initial = cache.prev_best_pL
        else:
            pL_initial, _ = render_with_pose(scene, cached_fk_vertices, extr_9d)
        render_maskL = (pL_initial[0, ..., 0] > 0.5).float()
    is_large_displacement, is_rotation, iou = detect_motion(probL_t, render_maskL, imgL_t, img_diagonal)
    # Save crop_coords from the cache for ROI visualization
    crop_coords = _cache.prev_crop_coords

    # Adaptive parameters
    target_lr, target_betas, clip_threshold, target_wd = compute_adaptive_params(
        args.ipcai_lr, args.weight_decay, is_large_displacement, is_rotation, iou)
    smooth_adaptive_params(target_lr, target_betas, target_wd)

    # Create/update AdamW optimizer
    if opt is None:
        opt = torch.optim.AdamW([extr_9d], lr=cache.smooth_lr, betas=cache.smooth_betas,
                                weight_decay=cache.smooth_weight_decay)
    else:
        old_params = opt.param_groups[0]['params']
        if old_params and old_params[0] in opt.state:
            opt.state[extr_9d] = opt.state.pop(old_params[0])
        opt.param_groups[0].update({
            'params': [extr_9d], 'lr': cache.smooth_lr,
            'betas': cache.smooth_betas, 'weight_decay': cache.smooth_weight_decay})

    # Optimization loop
    best_p = torch.empty_like(extr_9d)
    best_pL = torch.empty_like(tgtL_soft)
    best_pR = torch.empty_like(tgtR_soft)
    # best_loss tracked on GPU as a 0-d tensor — avoids a host-side sync
    # per inner iteration. Was previously `best_loss_value = float('inf')`
    # with `if loss.item() < best_loss_value: ...`, which forced a
    # `cudaStreamSynchronize` 2× per iter (5 iters × 2 = 10 syncs/frame).
    # The GPU `torch.where` selection below preserves identical semantics
    # (snapshot whenever loss decreases) without leaving GPU.
    best_loss = torch.full((), float('inf'), device=extr_9d.device,
                           dtype=tgtL_soft.dtype)

    iter_start_time = time.perf_counter()
    for it in range(1, args.max_iterations + 1):
        opt.zero_grad(set_to_none=True)
        pL, pR = render_with_pose(scene, cached_fk_vertices, extr_9d)
        loss = compute_registration_loss(pL, pR, tgtL_soft, tgtR_soft, mode, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([extr_9d], max_norm=clip_threshold)
        opt.step()
        # GPU-side selection: if loss < best_loss, snapshot params + renders
        # into the best_* buffers. `torch.where` broadcasts the 0-d bool
        # against the param/render shapes. Allocator-cached, no host sync.
        improved = loss < best_loss
        best_loss = torch.where(improved, loss.detach(), best_loss)
        best_p.copy_(torch.where(improved, extr_9d.data, best_p), non_blocking=True)
        best_pL.copy_(torch.where(improved, pL.data, best_pL), non_blocking=True)
        best_pR.copy_(torch.where(improved, pR.data, best_pR), non_blocking=True)

    iter_rate = it / max(time.perf_counter() - iter_start_time, 1e-9)
    # Cache the converged left render as next frame's motion-detection baseline
    # (next frame starts at this pose with the same FK, so this IS its pL_initial
    # — saves a render). Keep a detached reference; best_pL is a fresh buffer
    # each frame so this won't be overwritten under us.
    cache.prev_best_pL = best_pL.detach()
    # best_pL/best_pR are computed every frame regardless; always return them
    # so the overlay can REUSE the optimiser's render (no extra native render).
    # crop_coords stays gated on the viz flags (only it needs them).
    _cc = crop_coords if (args.display_progress or args.save_frames or args.save_video) else None
    return best_p, opt, best_pL, best_pR, iter_rate, _cc


def _detect_bag_storage_id(bag_path, override=None):
    """Detect rosbag2 storage backend ('mcap' or 'sqlite3') from the bag dir."""
    if override:
        return override
    meta = os.path.join(bag_path, 'metadata.yaml')
    if os.path.exists(meta):
        try:
            with open(meta, 'r') as f:
                data = yaml.safe_load(f)
            sid = data.get('rosbag2_bagfile_information', {}).get('storage_identifier')
            if sid:
                return sid
        except Exception:
            pass
    try:
        for fname in os.listdir(bag_path):
            if fname.endswith('.mcap'): return 'mcap'
            if fname.endswith('.db3'): return 'sqlite3'
    except OSError:
        pass
    return 'sqlite3'


def _decode_image_msg_to_numpy(msg):
    """Convert sensor_msgs/Image OR sensor_msgs/CompressedImage to a
    (H, W, 3) uint8 RGB numpy array.

    For CompressedImage (JPEG), uses cv2.imdecode on the host. Most
    callers should prefer `_decode_image_msg_to_gpu_chw` to avoid the
    host decode entirely when the GPU is available.

    For raw Image, uses numpy slicing for the channel reorder (BGR
    <-> RGB and alpha-strip) instead of cv2.cvtColor — saves ~10 ms/
    frame at 1080p stereo (measured in bag mode). `np.ascontiguousarray
    (buf[..., ::-1])` is a strided reverse-and-copy, near-bandwidth-
    bound on CPU.
    """
    # CompressedImage has `.format` (e.g. "jpeg") + `.data` (compressed bytes).
    # Raw Image has `.height` / `.width` / `.encoding` / `.data`.
    if hasattr(msg, 'format'):
        # Compressed images route through GPU decode (nvJPEG for JPEG,
        # PyAV NVDEC for H.264) then bounce back to numpy. This trades
        # one D2H copy for skipping the much slower cv2.imdecode +
        # cvtColor host path. On CUDA-less systems we fall back to
        # cv2 for JPEG only.
        if torch.cuda.is_available():
            chw_cuda = _decode_image_msg_to_gpu_chw(msg, device='cuda')
            return (chw_cuda.permute(1, 2, 0)
                              .contiguous()
                              .cpu()
                              .numpy())
        fmt = (msg.format or '').lower()
        if fmt.startswith('jpeg') or fmt == '':
            np_buf = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(
                    f"cv2.imdecode failed on CompressedImage "
                    f"(format='{msg.format}', {len(msg.data)} bytes)")
            return np.ascontiguousarray(bgr[..., ::-1])    # BGR → RGB
        raise RuntimeError(
            f"CompressedImage format '{msg.format}' on CPU-only host "
            f"is unsupported. Need CUDA for H.264 decode.")

    H, W = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(H, W, -1)
    if enc in ('bgr8', 'bgra8'):
        # BGR(A) -> RGB: reverse channel axis on the first 3, drop alpha.
        return np.ascontiguousarray(buf[..., 2::-1])
    if enc in ('rgb8', 'rgba8'):
        return np.ascontiguousarray(buf[:, :, :3])
    if enc == 'mono8':
        # mono -> RGB: tile single channel to 3.
        return np.broadcast_to(buf[..., :1], (H, W, 3)).copy()
    # Unknown — assume BGR-like.
    return np.ascontiguousarray(buf[..., 2::-1])


def _decode_image_msg_to_gpu_chw(msg, device='cuda'):
    """Decode a ROS image message directly to a (3, H, W) uint8 CUDA
    tensor.

    For sensor_msgs/CompressedImage:
      - JPEG (format starts with "jpeg") goes through nvJPEG via
        `torchvision.io.decode_jpeg(device='cuda')`. Only the
        compressed bytes (~30 KB/eye at q95) cross PCIe — the full-
        pixel upload (~6 MB/eye at 1080p) and the ~10 ms host decode
        are both elided. Net cost per eye: ~2-3 ms vs ~12 ms on CPU.
      - H.264 (format starts with "h264" or "h.264") goes through
        PyAV's h264_cuvid (NVDEC) when available, falling back to
        CPU h264. Decoder state is cached per `msg.header.frame_id`
        because H.264 has inter-frame dependencies — the SAME
        decoder instance must see every packet of a stream in order.

    For sensor_msgs/Image, falls back to the CPU decode in
    `_decode_image_msg_to_numpy` then uploads + permutes to CHW.

    Returns: (3, H, W) torch.uint8 tensor on `device`.

    Lifetime note (CompressedImage JPEG path):
      `bytearray(msg.data)` makes a Python-owned copy of the JPEG
      bytes. nvJPEG's H2D copy is asynchronous — referencing
      msg.data directly via torch.frombuffer can race with the
      ROS msg going out of scope before the copy actually runs.

      No host-side sync is needed here. By the time the dual-stream
      pipeline's seg/opt kernels actually dispatch on their streams,
      the default-stream decode kernel has long since retired (the
      kernel launch sequence is decode → .float() → .div_() → host
      returns → minimum ~50 µs of host code → seg block dispatches).
      In practice the GPU is well past the decode by the time any
      consumer reads. If a race ever does surface, the right fix is
      `default_stream.synchronize()` after the prep — a host-side
      drain of the decode stream that leaves stream_seg / stream_opt
      / stream_display untouched.

    Inter-frame dep note (H.264 path):
      PyAV CodecContext.decode() returns a list of frames per
      packet. For low-latency streaming (no B-frames) this is usually
      one frame per packet after warmup. The very first packet for a
      stream — the IDR with prepended SPS+PPS — typically returns one
      frame. If 0 frames come back, the stream is malformed (e.g.
      bag truncated mid-GOP, first frame is a P-frame with no
      preceding IDR) — we raise so the caller can decide whether to
      skip the frame.
    """
    if hasattr(msg, 'format'):
        fmt = (msg.format or '').lower()
        # JPEG path — nvJPEG, unchanged.
        if fmt.startswith('jpeg') or fmt == '':
            jpeg_bytes_t = torch.frombuffer(bytearray(msg.data),
                                            dtype=torch.uint8)
            from torchvision.io import decode_jpeg, ImageReadMode
            return decode_jpeg(jpeg_bytes_t,
                               mode=ImageReadMode.RGB,
                               device=device)
        # H.264 path — PyAV NVDEC.
        if fmt.startswith('h264') or fmt.startswith('h.264'):
            av = _ensure_av()
            frame_id = getattr(msg.header, 'frame_id', '') \
                       if hasattr(msg, 'header') else ''
            decoder = _get_h264_decoder(frame_id)
            packet = av.Packet(bytes(msg.data))
            frames = decoder.decode(packet)
            if not frames:
                # Decoder buffered the packet — typical at the very
                # start of a bag captured mid-GOP (first packet is a
                # P-frame, no IDR/SPS/PPS yet). Signal the caller so
                # the whole triplet can be skipped; the decoder retains
                # the packet's state so once an IDR arrives it'll
                # produce frames normally.
                raise H264NeedsMorePackets(
                    f"H.264 decoder for frame_id '{frame_id}' has not "
                    f"produced a frame yet ({len(msg.data)}-byte "
                    f"packet buffered). Waiting for IDR.")
            frame = frames[0]
            # PyAV uses libswscale to convert YUV → RGB on the host.
            # NVDEC decoded on the GPU but the output is bounced back
            # for the colorspace convert (no zero-copy GPU→GPU in the
            # public PyAV API). Net cost: ~3-4 ms/eye at 1080p — still
            # well below the cv2 + cudaMemcpy path.
            rgb_hwc = frame.to_ndarray(format='rgb24')
            return (torch.from_numpy(rgb_hwc)
                          .to(device, non_blocking=True)
                          .permute(2, 0, 1)
                          .contiguous())
        raise RuntimeError(
            f"Unsupported CompressedImage format: '{msg.format}'. "
            f"Expected 'jpeg' or 'h264'.")

    # Raw Image fallback (unchanged) — CPU decode then upload.
    np_rgb = _decode_image_msg_to_numpy(msg)
    return (torch.from_numpy(np_rgb)
                .to(device, non_blocking=False)
                .permute(2, 0, 1)
                .contiguous())


def _stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class BagStreamReader:
    """Sequentially streams synchronised (left, right, joint_state) triplets from
    a rosbag2 directory. Holds a `SequentialReader` open and pumps messages on
    demand via next_triplet(). No upfront scan, no full-bag RAM cost — only the
    pending per-topic deques (small, since the matcher drains them quickly).

    `total_frames` is the per-image-topic message count read from metadata.yaml,
    which is the upper bound of synchronisable triplets. Actual yielded count
    may be smaller if some frames fail the max_dt synchronisation.
    """
    def __init__(self, bag_path, left_topic, right_topic, js_topic,
                 max_dt_ms=50, storage_id=None):
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        self._deserialize = deserialize_message

        self.left_topic, self.right_topic, self.js_topic = left_topic, right_topic, js_topic
        self.max_dt_ns = int(max_dt_ms * 1e6)

        sid = _detect_bag_storage_id(bag_path, override=storage_id)
        print(f"[Bag] Streaming from {bag_path} (storage='{sid}')")
        print(f"      left='{left_topic}', right='{right_topic}', js='{js_topic}', max_dt={max_dt_ms}ms")

        self.reader = SequentialReader()
        self.reader.open(StorageOptions(uri=bag_path, storage_id=sid),
                         ConverterOptions(input_serialization_format='cdr',
                                          output_serialization_format='cdr'))
        type_map = {t.name: t.type for t in self.reader.get_all_topics_and_types()}
        for topic in (left_topic, right_topic, js_topic):
            if topic not in type_map:
                raise RuntimeError(f"[Bag] Topic '{topic}' not in bag. "
                                   f"Available: {list(type_map.keys())}")
        self.msg_cls = {t: get_message(type_map[t]) for t in
                        (left_topic, right_topic, js_topic)}
        self.pending = {left_topic: deque(), right_topic: deque(), js_topic: deque()}

        # Total frames from metadata.yaml — count of left-image messages.
        self.total_frames = self._read_left_count(bag_path, left_topic)
        print(f"[Bag] {self.total_frames} '{left_topic}' messages in metadata")

    @staticmethod
    def _read_left_count(bag_path, left_topic):
        meta_path = os.path.join(bag_path, 'metadata.yaml')
        try:
            with open(meta_path, 'r') as f:
                meta = yaml.safe_load(f)
            for entry in meta.get('rosbag2_bagfile_information', {}).get('topics_with_message_count', []):
                if entry.get('topic_metadata', {}).get('name') == left_topic:
                    return int(entry.get('message_count', 0))
        except Exception:
            pass
        return 0

    def _try_emit(self):
        L, R, J = self.left_topic, self.right_topic, self.js_topic
        while self.pending[L] and self.pending[R] and self.pending[J]:
            tL, _ = self.pending[L][0]
            tR, _ = self.pending[R][0]
            tJ, _ = self.pending[J][0]
            t_min, t_max = min(tL, tR, tJ), max(tL, tR, tJ)
            if t_max - t_min <= self.max_dt_ns:
                _, mL = self.pending[L].popleft()
                _, mR = self.pending[R].popleft()
                _, mJ = self.pending[J].popleft()
                return mL, mR, mJ
            oldest = min((L, R, J), key=lambda t: self.pending[t][0][0])
            self.pending[oldest].popleft()
        return None

    def next_triplet(self):
        """Pump bag messages until one synchronised triplet is available, or
        the bag is exhausted. Returns (left_msg, right_msg, js_msg) or None.
        """
        while True:
            t = self._try_emit()
            if t is not None:
                return t
            if not self.reader.has_next():
                return None
            topic, raw, t_ns = self.reader.read_next()
            if topic not in self.pending:
                continue
            msg = self._deserialize(raw, self.msg_cls[topic])
            stamp_ns = _stamp_to_ns(msg.header.stamp) if hasattr(msg, 'header') else int(t_ns)
            self.pending[topic].append((stamp_ns, msg))


class BagStreamData:
    """Wraps BagStreamReader with a dict-like API used by run_pipeline:
      - data['source'] == 'bag-stream'
      - len(data['left_images']) returns total_frames (via _BagLenShim)
      - the next_triplet() method is the actual data accessor
    """
    def __init__(self, reader: 'BagStreamReader'):
        self.reader = reader
        self.source = 'bag-stream'
        self.frames_read = 0

    def __getitem__(self, key):
        if key == 'source':
            return self.source
        # 'left_images' is only used for len(); we shim that via a tiny proxy.
        if key in ('left_images', 'right_images', 'joint_states'):
            return _BagLenShim(self.reader.total_frames)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _BagLenShim:
    """A list-shaped object that only supports len(); used so existing
    `len(data['left_images'])` calls return the bag's total frame count."""
    __slots__ = ('_n',)
    def __init__(self, n): self._n = n
    def __len__(self): return self._n


def load_bag_data(bag_path, left_topic, right_topic, js_topic, max_dt_ms=50, storage_id=None):
    """Open the bag and return a streaming wrapper. No preload, no full-bag scan."""
    reader = BagStreamReader(bag_path, left_topic, right_topic, js_topic,
                             max_dt_ms=max_dt_ms, storage_id=storage_id)
    return BagStreamData(reader)


# === PER-FRAME PROCESSING ===
def process_one_frame_dual_stream(
        frame_data, next_frame_data, prefetched_seg,
        scene, args, prev_9d, opt, video_writer, per_dir, img_diagonal,
        segmentation_events, optimization_events, dt_events, iteration_rates,
        win, H_final_gt=None, target_9d=None, K_left=None,
        H_base_target_world=None, camera_trajectory=None,
        prev_opt_end_event=None, read_next_cb=None):
    """Run segmentation + optimization for ONE frame using the dual-stream
    architecture. This is the shared per-frame core used by both the batched
    and the streaming run paths — they only differ in how `frame_data` and
    `next_frame_data` are sourced (array slice vs streaming reader).

    Inputs:
      frame_data:        the frame to seg+opt now (left_img/right_img/joint_state/frame_idx)
      next_frame_data:   the frame whose seg should be prefetched on next_stream
                         (None if no further frame — disables prefetch)
      prefetched_seg:    (prob_batch, seg_start, seg_end) tuple if a previous
                         iteration prefetched THIS frame's seg on what is now
                         current_stream; None otherwise (will run seg fresh).
      prev_opt_end_event: the previous frame's `opt_end_event` for pipeline
                         interval timing (None for the first frame). Pair-wise
                         `elapsed_time` between these events gives true
                         frame-to-frame GPU pacing — the same pattern as
                         `process_batch_dual_stream` in the original
                         `stereo_ipcai_pipeline.py`.

    Returns:
      (prev_9d, opt, opt_end_event, pipeline_interval, frame_error,
       next_prefetched_seg)
        - opt_end_event: pass back as `prev_opt_end_event` next call.
        - next_prefetched_seg: the (prob_batch, seg_start, seg_end) for
          next_frame_data, ready to be passed in as `prefetched_seg` next call
          AFTER the streams have been swapped.
    """
    global dual_stream_mgr, device

    frame_i = frame_data['frame_idx']
    left_img_t = frame_data['left_img']
    right_img_t = frame_data['right_img']
    js_buf = frame_data['joint_state']
    current_stream = dual_stream_mgr.get_current_stream()
    next_stream = dual_stream_mgr.get_next_stream()

    seg_start_event = torch.cuda.Event(enable_timing=True)
    seg_end_event = torch.cuda.Event(enable_timing=True)
    opt_start_event = torch.cuda.Event(enable_timing=True)
    opt_end_event = torch.cuda.Event(enable_timing=True)

    no_pipeline = args.no_pipeline

    # Segmentation: prefetched-from-previous-iter > fresh.
    if prefetched_seg is not None:
        prob_batch, seg_start_event, seg_end_event = prefetched_seg
        # `prob_batch` was produced on next_stream by the previous
        # iteration's prefetch. Host-sync next_stream here so the
        # data is fully settled before any subsequent reader touches
        # it. This matches the reference `stereo_ipcai_pipeline.py`
        # pattern. We tried `current_stream.wait_stream(next_stream)`
        # earlier (cheaper, GPU-side fence) but that only orders
        # current_stream's reads — the default stream (where
        # `create_optimizer_visualization` runs) doesn't know
        # next_stream is still working, races with it via
        # `probL_t` / `probR_t` slices, and the allocator can reuse
        # memory under the still-running prefetch. A host sync of
        # next_stream is the simple, robust fix.
        next_stream.synchronize()
    else:
        with torch.cuda.stream(current_stream):
            seg_start_event.record(current_stream)
            prob_batch = extract_mask(
                torch.stack([left_img_t, right_img_t], dim=0), current_stream)
            seg_end_event.record(current_stream)
        # `extract_mask` writes the global `t_small_static` and reads
        # the global `prob_small_static` via `seg_graph.replay()`. The
        # prefetch we're about to launch on `next_stream` touches the
        # SAME static buffers. Without this barrier, the two replays
        # can run concurrently on different streams and race on the
        # shared buffers — the prefetch's `t_small_static.copy_(tmp)`
        # can land mid-way through this seg's `seg_graph.replay()`
        # read, producing the "occasional mask flushing" symptom.
        # This branch only fires on iteration 1 (no prefetched seg
        # available yet) — subsequent iterations skip the fresh-seg
        # path entirely, so this host wait is paid exactly once per
        # pipeline run.
        current_stream.synchronize()

    # Pre-launch next frame's segmentation on next_stream — overlapped with
    # this frame's optimisation.
    next_prefetched_seg = None
    if not no_pipeline and next_frame_data is not None:
        ns_start = torch.cuda.Event(enable_timing=True)
        ns_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(next_stream):
            ns_start.record(next_stream)
            n_prob = extract_mask(
                torch.stack([next_frame_data['left_img'], next_frame_data['right_img']], dim=0),
                next_stream)
            ns_end.record(next_stream)
        next_prefetched_seg = (n_prob, ns_start, ns_end)

    dual_stream_mgr.synchronize_current()
    if no_pipeline:
        torch.cuda.synchronize()
    segmentation_events.append((seg_start_event, seg_end_event))

    # Optimization
    with torch.cuda.stream(current_stream):
        prob_batch_permuted = prob_batch.permute(0, 2, 3, 1).contiguous()
        probL_t, probR_t = prob_batch[0:1], prob_batch[1:2]
        tgtL_soft, tgtR_soft = prob_batch_permuted[0:1], prob_batch_permuted[1:2]

        if args.use_gaussian_blur == "true":
            dt_start = torch.cuda.Event(enable_timing=True)
            dt_end = torch.cuda.Event(enable_timing=True)
            dt_start.record(current_stream)
            tgtL_soft = gpu_gaussian_blur(tgtL_soft, kernel_size=args.gaussian_kernel_size,
                                          sigma=args.gaussian_sigma)
            tgtR_soft = gpu_gaussian_blur(tgtR_soft, kernel_size=args.gaussian_kernel_size,
                                          sigma=args.gaussian_sigma)
            dt_end.record(current_stream)
            dt_events.append((dt_start, dt_end))

        opt_start_event.record(current_stream)
        # Per-frame target update for moving camera
        if H_base_target_world is not None and camera_trajectory is not None:
            if frame_i < len(camera_trajectory):
                H_cam_world_t = camera_trajectory[frame_i]
                H_cam2base_target_t = np.linalg.inv(H_base_target_world) @ H_cam_world_t
                target_9d = pk.matrix44_to_se3_9d(
                    torch.from_numpy(np.linalg.inv(H_cam2base_target_t)).float().to(device).unsqueeze(0))
        result = optimize_frame(js_buf, prev_9d, scene, tgtL_soft, tgtR_soft,
                                left_img_t, right_img_t, opt, frame_i, args,
                                img_diagonal=img_diagonal, probL_t=probL_t, probR_t=probR_t)
        # Render the target-pose overlay mask here (opt stream, inside the
        # opt_end_event bracket) so it overlaps the seg prefetch instead of
        # fencing the device from inside the viz. The light-sync viz reuses it.
        cache_target_overlay(args, scene, js_buf, target_9d)
        opt_end_event.record(current_stream)

    # The optimiser kernels for THIS frame are now queued (async) on stream_opt.
    # Before we host-sync on them, issue the NEXT frame's read+convert: it runs
    # on the host + default stream while the GPU is busy optimising, so the
    # ingest (copy-wait + convert) hides under the opt instead of running
    # serially after the sync. Streaming paths pass this; the batched path
    # leaves it None (its "read" is a free array slice). The GPU op/stream
    # assignments are unchanged — only the wall-time of the host read moves.
    if read_next_cb is not None:
        read_next_cb()

    dual_stream_mgr.synchronize_current()
    if no_pipeline:
        torch.cuda.synchronize()
    optimization_events.append((opt_start_event, opt_end_event))

    # Unpack result
    best_p, opt, best_pL, best_pR, iter_rate, crop_coords = result
    # Incoming (last-good) pose, captured BEFORE we advance prev_9d, so the
    # safe-inversion guard below can fall back to it on a degenerate frame.
    prev_9d_in = prev_9d.squeeze(0).detach().clone() if prev_9d.dim() > 1 else prev_9d.detach().clone()
    new_prev_9d = best_p.squeeze(0).detach().clone() if best_p.dim() > 1 else best_p.detach().clone()
    iteration_rates.append(iter_rate)

    # Save extrinsics.
    # Safe-inversion guard: the optimiser can occasionally emit a non-finite /
    # degenerate SE(3) (e.g. a cold first-frame FK or a NaN gradient), which
    # makes se3->4x4 singular and crashes torch.linalg.inv. When that happens,
    # hold the previous good pose for this frame and DON'T advance prev_9d to
    # the bad result, instead of taking down the whole pipeline.
    best_p_4x4 = pk.se3_9d_to_matrix44(
        best_p.unsqueeze(0) if best_p.dim() == 1 else best_p)[0]
    if torch.isfinite(best_p_4x4).all():
        try:
            H_l2b_t = torch.linalg.inv(best_p_4x4)
            prev_9d = new_prev_9d
        except torch._C._LinAlgError:
            print(f"[Frame {frame_i}] WARNING: singular optimiser pose; "
                  f"holding previous frame's pose (skipping update)")
            H_l2b_t = torch.linalg.inv(
                pk.se3_9d_to_matrix44(prev_9d_in.unsqueeze(0))[0])
            prev_9d = prev_9d_in
    else:
        print(f"[Frame {frame_i}] WARNING: non-finite optimiser pose; "
              f"holding previous frame's pose (skipping update)")
        H_l2b_t = torch.linalg.inv(
            pk.se3_9d_to_matrix44(prev_9d_in.unsqueeze(0))[0])
        prev_9d = prev_9d_in
    H_r2b_t = H_l2b_t @ scene.cameras["right"].extrinsics.to(device)

    # `H_l2b_np` is needed by three downstream consumers, all optional:
    #   - per-frame .npy save  (--save-per-frame-npy)
    #   - per-frame error      (when GT is provided)
    #   - visualization        (--display-progress or --save-video)
    # The `.cpu().detach().numpy()` call forces a GPU→CPU sync per frame —
    # we want to skip that work entirely when nothing actually consumes it.
    # Compute upfront whether any consumer is active; resolve `H_target_for_error`
    # too so the error block below can short-circuit. Both are pure host-side
    # checks, no GPU work yet.
    H_target_for_error = None
    if (H_base_target_world is not None and camera_trajectory is not None
            and frame_i < len(camera_trajectory)):
        H_cam_world_t = camera_trajectory[frame_i]
        H_target_for_error = np.linalg.inv(H_base_target_world) @ H_cam_world_t
    elif H_final_gt is not None:
        H_target_for_error = H_final_gt

    save_npy = getattr(args, 'save_per_frame_npy', False)
    need_error = H_target_for_error is not None
    need_viz = (args.display_progress or args.save_video) and best_pL is not None
    H_l2b_np = (H_l2b_t.cpu().detach().numpy()
                if (save_npy or need_error or need_viz) else None)

    if save_npy:
        np.save(os.path.join(per_dir, f"camera_to_base_left_{frame_i}.npy"), H_l2b_np)
        np.save(os.path.join(per_dir, f"camera_to_base_right_{frame_i}.npy"),
                H_r2b_t.cpu().detach().numpy())

    # Per-frame error against target
    t_err = r_err = 0.0
    frame_error = None
    if need_error:
        t_err = np.linalg.norm(H_l2b_np[:3, 3] - H_target_for_error[:3, 3]) * 1000.0
        R_err_mat = H_target_for_error[:3, :3].T @ H_l2b_np[:3, :3]
        r_err = np.degrees(np.arccos(np.clip((np.trace(R_err_mat) - 1) / 2, -1, 1)))
        frame_error = {'frame': frame_i, 'trans_err_mm': t_err, 'rot_err_deg': r_err}
        print(f"    Frame {frame_i}: trans={t_err:.2f} mm, rot={r_err:.3f}°")

    # Pipeline interval: pair last call's `opt_end_event` with this call's.
    # Note: these events live on different streams (current/next alternate
    # via swap_streams), but CUDA records absolute GPU timestamps when
    # `record()` is called, so `elapsed_time` returns a meaningful wall-time
    # interval once both events have completed. Same pattern as the original
    # `process_batch_dual_stream` in `stereo_ipcai_pipeline.py`.
    pipeline_interval = ((prev_opt_end_event, opt_end_event)
                         if prev_opt_end_event is not None else None)

    # Publish (cheap mask) and/or display (full composite).
    #   * --publish-overlay: emit ONLY the optimised-pose mask to the ROS topic
    #     (transparent PNG / opaque JPEG). No composite — cheap, doesn't stall
    #     opt. (The in-headset overlay is sent in the run loop, not here.)
    #   * --publish-composite: publish the FULL composite (camera image + red
    #     mask + green target wireframe + axes) as JPEG to the ROS topic instead
    #     of the bare mask. Heavier (builds + GPU-JPEG-encodes a full frame), so
    #     it can stall opt more than the mask path.
    #   * --display-progress / --save-video: build the full composite for the
    #     window/video.
    _pub = getattr(args, '_overlay_pub', None)
    _pub_composite = bool(getattr(args, 'publish_composite', False)) and (_pub is not None)
    # Mask sink only in mask mode; in composite mode we publish the composite.
    _need_sinks = (_pub is not None) and not _pub_composite
    _need_composite = (args.display_progress or args.save_video or _pub_composite)
    if best_pL is not None and (_need_sinks or _need_composite):
        light = (getattr(args, 'viz_light_sync', False) and opt_end_event is not None)
        pose_label = (f"t={t_err:.1f}mm r={r_err:.2f}deg"
                      if H_target_for_error is not None else None)
        if light:
            # Lowest-latency path. Run everything on a dedicated stream that waits
            # only on THIS frame's opt, on PRIVATE CLONES of the inputs:
            #   * viz_stream.wait_event(opt_end_event) — inputs (and the target
            #     mask cached on the opt stream) are valid before we read them.
            #   * clone() on viz_stream + record_stream(originals) — the next
            #     frame may freely reuse/overwrite the originals; the viz only
            #     touches its clones (allocated/freed on viz_stream), so the
            #     allocator serialises reuse correctly. Removes the flicker.
            #   * NO nvdiffrast in the viz — the green target wireframe reuses
            #     `_overlay_target_maskL`, rendered earlier on the opt stream.
            srcs = (best_pL, best_pR, probL_t, probR_t, left_img_t, right_img_t,
                    _overlay_target_maskL)
            viz_stream = _get_viz_stream(best_pL.device)
            viz_stream.wait_event(opt_end_event)
            with torch.cuda.stream(viz_stream):
                s = [t.clone() if (t is not None and t.is_cuda) else t for t in srcs]
                for _t in srcs:
                    try:
                        if _t is not None and _t.is_cuda:
                            _t.record_stream(viz_stream)
                    except Exception:
                        pass
                if _need_sinks:
                    _emit_overlay_sinks(s[0], s[1], args, frame_data)
                if _need_composite:
                    arr = create_optimizer_visualization(
                        s[0], s[1], s[2], s[3], s[4], s[5], args, frame_i,
                        scene=scene, cached_fk_vertices=None, target_9d=None,
                        H_current=H_l2b_np, H_target=H_target_for_error, K_left=K_left,
                        pose_label=pose_label, crop_coords=crop_coords,
                        target_mask_cached=s[6])
                    if _pub_composite:
                        _pub.publish_jpeg(arr)
                    if args.display_progress or args.save_video:
                        _route_overlay(arr, pose_label, args, win, video_writer)
        else:
            # Robust full sync: all pending GPU work (incl. the prefetch) is done
            # before the viz reads. Reuses the opt-stream target mask cache too.
            torch.cuda.synchronize()
            if _need_sinks:
                _emit_overlay_sinks(best_pL, best_pR, args, frame_data)
            if _need_composite:
                arr = create_optimizer_visualization(
                    best_pL, best_pR, probL_t, probR_t, left_img_t, right_img_t, args, frame_i,
                    scene=scene, cached_fk_vertices=None, target_9d=None,
                    H_current=H_l2b_np, H_target=H_target_for_error, K_left=K_left,
                    pose_label=pose_label, crop_coords=crop_coords,
                    target_mask_cached=_overlay_target_maskL)
                if _pub_composite:
                    _pub.publish_jpeg(arr)
                if args.display_progress or args.save_video:
                    _route_overlay(arr, pose_label, args, win, video_writer)

    dual_stream_mgr.swap_streams()
    # Stash the optimiser's rendered masks so the in-headset overlay REUSES them
    # (no extra render) — the current best-pose silhouette, sent with minimal
    # latency by overlay_sender.AsyncMonoStreamer (dedicated stream + worker).
    global _overlay_pL, _overlay_pR
    _overlay_pL = best_pL
    _overlay_pR = best_pR
    return (prev_9d, opt, opt_end_event, pipeline_interval, frame_error,
            next_prefetched_seg)



# === HELPERS ===
def _compute_event_stats(events):
    """Reduce a list of (start_event, end_event) pairs to (mean_s, std_s, fps, times_s)."""
    times = [s.elapsed_time(e) / 1000.0 for s, e in events]
    if not times:
        return 0, 0, 0, times
    mean = np.mean(times)
    return mean, np.std(times), (1.0 / mean if mean > 0 else 0), times