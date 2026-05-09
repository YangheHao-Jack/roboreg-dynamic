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
    _decode_image_msg_to_numpy        sensor_msgs/Image -> RGB numpy
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
    """Extract segmentation mask at 576x1024, upsample to original size."""
    global t_small_static, prob_small_static, seg_graph, args
    with torch.cuda.stream(stream):
        tmp = F.interpolate(img_batch, (576, 960), mode='bicubic', align_corners=False, antialias=True)
        t_small_static.copy_(tmp)
        seg_graph.replay()
        prob_out = F.interpolate(prob_small_static, size=img_batch.shape[2:], mode='bilinear', align_corners=False, antialias=True)
        result = prob_out.clone()
        result[prob_out > args.pth] = 1.0
        return result


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
        self.prev_probL_cropped = None
        self.prev_crop_coords = None
        self.prev_moments = None
        self.frame_counter = 0
        self.smooth_lr = None
        self.smooth_betas = (0.9, 0.999)
        self.smooth_weight_decay = None

    def reset(self, base_lr, weight_decay):
        self.cached_fk_vertices = None
        self.cached_js = None
        self.prev_probL_cropped = None
        self.prev_crop_coords = None
        self.prev_moments = None
        self.frame_counter = 0
        self.smooth_lr = base_lr
        self.smooth_betas = (0.9, 0.999)
        self.smooth_weight_decay = weight_decay


_cache = OptimizerCache()


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

    t_small_static = torch.empty((2, 3, 576, 960), device=device_obj, dtype=torch.float32)
    logits_static = torch.empty((2, 1, 576, 960), device=device_obj, dtype=torch.float32)
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
    """Configure robot and return FK vertices.
    
    configure() must run every frame to restore robot internal state
    after render_with_pose overwrites _configured_vertices.
    But we only clone vertices once since joints are constant.
    """
    with torch.no_grad():
        identity = torch.eye(4, device=js_buf.device, dtype=torch.float32).unsqueeze(0)
        scene.robot.configure(js_buf, identity)
        if _cache.cached_fk_vertices is None:
            _cache.cached_fk_vertices = scene.robot._configured_vertices.clone()
            _cache.cached_js = js_buf.clone()
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


def create_optimizer_visualization(final_pL, final_pR, probL_t, probR_t, imgL_t, imgR_t, args, frame_i=None,
                                   scene=None, cached_fk_vertices=None, target_9d=None,
                                   H_current=None, H_target=None, K_left=None, pose_label=None,
                                   crop_coords=None):
    """Create 2x2 visualization: [overlay_L | seg_L] / [overlay_R | seg_R].
    Draws target wireframe, pose axes, and error label on overlay_L.
    """
    if not (args.display_progress or args.save_frames or args.save_video):
        return None
    predL, predR = (final_pL[0, ..., 0] > 0.5), (final_pR[0, ..., 0] > 0.5)
    gtL, gtR = (probL_t[0, 0] > 0.5), (probR_t[0, 0] > 0.5)
    origL, origR = (imgL_t * 255).byte(), (imgR_t * 255).byte()

    # Render target wireframe
    target_maskL = None
    if scene is not None and cached_fk_vertices is not None and target_9d is not None:
        with torch.no_grad():
            tgt_pL, _ = render_with_pose(scene, cached_fk_vertices, target_9d)
            target_maskL = (tgt_pL[0, ..., 0] > 0.5)

    ovL = draw_segmentation_masks(origL, predL, alpha=args.alpha, colors=['red'])
    ovR = draw_segmentation_masks(origR, predR, alpha=args.alpha, colors=['red'])

    # Draw all overlays on ovL
    needs_overlay = target_maskL is not None or (H_current is not None and K_left is not None) or pose_label
    if needs_overlay:
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
    combo = torch.cat([torch.cat([ovL, msL], dim=2), torch.cat([ovR, msR], dim=2)], dim=1)
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

    # Motion detection
    with torch.no_grad():
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
    if args.display_progress or args.save_frames or args.save_video:
        return best_p, opt, best_pL, best_pR, iter_rate, crop_coords
    return best_p, opt, None, None, iter_rate, None


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
    """Convert sensor_msgs/Image to a (H, W, 3) uint8 RGB numpy array.

    Uses numpy slicing for the channel reorder (BGR<->RGB and alpha-strip)
    instead of cv2.cvtColor — saves ~10 ms/frame at 1080p stereo (measured
    in bag mode). `np.ascontiguousarray(buf[..., ::-1])` is a strided
    reverse-and-copy, near-bandwidth-bound on CPU.
    """
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
        prev_opt_end_event=None):
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
        # `prob_batch` was produced on stream_seg (next_stream) by the
        # previous iteration's prefetch. We use it below on stream_opt
        # (current_stream). With fixed-role streams there's no swap that
        # would put consumer and producer on the same stream automatically,
        # so insert an explicit GPU-side fence: stream_opt's queued kernels
        # won't dispatch until stream_seg has reached this point. The host
        # does NOT block — both streams stay free for additional work.
        # In steady state the prefetch has typically completed during the
        # ~30 ms gap of the prior iteration's opt, so the fence is usually
        # a no-op on the GPU.
        current_stream.wait_stream(next_stream)
    else:
        with torch.cuda.stream(current_stream):
            seg_start_event.record(current_stream)
            prob_batch = extract_mask(
                torch.stack([left_img_t, right_img_t], dim=0), current_stream)
            seg_end_event.record(current_stream)

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
        opt_end_event.record(current_stream)

    dual_stream_mgr.synchronize_current()
    if no_pipeline:
        torch.cuda.synchronize()
    optimization_events.append((opt_start_event, opt_end_event))

    # Unpack result
    best_p, opt, best_pL, best_pR, iter_rate, crop_coords = result
    prev_9d = best_p.squeeze(0).detach().clone() if best_p.dim() > 1 else best_p.detach().clone()
    iteration_rates.append(iter_rate)

    # Save extrinsics
    best_p_4x4 = pk.se3_9d_to_matrix44(
        best_p.unsqueeze(0) if best_p.dim() == 1 else best_p)[0]
    H_l2b_t = torch.linalg.inv(best_p_4x4)
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

    # Display and save video
    if (args.display_progress or args.save_video) and best_pL is not None:
        pose_label = (f"t={t_err:.1f}mm r={r_err:.2f}deg"
                      if H_target_for_error is not None else None)
        cached_fk_vertices = update_fk_cache(js_buf, scene)
        arr = create_optimizer_visualization(
            best_pL, best_pR, probL_t, probR_t, left_img_t, right_img_t, args, frame_i,
            scene=scene, cached_fk_vertices=cached_fk_vertices, target_9d=target_9d,
            H_current=H_l2b_np, H_target=H_target_for_error, K_left=K_left,
            pose_label=pose_label, crop_coords=crop_coords)
        if arr is not None:
            frame_display = arr.cpu().numpy() if hasattr(arr, 'cpu') else np.asarray(arr)
            if frame_display.dtype != np.uint8:
                frame_display = frame_display.astype(np.uint8)
            if args.save_video and video_writer is not None:
                if args.video_scale != 1.0:
                    new_h = int(frame_display.shape[0] * args.video_scale)
                    new_w = int(frame_display.shape[1] * args.video_scale)
                    video_writer.write(cv2.resize(frame_display, (new_w, new_h),
                                                  interpolation=cv2.INTER_AREA))
                else:
                    video_writer.write(frame_display)
            if args.display_progress:
                cv2.imshow(win, frame_display)
                cv2.waitKey(1)

    dual_stream_mgr.swap_streams()
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