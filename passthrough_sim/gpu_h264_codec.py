#!/usr/bin/env python3
"""
gpu_h264_codec.py

GPU-resident H.264 encode/decode for the passthrough pipeline. The frame stays
a CUDA tensor across the boundary; only the compressed NAL byte string ever
touches host memory.

  GpuH264Encoder.encode(rgba_hwc) : (H,W,4) uint8 CUDA RGBA -> H.264 Annex-B bytes
        NVENC consumes the packed RGBA surface directly (ABGR input format) and
        does the RGB->YUV conversion ON THE ENCODE ENGINE — no SM colour convert.
        Used by passthrough_cloudxr_producer (one encoder per eye).

  GpuH264Decoder.decode(payload)  : H.264 Annex-B bytes -> (3,H,W) uint8 CUDA RGB
        NVDEC emits RGB directly (output_color_type=RGB) so the YUV->RGB convert
        runs ON THE DECODE ENGINE, not the SMs. Falls back to NV12 + a torch CSC
        only if the installed build's low-level decoder rejects RGB output.
        Used by passthrough_rectifier / consumer (one decoder per stream).

The whole point: NVENC/NVDEC are dedicated engines, so keeping the colour
conversion on them (not in torch kernels on the SMs) means the codec never
competes with the optimiser for shader cores.

NOTE ON THE PyNvVideoCodec API: CreateDecoder/Decode below are copied from the
working GpuCameraDecoder on this stack and are safe. CreateEncoder/Encode/
EndEncode follow the PyNvVideoCodec 2.x encoder API; if your installed point
release differs (e.g. the Encode input wants an AppFrame/device pointer rather
than the tensor directly), that is the only thing to adjust — everything else
here is environment-independent.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── BT.601 limited-range colour transforms (lazy, per-device) ───────────────
_RGB2YUV = None
_YUV2RGB = None
_YUV_OFF = None


def _ensure_coeffs(device):
    global _RGB2YUV, _YUV2RGB, _YUV_OFF
    if _RGB2YUV is not None and _RGB2YUV.device == device:
        return
    # full-range RGB[0,255] -> studio-swing YCbCr (Y in [16,235], C in [16,240])
    _RGB2YUV = torch.tensor([
        [0.256788,  0.504129,  0.097906],
        [-0.148223, -0.290993, 0.439216],
        [0.439216, -0.367788, -0.071427],
    ], dtype=torch.float32, device=device)
    _YUV_OFF = torch.tensor([16.0, 128.0, 128.0],
                            dtype=torch.float32, device=device)
    # inverse (studio-swing YCbCr -> full-range RGB)
    _YUV2RGB = torch.tensor([
        [1.164383, 0.000000,  1.596027],
        [1.164383, -0.391762, -0.812968],
        [1.164383, 2.017232,  0.000000],
    ], dtype=torch.float32, device=device)


# NOTE: rgb_chw_to_nv12 was removed — NVENC now does RGB->YUV on the encode
# engine (ABGR input format), so the producer no longer runs a CSC on the SMs.
# nv12_to_rgb_chw is kept only as the decoder's fallback when the installed
# build can't emit RGB directly from NVDEC.


def nv12_to_rgb_chw(nv12: "torch.Tensor") -> "torch.Tensor":
    """(3H//2, W) uint8 CUDA NV12 -> (3,H,W) uint8 CUDA RGB."""
    device = nv12.device
    _ensure_coeffs(device)
    h32, W = int(nv12.shape[0]), int(nv12.shape[1])
    H = (h32 * 2) // 3
    Y = nv12[:H].to(torch.float32)                                  # (H,W)
    uv = nv12[H:].reshape(H // 2, W // 2, 2).to(torch.float32)
    Cb, Cr = uv[..., 0], uv[..., 1]                                 # (H/2,W/2)
    UV = F.interpolate(torch.stack((Cb, Cr), 0).unsqueeze(0),
                       size=(H, W), mode="bilinear",
                       align_corners=False)[0]                       # (2,H,W)
    ycbcr = torch.stack((Y, UV[0], UV[1]), 0) - _YUV_OFF[:, None, None]
    rgb = torch.einsum("cd,dhw->chw", _YUV2RGB, ycbcr)
    return rgb.clamp_(0, 255).round().to(torch.uint8)


def ycbcr_planes_to_rgb(y: "torch.Tensor",
                        cb: "torch.Tensor",
                        cr: "torch.Tensor") -> "torch.Tensor":
    """Studio-swing YCbCr planes (each float32 (H,W), same size) -> (3,H,W)
    float32 RGB in [0,255] (caller rounds/casts). Used by the rectifier's
    NV12-domain path: the planes are rectified/downsampled FIRST (grid_sample
    on Y and UV at output resolution), so this CSC runs on ~0.55 MP instead of
    the decoder's full ~3.7 MP — that was the bulk of the codec path's SM tax."""
    _ensure_coeffs(y.device)
    ycbcr = torch.stack((y, cb, cr), 0) - _YUV_OFF[:, None, None]
    rgb = torch.einsum("cd,dhw->chw", _YUV2RGB, ycbcr)
    return rgb.clamp_(0, 255)


# ── Encoder (producer side) ─────────────────────────────────────────────────
class GpuH264Encoder:
    """One NVENC stream. encode(rgba_hwc) -> H.264 Annex-B NAL bytes.

    Stays GPU-resident and OFF the SMs: NVENC consumes the packed RGBA surface
    directly (ABGR input) and does the RGB->YUV conversion on the encode engine.
    Configured for low latency (no B-frames) with a periodic IDR so a persistent
    NVDEC decoder can lock on at the first keyframe.
    """

    def __init__(self, width: int, height: int, fps: int = 30,
                 gop: int = 30, bitrate: int = 12_000_000,
                 device: str = "cuda:0", gpuid: int = 0, fmt: str = "ABGR"):
        import PyNvVideoCodec as nvc
        self._nvc = nvc
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self.fmt = fmt
        # NOTE (channel order): NVENC "ABGR" is the word-ordered A8B8G8R8 format,
        # i.e. byte order R,G,B,A — which matches the FD-export RGBA buffer, so we
        # feed it straight through. If red/blue come out swapped on the overlay,
        # switch fmt to "ARGB" (byte order B,G,R,A). Both are always-supported.
        # NOTE (API): PyNvVideoCodec 2.x CreateEncoder; usecpuinputbuffer=False so
        # NVENC reads the CUDA surface directly.
        self._enc = nvc.CreateEncoder(
            self.width, self.height, fmt, False,
            codec="h264",
            preset="P3",
            tuning_info="ultra_low_latency",
            gop=int(gop),
            idrperiod=int(gop),
            bf=0,                                      # no B-frames
            bitrate=int(bitrate),
            fps=int(fps),
            rc="cbr",
            repeatspspps=1,                            # SPS/PPS on every IDR
        )

    def encode(self, rgba_hwc: "torch.Tensor") -> bytes:
        """rgba_hwc: (H,W,4) uint8 CUDA RGBA at the encoder's resolution. NVENC
        does the CSC. Returns the Annex-B bytes (may be empty if buffering).
        NOTE (API): if Encode wants a flat/AppFrame buffer rather than the (H,W,4)
        tensor's CUDA-array-interface, that's the only thing to adjust here."""
        bitstream = self._enc.Encode(rgba_hwc.contiguous())
        return bytes(bitstream) if bitstream else b""

    def flush(self) -> bytes:
        try:
            return bytes(self._enc.EndEncode())
        except Exception:
            return b""


# ── Decoder (rectifier / consumer side) ─────────────────────────────────────
class GpuH264Decoder:
    """One NVDEC stream. decode(payload) -> (3,H,W) uint8 CUDA RGB, or None on
    parser warmup / no frame yet.

    Asks NVDEC to emit RGB directly so the YUV->RGB conversion runs on the
    decode engine, not the SMs. If the installed build's low-level decoder
    rejects RGB output, falls back to NV12 + the torch CSC (correct, just back
    on the SMs)."""

    def __init__(self, device: str = "cuda:0", gpuid: int = 0,
                 output: str = "rgb"):
        import PyNvVideoCodec as nvc
        self._nvc = nvc
        self.device = device
        self._output = output
        self._rgb_out = False
        common = dict(
            gpuid=gpuid,
            codec=nvc.cudaVideoCodec.H264,
            cudacontext=0,        # 0 => torch's primary context (zero-copy)
            cudastream=0,
            usedevicememory=True,
        )
        # ── output="nv12": engine-native surfaces, zero post-processing ─────
        # The decoder hands back NVDEC's NV12 surface AS-IS: no library CSC
        # kernel, no clone, nothing on the SMs. decode() returns the (3H/2, W)
        # uint8 CUDA tensor VIEWING the decoder's reused surface pool.
        # CONTRACT: the caller must fully consume (materialise) the frame
        # before calling decode() again on this instance — any pending kernels
        # reading the surface race with the next decode overwriting it. The
        # rectifier satisfies this structurally: its per-frame publish ends in
        # a blocking D2H (.cpu()), which drains every kernel that read the
        # surface before the next ROS callback can decode.
        if output == "nv12":
            self._dec = nvc.CreateDecoder(**common)
            self._fmt = None          # confirmed 'nv12' on first frame
            print("[GpuH264Decoder] output=nv12 (engine-native, no CSC, "
                  "no clone; zero-copy contract — see class docs)")
            return
        # ── output="rgb" (legacy / consumer path) ────────────────────────────
        # NVDEC engine-side colour conversion. The LOW-LEVEL decoder uses the
        # camelCase `outputColorType` (snake_case `output_color_type` is the
        # SimpleDecoder form — passing it here is silently rejected and we fall
        # back to NV12+SM CSC, which is the bug this fixes). RGBP = planar RGB
        # (3,H,W), exactly what _publish/the optimiser want; RGB = HWC fallback.
        for _oct in ("RGBP", "RGB"):
            try:
                self._dec = nvc.CreateDecoder(
                    outputColorType=getattr(nvc.OutputColorType, _oct), **common)
                self._rgb_out = True
                break
            except Exception:
                continue
        if not self._rgb_out:
            self._dec = nvc.CreateDecoder(**common)        # NV12 + SM CSC fallback
        # _rgb_out only means CreateDecoder ACCEPTED the RGB request; it does NOT
        # guarantee the engine emits RGB. The true format is detected from the
        # first decoded frame's shape (see decode) and logged honestly there.
        self._fmt = None              # 'chw' | 'hwc' | 'nv12', set on first decode
        print(f"[GpuH264Decoder] RGBP requested (CreateDecoder accepted="
              f"{self._rgb_out}); actual engine output confirmed on first frame")

    @staticmethod
    def _classify(fr) -> str:
        if fr.ndim == 3 and fr.shape[0] == 3:        # planar (3,H,W) — engine RGB
            return "chw"
        if fr.ndim == 3 and fr.shape[2] in (3, 4):   # packed (H,W,3|4) — engine RGB
            return "hwc"
        return "nv12"                                # 2-D NV12 — engine RGB NOT honored

    def decode(self, payload: bytes):
        # numpy view over the payload; keep `arr` alive across Decode() because
        # PacketData holds a raw pointer into it.
        arr = np.frombuffer(payload, dtype=np.uint8)
        pd = self._nvc.PacketData()
        pd.bsl_data = arr.ctypes.data
        pd.bsl = int(arr.size)
        try:
            frames = self._dec.Decode(pd) or []
        except Exception:
            return None
        if not frames:
            return None
        fr = torch.from_dlpack(frames[-1])
        # ── output="nv12": hand back the raw surface view, zero post-work ───
        if self._output == "nv12":
            if self._fmt is None:
                self._fmt = "nv12_raw"
                print(f"[GpuH264Decoder] first frame shape={tuple(fr.shape)} "
                      f"dtype={fr.dtype} -> nv12 surface view (no clone)")
                if fr.ndim != 2:
                    print("[GpuH264Decoder] *** expected 2-D NV12 from the "
                          "engine but got the shape above — the installed "
                          "build post-processes by default; NV12-domain path "
                          "may be invalid here ***")
            return fr if fr.ndim == 2 else None
        if self._fmt is None:
            # Route on the ACTUAL frame shape, not on whether CreateDecoder accepted
            # the RGB request. If this prints 'nv12', the engine ignored the RGB
            # request and the YUV->RGB convert is running on the SMs every frame
            # (full-res einsum + bilinear) — the unnecessary GPU load. 'chw'/'hwc'
            # means the convert genuinely ran on the decode engine.
            self._fmt = self._classify(fr)
            msg = (f"[GpuH264Decoder] first frame shape={tuple(fr.shape)} "
                   f"dtype={fr.dtype} contig={fr.is_contiguous()} -> {self._fmt}")
            if self._fmt == "nv12":
                msg += "  *** engine RGB NOT honored: YUV->RGB on the SMs ***"
            print(msg)
        # One copy out of NVDEC's reused surface pool, then return (3,H,W) uint8.
        if self._fmt == "chw":
            return fr.clone()                                # planar (3,H,W)
        if self._fmt == "hwc":
            return fr[..., :3].permute(2, 0, 1).contiguous() # single copy
        return nv12_to_rgb_chw(fr.clone())                   # SM CSC (engine RGB off)