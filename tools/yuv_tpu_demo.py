#!/usr/bin/env python3
"""
End-to-end demo of the v4 YUV420-input cvimodel, simulating the real camera pipeline:

    tennis .jpg
      -> letterbox to 640x480 (keep aspect ratio, gray-114 pad)
      -> RGB -> YUV (BT.601 studio-swing, the exact inverse of the TPU's fused YUV->RGB)
      -> subsample to YUV422P     (this is what the camera hands us)
      -> CPU convert YUV422P -> YUV420P (I420)   <-- mirrors the Rust yuv422p_to_yuv420p()
      -> feed the raw I420 buffer (460800 bytes) into the cvimodel (TPU does YUV->RGB + normalize)
      -> decode YOLOv8 head, NMS, un-letterbox
      -> draw boxes on the original image

Run INSIDE the tpu-mlir container (model_runner.py must be on PATH).
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

# Model spatial size (W x H). The v4 cvimodel input is 640 wide, 480 tall.
W, H = 640, 480
Y_SIZE = W * H                     # 307200
C_W, C_H = W // 2, H // 2          # 320 x 240
C_SIZE = C_W * C_H                 # 76800
I420_SIZE = Y_SIZE + 2 * C_SIZE    # 460800


# --------------------------------------------------------------------------- #
# color conversion : BT.601 limited/studio range  (matches TPU fused-preprocess)
# --------------------------------------------------------------------------- #
def rgb_to_yuv601_limited(rgb: np.ndarray):
    """rgb: (H,W,3) float32 in 0..255 (R,G,B order). Returns full-res Y, Cb, Cr float32."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 16.0 + (65.481 * R + 128.553 * G + 24.966 * B) / 255.0
    Cb = 128.0 + (-37.797 * R - 74.203 * G + 112.000 * B) / 255.0
    Cr = 128.0 + (112.000 * R - 93.786 * G - 18.214 * B) / 255.0
    return Y, Cb, Cr


def letterbox(img_rgb: np.ndarray, new_w=W, new_h=H, color=(114, 114, 114)):
    """Aspect-preserving resize + center pad. Returns (canvas, scale, pad_x, pad_y)."""
    h0, w0 = img_rgb.shape[:2]
    r = min(new_w / w0, new_h / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    px, py = (new_w - nw) // 2, (new_h - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, r, px, py


# --------------------------------------------------------------------------- #
# camera simulation + the CPU 422->420 step we want to offload to the TPU
# --------------------------------------------------------------------------- #
def rgb_to_yuv422p(lb_rgb: np.ndarray):
    """RGB (H,W,3) uint8 -> planar YUV422P buffers (Y HxW, U/V Hx(W/2)) — 'camera frame'."""
    Yf, Cbf, Crf = rgb_to_yuv601_limited(lb_rgb.astype(np.float32))
    Yf = np.clip(np.round(Yf), 0, 255).astype(np.uint8)
    Cbf = np.clip(np.round(Cbf), 0, 255).astype(np.uint8)
    Crf = np.clip(np.round(Crf), 0, 255).astype(np.uint8)
    # horizontal 2:1 chroma subsample (average the pair) -> 4:2:2
    U = ((Cbf[:, 0::2].astype(np.uint16) + Cbf[:, 1::2] + 1) // 2).astype(np.uint8)  # (H, W/2)
    V = ((Crf[:, 0::2].astype(np.uint16) + Crf[:, 1::2] + 1) // 2).astype(np.uint8)
    return Yf, U, V


def yuv422p_to_yuv420p(Y: np.ndarray, U422: np.ndarray, V422: np.ndarray) -> np.ndarray:
    """Mirror of the board-side Rust yuv422p_to_yuv420p: vertical 2:1 average of chroma.
    Returns the tight I420 buffer (Y then U then V), 460800 bytes."""
    U420 = ((U422[0::2, :].astype(np.uint16) + U422[1::2, :] + 1) // 2).astype(np.uint8)  # (H/2, W/2)
    V420 = ((V422[0::2, :].astype(np.uint16) + V422[1::2, :] + 1) // 2).astype(np.uint8)
    i420 = np.concatenate([Y.reshape(-1), U420.reshape(-1), V420.reshape(-1)])
    assert i420.size == I420_SIZE, f"expected {I420_SIZE}, got {i420.size}"
    return i420


# --------------------------------------------------------------------------- #
# YOLOv8 decode
# --------------------------------------------------------------------------- #
def find_output(npz) -> np.ndarray:
    """Return the (5,6300) detection tensor regardless of the key/trailing dims."""
    for k in npz.files:
        a = np.squeeze(npz[k])
        if a.ndim == 2 and a.shape == (5, 6300):
            return a
        if a.ndim == 2 and a.shape == (6300, 5):
            return a.T
    raise RuntimeError(f"no (5,6300) output found; keys={list(npz.files)}")


def nms(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float):
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input tennis .jpg")
    ap.add_argument("--model", default="/workspace/tpu_convert/yolov8n_tennis_v4.cvimodel")
    ap.add_argument("--out", default="/workspace/tpu_convert/v4_demo_out.jpg")
    ap.add_argument("--workdir", default="/workspace/tpu_convert")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    in_npz = str(workdir / "v4_demo_in.npz")
    out_npz = str(workdir / "v4_demo_out.npz")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    print(f"[1] loaded {args.image}  ({w0}x{h0})")

    lb, scale, px, py = letterbox(img_rgb)
    print(f"[2] letterbox -> {W}x{H}  scale={scale:.4f} pad=({px},{py})")

    # camera hands us YUV422P
    Y, U422, V422 = rgb_to_yuv422p(lb)
    print(f"[3] simulated YUV422P frame: Y{Y.shape} U{U422.shape} V{V422.shape} "
          f"(= {Y.size + U422.size + V422.size} bytes)")

    # CPU 422 -> 420 (the step we push onto the TPU by feeding YUV420 directly)
    t0 = time.perf_counter()
    i420 = yuv422p_to_yuv420p(Y, U422, V422)
    t_cvt = (time.perf_counter() - t0) * 1000
    print(f"[4] CPU 422->420 : {i420.size} bytes (tight I420, no padding)  [{t_cvt:.2f} ms]")

    np.savez(in_npz, images_raw=i420.reshape(I420_SIZE, 1, 1))

    t0 = time.perf_counter()
    subprocess.run(
        ["model_runner.py", "--input", in_npz, "--model", args.model, "--output", out_npz],
        check=True,
    )
    t_inf = (time.perf_counter() - t0) * 1000
    print(f"[5] TPU (emulated) inference done  [{t_inf:.1f} ms wall, host-emulated]")

    det = find_output(np.load(out_npz))          # (5,6300)
    boxes = det[0:4, :].T                          # cx,cy,w,h  (letterboxed px)
    scores = det[4, :]
    m = scores >= args.conf
    boxes, scores = boxes[m], scores[m]
    print(f"[6] raw candidates >{args.conf}: {len(scores)}  (max score {scores.max() if len(scores) else 0:.4f})")

    # xywh -> xyxy in letterboxed space
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(xyxy, scores, args.iou) if len(scores) else []
    xyxy, scores = xyxy[keep], scores[keep]

    # un-letterbox back to original image coords
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / scale
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0 - 1)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0 - 1)

    print(f"[7] detections after NMS: {len(scores)}")
    for (x1, y1, x2, y2), s in zip(xyxy, scores):
        print(f"      ball  conf={s:.3f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"ball {s:.2f}", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.out, img_bgr)
    print(f"[8] wrote {args.out}")


if __name__ == "__main__":
    main()
