#!/usr/bin/env python3
"""
g18_capture.py  —  RealSense D435 capture script for G18 Orthopedic Cast Project
==================================================================================
Run on your LOCAL PC (not Colab) with the RealSense D435 connected via USB 3.0.

USAGE
-----
    python g18_capture.py                     # 36 frames, save to ~/Desktop/g18_capture
    python g18_capture.py --n 72              # 72 frames (5° steps)
    python g18_capture.py --save D:/captures  # custom output folder

WORKFLOW
--------
  1. Run this script.
  2. Step 0:  aim camera at EMPTY scene (no arm) → press SPACE → background saved.
  3. Steps 1-36:  place arm → press SPACE for each rotation position.
  4. Script saves everything and prints zip instructions.
  5. Zip the g18_capture/ folder → upload to Google Drive → G18_Project/g18_capture.zip
  6. In Colab set CAPTURE_MODE = 'real_rgbd' and run all cells.

OUTPUT STRUCTURE
----------------
    ~/Desktop/g18_capture/
    ├── intrinsics.json          ← camera fx fy cx cy, width, height
    ├── background.png           ← captured BEFORE arm (used for colour silhouette)
    ├── capture_log.json         ← per-frame timestamps and depth statistics
    ├── rgb/
    │   ├── frame_000.png        ← 640×480 BGR colour image, position 0
    │   ├── frame_001.png        ← position 1  (rotate ~10° before pressing SPACE)
    │   └── ...  frame_035.png   ← position 35
    └── depth/
        ├── frame_000.npy        ← float32 numpy array, 640×480, METRES
        ├── frame_001.npy
        └── ...  frame_035.npy

IMPORTANT NOTES
---------------
  • Depth arrays are float32 in METRES (0.452 = 452 mm).  Zero = no data.
  • frame_000.png and frame_000.npy are the SAME physical position.
  • Frame numbers are zero-padded to 3 digits: frame_000, not frame_0.
  • Keep USB 3.0 (blue port). USB 2.0 will work but drops depth quality.
  • Even diffuse lighting gives better silhouettes (avoid direct sunlight).
  • Plain cloth background (dark or white) improves colour silhouette quality.

CAPTURE TIPS
------------
  • Camera should be ~40–60 cm from the arm.
  • Rest arm on flat surface, palm-down or natural position.
  • Rotate THE ARM (not the camera) between positions.
  • Each position: rotate ~10° (for 36 frames) or ~5° (for 72 frames).
  • Keep arm still when pressing SPACE — motion blur degrades depth.
  • For 36 frames: every 10° = one full rotation around the long axis.
  • You do NOT need to go beyond 360° — stop after one full rotation.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print('ERROR: pyrealsense2 not installed.')
    print('Install with:  pip install pyrealsense2')
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (change these if needed)
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_SAVE_DIR  = os.path.join(os.path.expanduser('~'), 'Desktop', 'g18_capture')
DEFAULT_N_FRAMES  = 36       # 36 = every 10°,  72 = every 5°
WARMUP_FRAMES     = 60       # frames to discard while auto-exposure settles
RES_W, RES_H      = 640, 480
CAMERA_FPS        = 30
DEPTH_NEAR_M      = 0.10     # ignore depth closer than 10 cm
DEPTH_FAR_M       = 1.50     # ignore depth farther than 150 cm
PREVIEW_W         = 1000     # total preview window width (RGB left + depth right)
PREVIEW_H         = 500      # preview window height
# ════════════════════════════════════════════════════════════════════════════


# ── Helpers ─────────────────────────────────────────────────────────────────

def colorise_depth(depth_m: np.ndarray,
                   near: float = DEPTH_NEAR_M,
                   far:  float = DEPTH_FAR_M) -> np.ndarray:
    """Convert float32 metres depth map → uint8 BGR JET heatmap for display."""
    d_clip = np.clip(depth_m, near, far)
    norm   = ((d_clip - near) / (far - near) * 255).astype(np.uint8)
    norm[depth_m == 0] = 0                        # zero = no data → black
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def grab_aligned_frame(pipeline, align, depth_scale: float):
    """
    Grab one aligned RGB+depth frame pair from the RealSense pipeline.
    Returns:
        bgr   (H×W×3 uint8)          — BGR colour image
        depth (H×W float32, metres)  — depth in metres, invalid pixels = 0
    Raises RuntimeError after 10 failed attempts.
    """
    for attempt in range(10):
        frames   = pipeline.wait_for_frames()
        aligned  = align.process(frames)
        color_f  = aligned.get_color_frame()
        depth_f  = aligned.get_depth_frame()
        if color_f and depth_f:
            bgr     = np.asanyarray(color_f.get_data())             # uint8 BGR
            d_raw   = np.asanyarray(depth_f.get_data()).astype(np.float32)
            depth_m = d_raw * depth_scale                           # → metres
            # Clamp to valid range
            depth_m[(depth_m < DEPTH_NEAR_M) | (depth_m > DEPTH_FAR_M)] = 0.0
            return bgr, depth_m
    raise RuntimeError('Could not grab a valid aligned frame after 10 attempts.')


def depth_stats(depth_m: np.ndarray) -> dict:
    """Return basic statistics about valid depth pixels (non-zero, in range)."""
    valid = depth_m[(depth_m > DEPTH_NEAR_M) & (depth_m < DEPTH_FAR_M)]
    if len(valid) == 0:
        return {'min_m': 0.0, 'max_m': 0.0, 'mean_m': 0.0,
                'median_m': 0.0, 'valid_pct': 0.0}
    return {
        'min_m':     float(valid.min()),
        'max_m':     float(valid.max()),
        'mean_m':    float(valid.mean()),
        'median_m':  float(np.median(valid)),
        'valid_pct': float(len(valid) / depth_m.size * 100),
    }


def overlay_text(img: np.ndarray, lines: list, start_y: int = 30,
                 dy: int = 28, color=(0, 255, 0), scale: float = 0.65):
    """Draw multiple lines of text on an image in-place."""
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, start_y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def build_preview(bgr: np.ndarray, depth_m: np.ndarray,
                  lines_left: list, lines_right: list) -> np.ndarray:
    """
    Build a side-by-side preview: colour (left) + depth heatmap (right).
    Resized to PREVIEW_W × PREVIEW_H.
    """
    half_w = PREVIEW_W // 2
    left   = cv2.resize(bgr.copy(), (half_w, PREVIEW_H))
    right  = cv2.resize(colorise_depth(depth_m), (half_w, PREVIEW_H))

    overlay_text(left,  lines_left,  color=(0, 255, 0))
    overlay_text(right, lines_right, color=(0, 255, 255))

    return np.hstack([left, right])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='G18 RealSense capture script — saves RGB + depth for orthopedic cast.')
    parser.add_argument('--n',    type=int, default=DEFAULT_N_FRAMES,
                        help=f'Number of arm positions to capture (default {DEFAULT_N_FRAMES})')
    parser.add_argument('--save', type=str, default=DEFAULT_SAVE_DIR,
                        help=f'Output directory (default: {DEFAULT_SAVE_DIR})')
    args = parser.parse_args()

    save_dir   = args.save
    n_frames   = args.n
    rgb_dir    = os.path.join(save_dir, 'rgb')
    depth_dir  = os.path.join(save_dir, 'depth')

    os.makedirs(rgb_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print('=' * 62)
    print('  G18 RealSense Capture Script')
    print('  G18 — 3D Voronoi Orthopedic Cast System')
    print('  AUM · College of Engineering & Technology · GP2 2026')
    print('=' * 62)
    print(f'  Output folder : {save_dir}')
    print(f'  Frames to capture : {n_frames}')
    print(f'  Rotation step : {360/n_frames:.1f}° per frame')
    print(f'  Resolution : {RES_W}×{RES_H}  FPS={CAMERA_FPS}')
    print()

    # ── Start RealSense pipeline ──────────────────────────────────────────────
    print('Connecting to RealSense D435...')
    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, CAMERA_FPS)
    rs_cfg.enable_stream(rs.stream.depth, RES_W, RES_H, rs.format.z16,  CAMERA_FPS)

    try:
        profile = pipeline.start(rs_cfg)
    except RuntimeError as e:
        print(f'\nERROR: Could not start RealSense: {e}')
        print('→ Check USB 3.0 connection and RealSense SDK installation.')
        sys.exit(1)

    print('Camera connected.')

    # ── Read camera intrinsics ────────────────────────────────────────────────
    col_profile  = profile.get_stream(rs.stream.color)
    intr         = col_profile.as_video_stream_profile().get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()   # typically 0.001 (mm → m)

    intrinsics_dict = {
        'fx':              float(intr.fx),
        'fy':              float(intr.fy),
        'cx':              float(intr.ppx),
        'cy':              float(intr.ppy),
        'width':           intr.width,
        'height':          intr.height,
        'depth_scale':     1.0,             # depth .npy files already in metres
        'raw_depth_scale': float(depth_scale),
        'format':          'float32_metres',
        'capture_fps':     CAMERA_FPS,
        'n_frames':        n_frames,
        'step_deg':        round(360.0 / n_frames, 2),
    }

    intr_path = os.path.join(save_dir, 'intrinsics.json')
    with open(intr_path, 'w') as f:
        json.dump(intrinsics_dict, f, indent=2)

    print(f'✅ intrinsics.json saved')
    print(f'   fx={intr.fx:.2f}  fy={intr.fy:.2f}  '
          f'cx={intr.ppx:.2f}  cy={intr.ppy:.2f}')

    # ── Align depth → colour ─────────────────────────────────────────────────
    align = rs.align(rs.stream.color)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f'\nWaiting for camera to stabilise ({WARMUP_FRAMES} warmup frames)...')
    for wi in range(WARMUP_FRAMES):
        pipeline.wait_for_frames()
        if (wi + 1) % 20 == 0:
            print(f'  {wi+1}/{WARMUP_FRAMES}')
    print('Camera ready.\n')

    # ── Open preview window ───────────────────────────────────────────────────
    cv2.namedWindow('G18 Capture [LEFT=colour  RIGHT=depth]', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('G18 Capture [LEFT=colour  RIGHT=depth]', PREVIEW_W, PREVIEW_H)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 0 — Background capture
    # ════════════════════════════════════════════════════════════════════════
    print('━' * 62)
    print('  STEP 0 — BACKGROUND CAPTURE')
    print()
    print('  ▶  Remove the arm from the scene.')
    print('  ▶  Camera stays in SAME position, SAME lighting.')
    print('  ▶  Press  SPACE  to capture the background frame.')
    print('  ▶  Press  Q  to quit.')
    print('━' * 62)

    bg_captured = False
    while not bg_captured:
        bgr, depth_m = grab_aligned_frame(pipeline, align, depth_scale)
        stats_bg     = depth_stats(depth_m)
        preview = build_preview(
            bgr, depth_m,
            lines_left  = ['STEP 0: BACKGROUND',
                            'Remove arm from scene',
                            'SPACE = capture,  Q = quit'],
            lines_right = [f'Valid depth: {stats_bg["valid_pct"]:.0f}%',
                            f'Median: {stats_bg["median_m"]*1000:.0f} mm'],
        )
        cv2.imshow('G18 Capture [LEFT=colour  RIGHT=depth]', preview)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            bg_path = os.path.join(save_dir, 'background.png')
            cv2.imwrite(bg_path, bgr)
            print(f'  ✅ background.png saved')
            bg_captured = True

        elif key in (ord('q'), 27):   # Q or Escape
            print('Capture aborted at background step.')
            pipeline.stop()
            cv2.destroyAllWindows()
            sys.exit(0)

    # ════════════════════════════════════════════════════════════════════════
    # STEPS 1 – N_FRAMES — Arm frame capture
    # ════════════════════════════════════════════════════════════════════════
    print()
    print('━' * 62)
    print(f'  STEPS 1–{n_frames} — ARM CAPTURE')
    print()
    print(f'  ▶  Place arm in frame, palm-down or natural relaxed position.')
    print(f'  ▶  Keep arm still — rotate {360/n_frames:.0f}° BETWEEN presses.')
    print(f'  ▶  Press  SPACE  to capture each of the {n_frames} positions.')
    print(f'  ▶  Press  Q  to quit early.')
    print(f'  ▶  Target: rotate arm one full 360° turn over {n_frames} presses.')
    print('━' * 62)
    print()

    capture_log = []
    frame_idx   = 0

    while frame_idx < n_frames:
        bgr, depth_m = grab_aligned_frame(pipeline, align, depth_scale)
        stats        = depth_stats(depth_m)
        angle_deg    = 360.0 * frame_idx / n_frames
        progress_pct = 100.0 * frame_idx / n_frames

        # Progress bar string
        done  = int(progress_pct / 5)  # 20 chars wide
        bar   = '█' * done + '░' * (20 - done)
        pbar  = f'[{bar}] {progress_pct:.0f}%'

        preview = build_preview(
            bgr, depth_m,
            lines_left = [
                f'Frame {frame_idx:03d} / {n_frames-1:03d}',
                f'Angle: {angle_deg:.0f} deg',
                f'{pbar}',
                'SPACE=capture    Q=quit',
            ],
            lines_right = [
                f'Valid depth: {stats["valid_pct"]:.0f}%',
                f'Median: {stats["median_m"]*1000:.0f} mm',
                f'Range: {stats["min_m"]*1000:.0f}–{stats["max_m"]*1000:.0f} mm',
            ],
        )
        cv2.imshow('G18 Capture [LEFT=colour  RIGHT=depth]', preview)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            # ── Save RGB ──────────────────────────────────────────────────
            rgb_path = os.path.join(rgb_dir,   f'frame_{frame_idx:03d}.png')
            cv2.imwrite(rgb_path, bgr)

            # ── Save depth (float32 metres) ───────────────────────────────
            dep_path = os.path.join(depth_dir, f'frame_{frame_idx:03d}.npy')
            np.save(dep_path, depth_m.astype(np.float32))

            capture_log.append({
                'frame':     frame_idx,
                'angle_deg': round(angle_deg, 1),
                'timestamp': round(time.time(), 3),
                'depth':     stats,
            })

            print(f'  ✅ frame_{frame_idx:03d}  '
                  f'angle={angle_deg:5.1f}°  '
                  f'depth_med={stats["median_m"]*1000:.0f}mm  '
                  f'valid={stats["valid_pct"]:.0f}%')
            frame_idx += 1

        elif key in (ord('q'), 27):
            print(f'\n  Capture aborted at frame {frame_idx}/{n_frames}.')
            break

    # ── Cleanup ────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    pipeline.stop()

    # ── Save capture log ───────────────────────────────────────────────────
    log_data = {
        'project':      'G18 — 3D Voronoi Orthopedic Cast',
        'captured_at':  time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_captured':   frame_idx,
        'n_requested':  n_frames,
        'step_deg':     round(360.0 / n_frames, 2),
        'resolution':   {'width': RES_W, 'height': RES_H},
        'depth_range_m': [DEPTH_NEAR_M, DEPTH_FAR_M],
        'intrinsics':   intrinsics_dict,
        'frames':       capture_log,
    }
    log_path = os.path.join(save_dir, 'capture_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    if frame_idx > 0:
        median_depths = [fr['depth']['median_m'] for fr in capture_log
                         if fr['depth']['median_m'] > 0]
        median_cam_dist = np.median(median_depths) * 1000 if median_depths else 0
    else:
        median_cam_dist = 0

    print()
    print('═' * 62)
    print(f'  CAPTURE SUMMARY')
    print(f'  Frames captured : {frame_idx} / {n_frames}')
    print(f'  Output folder   : {save_dir}')
    print(f'  Camera-arm dist : ~{median_cam_dist:.0f} mm (median depth)')
    print()
    print(f'  Files saved:')
    print(f'    intrinsics.json       ← camera calibration')
    print(f'    background.png        ← background reference')
    print(f'    capture_log.json      ← per-frame statistics')
    print(f'    rgb/frame_000.png  … frame_{frame_idx-1:03d}.png')
    print(f'    depth/frame_000.npy … frame_{frame_idx-1:03d}.npy')
    print()
    print('  ─── NEXT STEPS ──────────────────────────────────────────')
    print()
    print('  1. ZIP the g18_capture folder:')
    print(f'     Right-click "{save_dir}"')
    print('     → Send to → Compressed (zipped) folder')
    print('     Result: g18_capture.zip')
    print()
    print('  2. Upload to Google Drive:')
    print('     My Drive → G18_Project → upload g18_capture.zip')
    print()
    print('  3. In Colab notebook:')
    print('     Change:  CAPTURE_MODE = "real_rgbd"')
    print('     Then run all cells top to bottom.')
    print('═' * 62)


if __name__ == '__main__':
    main()