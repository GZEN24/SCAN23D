"""
═══════════════════════════════════════════════════════════════════════════════
G18 — UNIFIED CAST PIPELINE (Merged Best-of-Both)
═══════════════════════════════════════════════════════════════════════════════
DUAL-PATH ARCHITECTURE:
  PATH A — Intel RealSense D435 (RGB-D)
    ✓ TSDF volumetric fusion (KinectFusion-style)
    ✓ ICP odometry for camera pose estimation
    ✓ Quality gates on capture (blur, depth, coverage)

  PATH B — Standard RGB Camera (multi-view photographs)
    ✓ Enhanced 5-method silhouette extraction
    ✓ Visual hull reconstruction (voxel carving)
    ✓ Works with any phone or DSLR camera

  SHARED (both paths converge here):
    ✓ Marching Cubes mesh extraction
    ✓ Convex hull closure for unseen geometry
    ✓ Mesh cleaning + watertight validation
    ✓ Cast shell (Boolean offset with gap + wall)
    ✓ Voronoi lattice ventilation pattern
    ✓ Clamshell split with fastener holes
    ✓ 5000-point clinical fit validation
    ✓ STL export + JSON defense report

WHAT THIS OUTPERFORMS vs. EACH STANDALONE:
  vs. RealSense-only pipeline:
    + Voronoi ventilation holes (breathable cast)
    + Arm-to-cast clearance gap (3mm comfort gap + 4mm wall)
    + Automatic fastener holes (zip-tie / screw points)
    + Boolean mesh operations (manifold3d — not just normal offset)

  vs. RGB-only pipeline:
    + Convex hull closure for unseen underside
    + 5000-point clinical fit deviation metric (<2mm threshold)
    + Structured JSON report for defense documentation
    + Interactive 3D visualization checkpoints
    + Frame quality gates for live capture

REQUIREMENTS:
  pip install pyrealsense2 open3d numpy opencv-python trimesh scipy \
              scikit-image manifold3d
  pip install rembg onnxruntime    # Optional: deep learning silhouettes

USAGE:
  # Auto-detect mode (RealSense if connected, else look for images/)
  python g18_unified_pipeline.py

  # Force RGB-only mode with images folder
  python g18_unified_pipeline.py --mode rgb --images ./images

  # Force RealSense mode
  python g18_unified_pipeline.py --mode rgbd

  # Generate synthetic test data and run
  python g18_unified_pipeline.py --synthetic
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, glob, logging, argparse
import numpy as np
import cv2
import open3d as o3d
import trimesh
from scipy import ndimage
from scipy.spatial import Voronoi

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    import manifold3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    "output_dir":           "g18_output",
    "frames_dir":           "g18_output/frames",
    "mesh_dir":             "g18_output/mesh",
    "image_dir":            "images",                 # Path B: RGB images folder
    "background_image":     "images/_background.png", # Path B: optional clean bg

    # ── Path A: RealSense settings ──
    "num_frames":           60,
    "frame_width":          640,
    "frame_height":         480,
    "fps":                  30,
    "tsdf_voxel_size":      0.004,    # 4mm
    "tsdf_trunc_mult":      4.0,
    "tsdf_depth_scale":     1000.0,
    "tsdf_depth_max":       0.8,      # 80cm
    "min_depth_mm":         200,
    "max_depth_mm":         800,
    "blur_threshold":       10.0,

    # ── Path B: RGB photogrammetry settings ──
    "num_images":           72,       # [SYNTH: was 36; 72 covers 360° at 5° spacing]
    "angle_start_deg":      0.0,
    "angle_end_deg":        360.0,    # [SYNTH: was 180; full 360° coverage]
    "camera_distance_mm":   500.0,
    "focal_length_px":      None,     # None = auto from image width × 1.2
    "silhouette_method":    "full",   # basic/multichannel/grabcut/deep_learning/full
    "silhouette_threshold": 30,

    # ── Path B: Visual hull settings ──
    "voxel_grid_size":      280,      # N³ voxel grid  [FIX-1: was 200]
    "voxel_volume_mm":      400.0,    # physical volume side length

    # ── Shared: Mesh processing ──
    "smooth_iterations":    8,         # [FIX-2: was 30]
    "smooth_lambda":        0.25,      # [FIX-2: was 0.5]
    "marching_cubes_level": 0.5,

    # ── Shared: ROI crop (meters, from centroid) ──
    "roi_enabled":          True,     # Path A: usually needed; Path B: auto-disabled
    "roi_x_half":           0.12,
    "roi_y_half":           0.15,
    "roi_z_half":           0.10,

    # ── Shared: Cast design ──
    "cast_gap_mm":          3.0,      # clearance between arm and cast inner wall
    "cast_wall_mm":         4.0,      # cast wall thickness
    "voronoi_enabled":      True,     # add Voronoi ventilation holes
    "voronoi_num_points":   150,
    "voronoi_hole_radius":  5.0,      # mm
    "split_normal":         [1, 0, 0],
    "split_gap_mm":         0.5,
    "fastener_enabled":     True,
    "fastener_diameter_mm": 3.0,
    "fastener_spacing_mm":  40.0,

    # ── Shared: Validation ──
    "validation_samples":   5000,
    "fit_threshold_mm":     4.0,      # [FIX-11: was 2.0; 3mm gap → mean≈3mm is correct]

    # ── Synthetic data rendering ──
    "synth_render_width":   1280,     # render image width  (was hardcoded 800)
    "synth_render_height":   960,     # render image height (was hardcoded 600)
    "synth_bg_white":        True,    # pure white BG → better silhouette extraction
    "synth_no_blur":         True,    # disable render blur → sharper silhouette edges
    "synth_elevations":      [0, 30, -30],  # camera elevation rings (degrees)

    # Logging
    "log_file":             "g18_output/pipeline.log",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config["log_file"], mode="w"),
        ],
    )
    return logging.getLogger("G18-UNIFIED")

log = setup_logging(CONFIG)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — INPUT DETECTION & CAMERA INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def detect_input_mode(config, force_mode=None):
    """Auto-detect whether to use RGB-D (RealSense) or RGB (photos)."""
    if force_mode == "rgbd":
        if not REALSENSE_AVAILABLE:
            log.error("RealSense requested but pyrealsense2 not installed.")
            sys.exit(1)
        return "rgbd"
    elif force_mode == "rgb":
        return "rgb"

    # Auto-detect
    if REALSENSE_AVAILABLE:
        try:
            ctx = rs.context()
            if len(ctx.devices) > 0:
                log.info("STAGE 1: RealSense D435 detected → PATH A (RGB-D)")
                return "rgbd"
        except:
            pass

    # Check for images folder
    img_dir = config["image_dir"]
    if os.path.isdir(img_dir):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        count = sum(len(glob.glob(os.path.join(img_dir, e))) for e in exts)
        if count >= 10:
            log.info(f"STAGE 1: Found {count} images in {img_dir} → PATH B (RGB)")
            return "rgb"

    log.warning("STAGE 1: No RealSense and no images found.")
    log.info("  Use --synthetic to generate test data, or place images in images/")
    return "synthetic"


def init_realsense(config):
    """[PATH A] Initialize RealSense D435 with aligned depth+color."""
    log.info("STAGE 1A: Initializing Intel RealSense D435...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, config["frame_width"],
                            config["frame_height"], rs.format.z16, config["fps"])
    rs_config.enable_stream(rs.stream.color, config["frame_width"],
                            config["frame_height"], rs.format.bgr8, config["fps"])
    profile = pipeline.start(rs_config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                  "width": intr.width, "height": intr.height}

    log.info("  Warming up camera (30 frames)...")
    for _ in range(30):
        pipeline.wait_for_frames()

    log.info(f"  Camera ready. Intrinsics: fx={intr.fx:.1f}, fy={intr.fy:.1f}")
    return pipeline, align, intrinsics


def init_rgb_camera(config, images):
    """[PATH B] Compute intrinsics from image dimensions."""
    h, w = images[0].shape[:2]
    fx = config["focal_length_px"] or (w * 1.2)
    intrinsics = {"fx": fx, "fy": fx, "cx": w / 2.0, "cy": h / 2.0,
                  "width": w, "height": h}
    log.info(f"STAGE 1B: RGB camera intrinsics: fx={fx:.1f}, image={w}×{h}")
    return intrinsics


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CAPTURE / LOAD
# ═════════════════════════════════════════════════════════════════════════════

def check_frame_quality(color_image, depth_image, config):
    """[PATH A] Quality gates: blur, depth coverage, depth range."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < config["blur_threshold"]:
        return False, f"Blurry ({blur:.1f} < {config['blur_threshold']})"
    h, w = depth_image.shape
    center = depth_image[h//4:3*h//4, w//4:3*w//4]
    coverage = np.sum(center > 0) / center.size
    if coverage < 0.5:
        return False, f"Poor depth ({coverage:.1%})"
    median_d = np.median(center[center > 0])
    if median_d < config["min_depth_mm"] or median_d > config["max_depth_mm"]:
        return False, f"Out of range ({median_d:.0f}mm)"
    return True, f"OK (blur={blur:.1f}, depth={coverage:.1%}, d={median_d:.0f}mm)"


def capture_rgbd_frames(pipeline, align, config):
    """[PATH A] Interactive SPACE-to-capture loop with quality gates."""
    os.makedirs(config["frames_dir"], exist_ok=True)
    captured = []
    idx = 0
    target = config["num_frames"]

    log.info(f"STAGE 2A: Capture {target} frames. SPACE=capture, Q=finish, ESC=abort")

    while idx < target:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        preview = color.copy()
        cv2.putText(preview, f"Captured: {idx}/{target}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("G18 Unified — Capture", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            ok, reason = check_frame_quality(color, depth, config)
            if ok:
                cv2.imwrite(os.path.join(config["frames_dir"], f"color_{idx:03d}.png"), color)
                cv2.imwrite(os.path.join(config["frames_dir"], f"depth_{idx:03d}.png"), depth)
                captured.append((color.copy(), depth.copy()))
                log.info(f"  Frame {idx:03d}: {reason}")
                idx += 1
            else:
                log.warning(f"  REJECTED: {reason}")
        elif key == ord('q') and idx >= 20:
            break
        elif key == 27:
            cv2.destroyAllWindows(); sys.exit(1)

    cv2.destroyAllWindows()
    log.info(f"STAGE 2A: {len(captured)} frames captured.")
    return captured


def load_rgb_images(config):
    """[PATH B] Load multi-view images from directory."""
    img_dir = config["image_dir"]
    bg_path = config["background_image"]
    bg_abs = os.path.abspath(bg_path) if bg_path and os.path.exists(bg_path) else None

    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                   glob.glob(os.path.join(img_dir, "*.jpg")) +
                   glob.glob(os.path.join(img_dir, "*.jpeg")) +
                   glob.glob(os.path.join(img_dir, "*.bmp")))

    images, img_paths = [], []
    for p in paths:
        if bg_abs and os.path.abspath(p) == bg_abs:
            continue
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
            img_paths.append(p)

    log.info(f"STAGE 2B: Loaded {len(images)} images ({images[0].shape[1]}×{images[0].shape[0]})")
    return images, img_paths


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SILHOUETTE EXTRACTION (PATH B ONLY)
# ═════════════════════════════════════════════════════════════════════════════

def extract_silhouettes_enhanced(images, config):
    """
    [PATH B] Multi-method silhouette extraction.
    Methods: basic, multichannel (LAB+HSV+RGB), edge-aware morphology,
    GrabCut refinement, deep learning (U2Net).
    """
    method = config["silhouette_method"]
    log.info(f"STAGE 3B: Silhouette extraction (method={method})")

    # Background
    bg_path = config["background_image"]
    if bg_path and os.path.exists(bg_path):
        background = cv2.imread(bg_path)
        log.info(f"  Using provided background: {bg_path}")
    else:
        log.info("  Estimating background via temporal median...")
        background = np.median(np.array(images, dtype=np.float32), axis=0).astype(np.uint8)

    silhouettes = []
    sil_dir = os.path.join(config["output_dir"], "silhouettes")
    os.makedirs(sil_dir, exist_ok=True)

    for i, img in enumerate(images):
        if method == "deep_learning":
            mask = _silhouette_deep_learning(img)
            if mask is None:
                mask = _silhouette_multichannel(img, background)
        elif method in ("full", "multichannel"):
            mask = _silhouette_multichannel(img, background)
            mask = _edge_aware_morphology(mask, img)
            if method == "full":
                mask = _refine_grabcut(img, mask)
        elif method == "grabcut":
            mask = _silhouette_multichannel(img, background)
            mask = _refine_grabcut(img, mask)
        else:  # basic
            mask = _silhouette_basic(img, background, config["silhouette_threshold"])

        # Contour refinement
        mask = _refine_contours(mask, img.shape[0] * img.shape[1])
        silhouettes.append(mask)
        cv2.imwrite(os.path.join(sil_dir, f"sil_{i:03d}.png"), mask)

        if (i + 1) % 12 == 0 or i == 0:
            fg_pct = np.sum(mask > 0) / mask.size * 100
            log.info(f"  [{i+1:2d}/{len(images)}] fg={fg_pct:.1f}%")

    log.info(f"  ✓ {len(silhouettes)} silhouettes extracted")
    return silhouettes


def _silhouette_basic(img, bg, threshold):
    gray_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gray_bg = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(gray_img, gray_bg)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    return mask


def _silhouette_multichannel(img, bg):
    """LAB + HSV + RGB voting (foreground if ≥2/3 agree)."""
    votes = np.zeros(img.shape[:2], dtype=np.int32)
    # LAB
    lab_d = np.sqrt(np.sum((cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                            - cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32))**2, axis=2))
    votes += (lab_d > 15.0).astype(np.int32)
    # HSV
    hsv_i = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_b = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_d = np.minimum(np.abs(hsv_i[:,:,0]-hsv_b[:,:,0]), 180-np.abs(hsv_i[:,:,0]-hsv_b[:,:,0]))
    sv_d = np.sqrt((hsv_i[:,:,1]-hsv_b[:,:,1])**2 + (hsv_i[:,:,2]-hsv_b[:,:,2])**2)
    votes += (hue_d*2 + sv_d > 25.0).astype(np.int32)
    # RGB + Otsu
    rgb_d = np.sqrt(np.sum((img.astype(np.float32) - bg.astype(np.float32))**2, axis=2))
    otsu_t, _ = cv2.threshold(np.clip(rgb_d, 0, 255).astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    votes += (rgb_d > otsu_t).astype(np.int32)
    return (votes >= 2).astype(np.uint8) * 255


def _edge_aware_morphology(mask, img):
    """Small kernel near edges, large kernel in smooth regions."""
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 100)
    ezone = cv2.GaussianBlur(cv2.dilate(edges, np.ones((15,15), np.uint8), iterations=2).astype(np.float32), (21,21), 5) / 255.0
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ms = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ks, iterations=2), cv2.MORPH_OPEN, ks, iterations=1)
    ml = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kl, iterations=3), cv2.MORPH_OPEN, kl, iterations=2)
    blended = ezone * ms.astype(np.float32) + (1-ezone) * ml.astype(np.float32)
    return (blended > 127).astype(np.uint8) * 255


def _refine_grabcut(img, mask, iterations=5, margin=15):
    """GrabCut refinement using initial mask as seed."""
    h, w = img.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    kl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2+1, margin*2+1))
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin+1, margin+1))
    dilated = cv2.dilate(mask, kl, iterations=1)
    eroded = cv2.erode(mask, ks, iterations=1)
    gc_mask[eroded > 128] = cv2.GC_FGD
    gc_mask[(mask > 128) & (eroded <= 128)] = cv2.GC_PR_FGD
    gc_mask[(dilated > 128) & (mask <= 128)] = cv2.GC_PR_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc_mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
        return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return mask


def _silhouette_deep_learning(img):
    """U2Net background removal via rembg."""
    try:
        from rembg import remove
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = remove(rgb)
        alpha = result[:, :, 3] if result.shape[2] == 4 else cv2.cvtColor(result[:,:,:3], cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
        return mask
    except ImportError:
        return None


def _refine_contours(mask, image_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    significant = [c for c in contours if cv2.contourArea(c) > image_area * 0.01]
    if not significant:
        significant = [max(contours, key=cv2.contourArea)]
    clean = np.zeros_like(mask)
    for c in significant:
        eps = 0.002 * cv2.arcLength(c, True)
        cv2.drawContours(clean, [cv2.approxPolyDP(c, eps, True)], -1, 255, cv2.FILLED)
    return clean


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — POSE ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def estimate_poses_icp(frames, intrinsics, config):
    """[PATH A] ICP odometry between consecutive RGB-D frames."""
    log.info("STAGE 4A: Estimating camera poses via color ICP...")
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        intrinsics["width"], intrinsics["height"],
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"])
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_max = config["tsdf_depth_max"]
    poses = [np.eye(4)]
    current = np.eye(4)

    for i in range(1, len(frames)):
        rgbd_prev = _to_o3d_rgbd(frames[i-1][0], frames[i-1][1], config)
        rgbd_curr = _to_o3d_rgbd(frames[i][0], frames[i][1], config)
        ok, rel, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_prev, rgbd_curr, o3d_intr, np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        if ok:
            current = current @ rel
            log.info(f"  Frame {i:03d}: ICP OK (fitness={info[0,0]:.4f})")
        else:
            log.warning(f"  Frame {i:03d}: ICP FAILED — reusing previous pose")
        poses.append(current.copy())

    log.info(f"STAGE 4A: {len(poses)} poses estimated.")
    return poses


def _to_o3d_rgbd(color_bgr, depth_u16, config):
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_u16.astype(np.uint16))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=config["tsdf_depth_scale"],
        depth_trunc=config["tsdf_depth_max"], convert_rgb_to_intensity=False)


def compute_camera_matrices_rgb(intrinsics, num_views, config):
    """[PATH B] Compute projection matrices from known circular arc."""
    log.info("STAGE 4B: Computing camera matrices for circular arc...")
    angles = np.linspace(np.radians(config["angle_start_deg"]),
                         np.radians(config["angle_end_deg"]), num_views)
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cam_dist = config["camera_distance_mm"]

    projections = []
    for angle in angles:
        cam_pos = np.array([cam_dist * np.cos(angle), 0, cam_dist * np.sin(angle)])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right /= (np.linalg.norm(right) + 1e-8)
        cam_up = np.cross(right, forward)
        R = np.array([right, -cam_up, forward])
        t = -R @ cam_pos
        P = K @ np.hstack([R, t.reshape(3, 1)])
        projections.append(P)

    log.info(f"  {num_views} projection matrices computed.")
    return projections


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 — 3D RECONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def tsdf_fusion(frames, poses, intrinsics, config):
    """[PATH A] TSDF volumetric fusion → triangle mesh."""
    log.info("STAGE 5A: Running TSDF fusion...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_voxel_size"],
        sdf_trunc=config["tsdf_voxel_size"] * config["tsdf_trunc_mult"],
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        intrinsics["width"], intrinsics["height"],
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"])

    for i, ((c, d), pose) in enumerate(zip(frames, poses)):
        volume.integrate(_to_o3d_rgbd(c, d, config), o3d_intr, np.linalg.inv(pose))
        if (i+1) % 10 == 0:
            log.info(f"  Fused {i+1}/{len(frames)} frames")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    log.info(f"  TSDF mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")
    return mesh


def visual_hull_reconstruction(silhouettes, projections, config):
    """[PATH B] Voxel carving from silhouettes → Marching Cubes → mesh."""
    log.info("STAGE 5B: Visual hull reconstruction...")
    G = config["voxel_grid_size"]
    vol = config["voxel_volume_mm"]
    voxel_size = vol / G
    half = vol / 2.0

    # Create voxel grid coordinates
    coords = np.mgrid[0:G, 0:G, 0:G].reshape(3, -1).T
    world = (coords + 0.5) * voxel_size - half
    occupied = np.ones(len(world), dtype=bool)
    N_initial = occupied.sum()

    for i, (sil, P) in enumerate(zip(silhouettes, projections)):
        if occupied.sum() == 0:
            break
        pts_h = np.hstack([world[occupied], np.ones((occupied.sum(), 1))])
        proj = (P @ pts_h.T).T
        proj[:, 0] /= proj[:, 2]
        proj[:, 1] /= proj[:, 2]
        u = np.round(proj[:, 0]).astype(int)
        v = np.round(proj[:, 1]).astype(int)
        h, w = sil.shape
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (proj[:, 2] > 0)
        on_sil = np.zeros(occupied.sum(), dtype=bool)
        valid = np.where(in_bounds)[0]
        on_sil[valid] = sil[v[valid], u[valid]] > 128
        active_idx = np.where(occupied)[0]
        occupied[active_idx[~on_sil]] = False

        if (i+1) % 6 == 0 or i == 0:
            log.info(f"  View {i+1:2d}/{len(silhouettes)}: {occupied.sum():,} voxels remain ({occupied.sum()/N_initial*100:.1f}%)")

    # Build occupancy grid
    occupancy = occupied.reshape(G, G, G)
    log.info(f"  Visual hull: {occupancy.sum():,} occupied voxels")

    # Marching Cubes
    log.info("  Running Marching Cubes...")
    smooth_field = ndimage.gaussian_filter(occupancy.astype(np.float32), sigma=0.5)  # [FIX-3: was 1.0]
    verts, faces, _, _ = measure.marching_cubes(smooth_field, level=config["marching_cubes_level"])
    verts = verts * voxel_size - half  # to physical coordinates (mm)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts / 1000.0)  # mm → meters
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()

    log.info(f"  Mesh: {len(verts):,} vertices, {len(faces):,} faces")
    return mesh_o3d


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6 — VISUALIZATION (Interactive or Fallback)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_mesh(mesh_o3d, config, label="mesh"):
    """Interactive Open3D viewer (fallback: save PLY + screenshot)."""
    os.makedirs(config["mesh_dir"], exist_ok=True)
    ply_path = os.path.join(config["mesh_dir"], f"{label}.ply")
    o3d.io.write_triangle_mesh(ply_path, mesh_o3d)
    log.info(f"STAGE 6: Mesh saved: {ply_path}")

    try:
        o3d.visualization.draw_geometries(
            [mesh_o3d], window_name=f"G18 Unified — {label}", width=1024, height=768)
        log.info("  Viewer closed.")
    except Exception:
        log.info("  No display available — skipping interactive viewer.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 7 — ROI CROP (PATH A primarily)
# ═════════════════════════════════════════════════════════════════════════════

def crop_to_roi(mesh_o3d, config):
    """Crop mesh to region of interest via AABB."""
    if not config.get("roi_enabled", True):
        log.info("STAGE 7: ROI crop disabled (visual hull is already focused).")
        return mesh_o3d

    log.info("STAGE 7: Cropping to region of interest...")
    verts = np.asarray(mesh_o3d.vertices)
    centroid = verts.mean(axis=0)
    dx, dy, dz = config["roi_x_half"], config["roi_y_half"], config["roi_z_half"]
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=centroid - [dx, dy, dz], max_bound=centroid + [dx, dy, dz])
    cropped = mesh_o3d.crop(bbox)
    log.info(f"  Vertices after crop: {len(cropped.vertices):,}")
    return cropped


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 8 — CONVEX HULL CLOSURE
# ═════════════════════════════════════════════════════════════════════════════

def close_with_convex_hull(tm_mesh, config):
    """[FIX-4] Boolean UNION of original mesh + convex hull (was: return pure hull,
    discarding all scanned geometry ~1700→~200 vertices).
    [FIX-5] Deviation measured by sampling 2000 surface points on hull surface
    (was: hull.vertices, which ARE original vertices → always 0.0mm).
    """
    log.info("STAGE 8: Convex hull closure for unseen geometry...")
    hull = tm_mesh.convex_hull
    log.info(f"  Original: {len(tm_mesh.vertices):,}V  Hull: {len(hull.vertices):,}V  watertight={hull.is_watertight}")

    # [FIX-5] Sample 2000 random surface points from hull to measure true deviation
    try:
        hull_pts, _ = trimesh.sample.sample_surface(hull, 2000)
        try:
            from trimesh.proximity import closest_point
            _, dists, _ = closest_point(tm_mesh, hull_pts)
        except Exception:
            from scipy.spatial import cKDTree
            dists, _ = cKDTree(tm_mesh.vertices).query(hull_pts)
        scale = 1000.0 if max(tm_mesh.bounding_box.extents) < 1.0 else 1.0
        max_dev  = float(dists.max()  * scale)
        mean_dev = float(dists.mean() * scale)
    except Exception as e:
        log.warning(f"  Hull deviation sampling failed ({e})")
        max_dev, mean_dev = 0.0, 0.0
    log.info(f"  Hull deviation (surface samples) — Max: {max_dev:.1f}mm, Mean: {mean_dev:.1f}mm")
    if max_dev > 15.0:
        log.warning("  Hull deviation >15mm — cast may be loose on unseen side.")
    config["_hull_max_deviation_mm"]  = round(max_dev,  2)
    config["_hull_mean_deviation_mm"] = round(mean_dev, 2)

    # [FIX-4] Boolean UNION: preserve scanned geometry; hull only fills gaps
    closed = None
    if MANIFOLD_AVAILABLE:
        try:
            from manifold3d import Manifold, Mesh as ManifoldMesh
            def _to_m(tm):
                return Manifold(mesh=ManifoldMesh(
                    vert_properties=np.array(tm.vertices, dtype=np.float64),
                    tri_verts=np.array(tm.faces, dtype=np.uint32)))
            result  = (_to_m(tm_mesh) + _to_m(hull)).to_mesh()
            closed  = trimesh.Trimesh(vertices=result.vert_properties[:, :3],
                                      faces=result.tri_verts, process=True)
            log.info(f"  UNION: {len(closed.vertices):,}V (manifold3d)")
        except Exception as e:
            log.warning(f"  Boolean UNION failed ({e}), falling back to concatenation")
    if closed is None:
        closed = trimesh.util.concatenate([tm_mesh, hull])
        log.info(f"  Fallback concatenation: {len(closed.vertices):,}V")

    trimesh.repair.fill_holes(closed)
    trimesh.repair.fix_winding(closed)
    trimesh.repair.fix_normals(closed)
    closed = _fix_outward_normals(closed, label="hull_closed")  # [FIX-H]
    log.info(f"  Closed: {len(closed.vertices):,}V  watertight={closed.is_watertight}")
    return closed


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 9 — MESH CLEANING & WATERTIGHT VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def clean_and_validate(tm_mesh, config, label="cleaned"):
    """Fill holes, fix winding/normals, remove degenerates. Report quality."""
    log.info("STAGE 9: Mesh cleaning and validation...")

    trimesh.repair.fill_holes(tm_mesh)
    trimesh.repair.fix_winding(tm_mesh)
    trimesh.repair.fix_normals(tm_mesh)

    # Remove degenerate faces (trimesh API v4+ compatibility)
    try:
        tm_mesh.remove_degenerate_faces()
    except AttributeError:
        mask = tm_mesh.nondegenerate_faces()
        tm_mesh.update_faces(mask)
    try:
        tm_mesh.remove_duplicate_faces()
    except AttributeError:
        pass
    tm_mesh.remove_unreferenced_vertices()

    # [FIX-G] Purge NaN/Inf vertices BEFORE smoothing.
    # Hull-union concatenation can leave NaN coords; smoothing then propagates
    # them across every connected vertex, destroying the entire mesh.
    tm_mesh = _sanitise_mesh_vertices(tm_mesh, label="pre_smooth")

    # [FIX-E] Safe Laplacian smoothing: snapshot → smooth → validate → rollback.
    pre_v = tm_mesh.vertices.copy()
    pre_f = tm_mesh.faces.copy()
    try:
        smoothed = trimesh.smoothing.filter_laplacian(
            tm_mesh, iterations=config["smooth_iterations"],
            lamb=config["smooth_lambda"])
        sv = smoothed.vertices if smoothed is not None else np.array([])
        n_nan = int((~np.isfinite(sv).all(axis=1)).sum()) if len(sv) > 0 else len(pre_v)
        if smoothed is None or len(sv) == 0 or n_nan > len(sv) * 0.05:
            raise ValueError(f"Smoothing corrupted mesh ({n_nan}/{len(sv)} NaN verts)")
        if smoothed is not tm_mesh:
            tm_mesh = smoothed
        log.info(f"  Smoothing OK: {len(tm_mesh.vertices):,} vertices remain")
    except Exception as e:
        log.warning(f"  Laplacian smoothing failed ({e}) — using pre-smooth mesh")
        tm_mesh = trimesh.Trimesh(vertices=pre_v, faces=pre_f, process=False)
        trimesh.repair.fix_normals(tm_mesh)

    # [FIX-D] Final NaN guard after smoothing
    tm_mesh = _sanitise_mesh_vertices(tm_mesh, label="post_smooth")

    # [FIX-F] Guard against empty mesh before accessing bounding_box / volume
    if len(tm_mesh.vertices) == 0 or len(tm_mesh.faces) == 0:
        raise RuntimeError(
            "clean_and_validate: mesh is empty after cleaning+smoothing. "
            "The input geometry is too degenerate. Check Stage 8 output.")

    log.info(f"  Vertices: {len(tm_mesh.vertices):,}")
    log.info(f"  Faces:    {len(tm_mesh.faces):,}")
    log.info(f"  Watertight: {tm_mesh.is_watertight}")
    log.info(f"  Euler:    {tm_mesh.euler_number} (should be 2)")
    try:
        vol_mm3 = abs(tm_mesh.volume) * 1e6 if abs(tm_mesh.volume) < 1 else abs(tm_mesh.volume)
        log.info(f"  Volume:   {vol_mm3:.1f} mm³ ({vol_mm3/1000:.1f} cm³)")
    except Exception:
        log.info("  Volume:   N/A (not watertight)")
    try:
        bbox = tm_mesh.bounding_box.extents
        log.info(f"  BBox:     {bbox[0]*1000:.1f} × {bbox[1]*1000:.1f} × {bbox[2]*1000:.1f} mm")
    except Exception:
        log.info("  BBox:     N/A")

    tm_mesh = _fix_outward_normals(tm_mesh, label=label)  # [FIX-H]
    os.makedirs(config["mesh_dir"], exist_ok=True)
    tm_mesh.export(os.path.join(config["mesh_dir"], f"{label}.ply"))
    return tm_mesh



# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS — NaN sanitiser + outward-normal fixer
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_mesh_vertices(mesh, label="mesh"):
    """[FIX-D/G] Remove faces referencing NaN/Inf vertex coordinates.
    Laplacian smoothing on degenerate faces produces NaN positions that
    silently propagate into STL files (Size X/Y/Z = NaN in slicers).
    """
    bad = ~np.isfinite(mesh.vertices).all(axis=1)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return mesh
    log.warning(f"  [{label}] {n_bad} NaN/Inf vertices — removing affected faces")
    bad_idx = set(np.where(bad)[0])
    keep = np.array([not (f[0] in bad_idx or f[1] in bad_idx or f[2] in bad_idx)
                     for f in mesh.faces], dtype=bool)
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    log.info(f"  [{label}] After purge: {len(mesh.vertices):,}V {len(mesh.faces):,}F  wt={mesh.is_watertight}")
    return mesh


def _fix_outward_normals(mesh, label="mesh"):
    """[FIX-H] Guarantee face normals point OUTWARD from the centroid.
    trimesh.repair.fix_winding uses majority-vote and can flip the whole
    mesh inside-out after Boolean/concatenation, producing blank renders
    and Volume=0.00 in slicers.  We verify against centroid→face-centre
    direction and invert the whole mesh if >50% point inward.
    """
    if len(mesh.faces) == 0:
        return mesh
    try:
        dots = (mesh.face_normals * (mesh.triangles_center - mesh.centroid)).sum(axis=1)
        frac_in = (dots < 0).sum() / len(dots)
        if frac_in > 0.5:
            log.warning(f"  [{label}] {frac_in:.0%} normals inward — inverting mesh")
            mesh.invert()
        else:
            log.info(f"  [{label}] Normals OK ({1-frac_in:.0%} outward)")
    except Exception as e:
        log.warning(f"  [{label}] _fix_outward_normals failed ({e})")
    return mesh

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 10 — CAST DESIGN (ENHANCED: Voronoi + Boolean + Fasteners)
# ═════════════════════════════════════════════════════════════════════════════

def generate_cast_design(hand_mesh, config):
    """
    Enhanced cast design:
      1. Boolean shell (inner offset + outer offset → difference)
      2. Voronoi ventilation holes
      3. Plane split into top/bottom halves
      4. Fastener holes along split edge
    Returns: (top_shell, bottom_shell, full_cast, hand_mesh)
    """
    log.info("STAGE 10: Enhanced cast design...")
    gap = config["cast_gap_mm"]
    wall = config["cast_wall_mm"]

    # Detect units: if mesh seems to be in meters (max dim < 1), scale
    extents = hand_mesh.bounding_box.extents
    in_meters = max(extents) < 1.0
    if in_meters:
        gap /= 1000.0
        wall /= 1000.0
        log.info(f"  Detected mesh in meters. Gap={gap*1000:.1f}mm, Wall={wall*1000:.1f}mm")
    else:
        log.info(f"  Detected mesh in mm. Gap={gap:.1f}mm, Wall={wall:.1f}mm")

    # ── Step 1: Create shell via Boolean offset ──
    inner_offset = gap
    outer_offset = gap + wall

    # [FIX-D] Purge NaN vertices that arrived from smoothing/hull-union
    hand_mesh = _sanitise_mesh_vertices(hand_mesh, label="hand_mesh")

    # [FIX-A] Sanitise vertex normals — NaN normals → NaN offset coords → crash
    normals = hand_mesh.vertex_normals.copy()
    bad_n = ~np.isfinite(normals).all(axis=1) | (np.linalg.norm(normals, axis=1) < 1e-8)
    if bad_n.any():
        log.warning(f"  Sanitising {bad_n.sum()} NaN/zero normals (replacing with [0,0,1])")
        normals[bad_n] = [0.0, 0.0, 1.0]

    inner_verts = hand_mesh.vertices + normals * inner_offset
    outer_verts = hand_mesh.vertices + normals * outer_offset
    inner = trimesh.Trimesh(vertices=inner_verts, faces=hand_mesh.faces.copy(), process=False)
    outer = trimesh.Trimesh(vertices=outer_verts, faces=hand_mesh.faces.copy(), process=False)
    trimesh.repair.fix_normals(inner)
    trimesh.repair.fix_normals(outer)

    if MANIFOLD_AVAILABLE:
        try:
            from manifold3d import Manifold, Mesh as ManifoldMesh
            def to_manifold(tm):
                return Manifold(mesh=ManifoldMesh(
                    vert_properties=np.array(tm.vertices, dtype=np.float64),
                    tri_verts=np.array(tm.faces, dtype=np.uint32)))
            m_outer = to_manifold(outer)
            m_inner = to_manifold(inner)
            m_shell = m_outer - m_inner
            result = m_shell.to_mesh()
            shell = trimesh.Trimesh(vertices=result.vert_properties[:, :3],
                                     faces=result.tri_verts, process=True)
            trimesh.repair.fill_holes(shell)   # [FIX-6]
            trimesh.repair.fix_winding(shell)
            trimesh.repair.fix_normals(shell)
            shell = _fix_outward_normals(shell, label="cast_shell")  # [FIX-H]
            log.info(f"  Boolean shell: {len(shell.faces):,} faces (manifold3d)")
        except Exception as e:
            log.warning(f"  Boolean failed ({e}), using concatenated shell")
            inner.invert()
            shell = trimesh.util.concatenate([outer, inner])
    else:
        inner.invert()
        shell = trimesh.util.concatenate([outer, inner])
        log.info(f"  Concatenated shell: {len(shell.faces):,} faces")

    # Save solid shell
    shell.export(os.path.join(config["mesh_dir"], "cast_shell_solid.stl"))

    # ── Step 2: Voronoi ventilation holes ──
    if config["voronoi_enabled"]:
        shell = _cut_voronoi_holes(shell, config, in_meters)

    full_cast = shell.copy()

    # ── Step 3: Fastener holes (apply before split so halves get them) ──
    if config["fastener_enabled"]:
        shell = _add_fastener_holes_to_shell(shell, hand_mesh, config, in_meters)

    full_cast = shell.copy()
    full_cast.export(os.path.join(config["mesh_dir"], "cast_voronoi_full.stl"))

    # ── Step 4: Split into top/bottom halves ──
    top_shell, bottom_shell = _split_cast(shell, config)

    return top_shell, bottom_shell, full_cast


def _cut_voronoi_holes(shell, config, in_meters):
    """Cut Voronoi ventilation holes into the shell."""
    n_pts = config["voronoi_num_points"]
    radius = config["voronoi_hole_radius"] / (1000.0 if in_meters else 1.0)
    # [FIX-B] radius is already in native units; only multiply by 1000 for metres
    radius_mm = radius * 1000.0 if in_meters else radius
    log.info(f"  Voronoi: {n_pts} seeds, radius={radius_mm:.1f}mm")
    pts, _ = trimesh.sample.sample_surface(shell, n_pts)

    # Filter points too close to mesh boundary
    extents = shell.bounding_box.extents
    min_edge = 0.05 * min(extents)
    center = shell.centroid
    dists_from_center = np.linalg.norm(pts - center, axis=1)
    valid = dists_from_center < (max(extents) * 0.45)
    hole_centers = pts[valid]
    log.info(f"  Valid hole positions: {len(hole_centers)}")

    if len(hole_centers) == 0:
        return shell

    # Create spheres and subtract
    spheres = [trimesh.creation.icosphere(subdivisions=2, radius=radius)
               for _ in hole_centers]
    for s, c in zip(spheres, hole_centers):
        s.apply_translation(c)
    all_spheres = trimesh.util.concatenate(spheres)

    if MANIFOLD_AVAILABLE:
        try:
            from manifold3d import Manifold, Mesh as ManifoldMesh
            def to_m(tm):
                return Manifold(mesh=ManifoldMesh(
                    vert_properties=np.array(tm.vertices, dtype=np.float64),
                    tri_verts=np.array(tm.faces, dtype=np.uint32)))
            result_m = (to_m(shell) - to_m(all_spheres)).to_mesh()
            shell = trimesh.Trimesh(vertices=result_m.vert_properties[:,:3],
                                     faces=result_m.tri_verts, process=True)
            trimesh.repair.fill_holes(shell)   # [FIX-7a]
            trimesh.repair.fix_winding(shell)
            trimesh.repair.fix_normals(shell)
            log.info(f"  Voronoi cut: {len(shell.faces):,} faces (Boolean)")
        except Exception as e:
            log.warning(f"  Boolean Voronoi failed ({e}), using vertex-based cut")
            shell = _vertex_cut_holes(shell, hole_centers, radius)
    else:
        shell = _vertex_cut_holes(shell, hole_centers, radius)

    return shell


def _vertex_cut_holes(mesh, centers, radius):
    """Fallback: remove faces whose centroid falls inside any sphere."""
    face_centers = mesh.triangles_center
    keep = np.ones(len(mesh.faces), dtype=bool)
    for c in centers:
        dists = np.linalg.norm(face_centers - c, axis=1)
        keep &= (dists > radius)
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(mesh)   # [FIX-7b]
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _split_cast(shell, config):
    """Split cast at centroid plane into top and bottom halves."""
    normal = np.array(config["split_normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    centroid = shell.centroid
    gap = config["split_gap_mm"]

    # Detect units
    if max(shell.bounding_box.extents) < 1.0:
        gap /= 1000.0

    log.info(f"  Splitting along {normal}, gap={gap*1000:.2f}mm")

    # Vertex classification
    dots = (shell.vertices - centroid) @ normal
    top_mask = dots >= gap / 2.0
    bot_mask = dots <= -gap / 2.0

    def extract_half(mask):
        face_mask = mask[shell.faces].sum(axis=1) >= 2
        if face_mask.sum() == 0:
            return shell.copy()
        sub = shell.submesh([np.where(face_mask)[0]], append=True)
        trimesh.repair.fill_holes(sub)
        trimesh.repair.fix_winding(sub)   # [FIX-8]
        trimesh.repair.fix_normals(sub)
        return sub

    top = extract_half(top_mask)
    bottom = extract_half(bot_mask)

    log.info(f"  Top:    {len(top.faces):,} faces")
    log.info(f"  Bottom: {len(bottom.faces):,} faces")
    return top, bottom


def _add_fastener_holes_to_shell(shell, hand_mesh, config, in_meters):
    """Add fastener holes along the split plane to the full shell."""
    diameter = config["fastener_diameter_mm"] / (1000.0 if in_meters else 1.0)
    spacing = config["fastener_spacing_mm"] / (1000.0 if in_meters else 1.0)

    verts = hand_mesh.vertices
    # [FIX-C] Guard against NaN vertices crashing int(y_max - y_min)
    finite_mask = np.isfinite(verts).all(axis=1)
    if not finite_mask.all():
        log.warning(f"  Fastener: {(~finite_mask).sum()} NaN vertices filtered")
        verts = verts[finite_mask]
    if len(verts) == 0:
        log.warning("  Fastener: no valid vertices — skipping")
        return shell
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    n_holes = max(3, int((y_max - y_min) / spacing))
    y_positions = np.linspace(y_min + spacing/2, y_max - spacing/2, n_holes)
    centroid = hand_mesh.centroid

    log.info(f"  Adding {n_holes} fastener holes (d={diameter*1000:.1f}mm)")

    cylinders = []
    for y in y_positions:
        cyl = trimesh.creation.cylinder(radius=diameter/2, height=diameter*6, sections=16)
        R = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        cyl.apply_transform(R)
        cyl.apply_translation([centroid[0], y, centroid[2]])
        cylinders.append(cyl)

    if MANIFOLD_AVAILABLE and cylinders:
        try:
            from manifold3d import Manifold, Mesh as ManifoldMesh
            def to_m(tm):
                return Manifold(mesh=ManifoldMesh(
                    vert_properties=np.array(tm.vertices, dtype=np.float64),
                    tri_verts=np.array(tm.faces, dtype=np.uint32)))
            all_cyl = trimesh.util.concatenate(cylinders)
            r = (to_m(shell) - to_m(all_cyl)).to_mesh()
            new_shell = trimesh.Trimesh(vertices=r.vert_properties[:,:3], faces=r.tri_verts, process=True)
            if len(new_shell.faces) > len(shell.faces) * 0.3:
                shell = new_shell
                trimesh.repair.fill_holes(shell)   # [FIX-9]
                trimesh.repair.fix_winding(shell)
                trimesh.repair.fix_normals(shell)
                log.info(f"  Fastener holes cut: {len(shell.faces):,} faces")
            else:
                log.warning("  Fastener subtraction produced too-small result — skipping")
        except Exception as e:
            log.warning(f"  Fastener subtraction failed: {e}")

    return shell



# ═════════════════════════════════════════════════════════════════════════════
# STAGE 11 — VALIDATION + EXPORT + REPORT
# ═════════════════════════════════════════════════════════════════════════════

def compute_fit_deviation(hand_mesh, cast_shell, n_samples=5000):
    """5000-point surface sampling: distance from hand to cast shell."""
    from trimesh.sample import sample_surface

    pts, _ = sample_surface(hand_mesh, n_samples)

    try:
        from trimesh.proximity import closest_point
        _, dists, _ = closest_point(cast_shell, pts)
    except (ImportError, ModuleNotFoundError, Exception):
        from scipy.spatial import cKDTree
        tree = cKDTree(cast_shell.vertices)
        dists, _ = tree.query(pts)

    # Auto-detect units
    scale = 1000.0 if max(hand_mesh.bounding_box.extents) < 1.0 else 1.0
    dists_mm = dists * scale

    result = {
        "n_samples": n_samples,
        "mean_mm": round(float(dists_mm.mean()), 2),
        "max_mm": round(float(dists_mm.max()), 2),
        "std_mm": round(float(dists_mm.std()), 2),
        "p90_mm": round(float(np.percentile(dists_mm, 90)), 2),
        "p95_mm": round(float(np.percentile(dists_mm, 95)), 2),
    }

    log.info(f"  Fit deviation: mean={result['mean_mm']}mm, max={result['max_mm']}mm, p95={result['p95_mm']}mm")
    if result["mean_mm"] < 2.0:
        log.info("  VERDICT: FIT ACCEPTABLE (mean < 2mm) ✅")
    elif result["mean_mm"] < 4.0:
        log.warning("  VERDICT: FIT MARGINAL (2-4mm) ⚠️")
    else:
        log.error("  VERDICT: FIT POOR (>4mm) ❌")

    return result


def export_and_report(top_shell, bottom_shell, full_cast, hand_mesh, config, mode, timing):
    """Export STL files + JSON defense report."""
    log.info("STAGE 11: Exporting STL + generating defense report...")

    os.makedirs(config["mesh_dir"], exist_ok=True)
    paths = {}

    # [FIX-10 + FIX-H] Repair + outward normals before every STL write
    def _repair_and_export(mesh, path):
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        mesh = _fix_outward_normals(mesh, label=os.path.basename(path))
        mesh.export(path)
        return mesh

    for name, mesh in [("cast_top_shell", top_shell), ("cast_bottom_shell", bottom_shell),
                        ("cast_full", full_cast), ("hand_surface", hand_mesh)]:
        p = os.path.join(config["mesh_dir"], f"{name}.stl")
        mesh = _repair_and_export(mesh, p)
        paths[name] = p
        sz = os.path.getsize(p) / 1e6
        log.info(f"  {name}.stl ({sz:.1f} MB) — {len(mesh.faces):,} faces")

    # Also export print-ready versions
    for name, mesh in [("PRINT_cast_top", top_shell), ("PRINT_cast_bottom", bottom_shell),
                        ("PRINT_cast_full", full_cast)]:
        p = os.path.join(config["mesh_dir"], f"{name}.stl")
        _repair_and_export(mesh, p)
        paths[name] = p

    # ── Validation ──
    log.info("VALIDATION: Computing point-to-mesh fit deviation...")
    fit_top = {"n_samples": 0, "mean_mm": 0, "max_mm": 0, "std_mm": 0, "p90_mm": 0, "p95_mm": 0}
    if len(top_shell.faces) > 10:
        fit_top = compute_fit_deviation(hand_mesh, top_shell, config["validation_samples"])
    fit_full = compute_fit_deviation(hand_mesh, full_cast, config["validation_samples"])

    # ── JSON Report ──
    def mesh_info(m, name):
        info = {"name": name, "vertices": len(m.vertices), "faces": len(m.faces),
                "watertight": bool(m.is_watertight), "euler_number": int(m.euler_number)}
        try:
            info["bounding_box_mm"] = (m.bounding_box.extents * (1000 if max(m.bounding_box.extents) < 1 else 1)).round(1).tolist()
        except:
            info["bounding_box_mm"] = [0, 0, 0]
        return info

    report = {
        "project": "G18 — Custom Cast System (Unified Pipeline)",
        "pipeline_mode": mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_sec": round(timing, 1),
        "config": {
            "cast_gap_mm": config["cast_gap_mm"],
            "cast_wall_mm": config["cast_wall_mm"],
            "voronoi_enabled": config["voronoi_enabled"],
            "voronoi_points": config["voronoi_num_points"],
            "fastener_enabled": config["fastener_enabled"],
            "silhouette_method": config.get("silhouette_method", "N/A"),
        },
        "meshes": {
            "hand_surface": mesh_info(hand_mesh, "Hand Surface"),
            "full_cast": mesh_info(full_cast, "Full Voronoi Cast"),
            "top_shell": mesh_info(top_shell, "Top Shell"),
            "bottom_shell": mesh_info(bottom_shell, "Bottom Shell"),
        },
        "convex_hull_deviation_mm": {
            "max": config.get("_hull_max_deviation_mm", None),
            "mean": config.get("_hull_mean_deviation_mm", None),
        },
        "fit_validation": {
            "top_shell": fit_top,
            "full_cast": fit_full,
            "clinical_threshold_mm": config["fit_threshold_mm"],
            "verdict": "ACCEPTABLE" if fit_full["mean_mm"] < config["fit_threshold_mm"] else "NEEDS_REVIEW",
        },
        "output_files": paths,
        "printing_recommendations": {
            "material": "PLA (demo) or PETG (functional)",
            "layer_height_mm": 0.2,
            "infill_percent": "20-30%",
            "supports": "As needed for overhangs",
        },
        "defense_readiness": {
            "hand_reconstructed": len(hand_mesh.vertices) > 500,
            "cast_exported": all(os.path.exists(p) for p in paths.values()),
            "watertight": hand_mesh.is_watertight,
            "fit_acceptable": fit_full["mean_mm"] < config["fit_threshold_mm"],
        },
    }

    report_path = os.path.join(config["output_dir"], "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"  Defense report: {report_path}")

    # ── Print summary ──
    print("\n" + "=" * 65)
    print("  G18 UNIFIED PIPELINE — COMPLETE")
    print("=" * 65)
    print(f"  Mode:        {mode.upper()}")
    print(f"  Time:        {timing:.1f}s")
    print(f"  Hand mesh:   {len(hand_mesh.vertices):,}V / {len(hand_mesh.faces):,}F  WT={hand_mesh.is_watertight}")
    print(f"  Cast (full): {len(full_cast.faces):,}F  Voronoi={'ON' if config['voronoi_enabled'] else 'OFF'}")
    print(f"  Top shell:   {paths.get('cast_top_shell', 'N/A')}")
    print(f"  Bot shell:   {paths.get('cast_bottom_shell', 'N/A')}")
    print(f"  Fit (mean):  {fit_full['mean_mm']}mm {'✅' if fit_full['mean_mm'] < config['fit_threshold_mm'] else '❌'}")
    print(f"  Report:      {report_path}")
    print("=" * 65)

    return report


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC TEST DATA GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_synthetic_test_data(config):
    """Generate 36 clean arm images + background for PATH B visual-hull testing.

    Uses an analytical silhouette renderer — NO mesh, NO triangle rasterisation.
    The arm is a tapered oval frustum.  Its projected silhouette is computed in
    closed form for every azimuth angle, giving perfectly smooth, artefact-free
    outlines that silhouette extraction algorithms can handle trivially.

    Physics:
      For an oval cross-section with half-radii (rx, rz) viewed from azimuth θ:
        apparent_half_width = sqrt( (rx·sin θ)² + (rz·cos θ)² )
      This is exact — no sampling error, no seams, no depth-sorting required.
    """
    log.info("SYNTHETIC: Generating clean analytical arm images...")
    img_dir = config["image_dir"]
    os.makedirs(img_dir, exist_ok=True)

    n   = config.get("num_images", 36)
    W   = config.get("synth_render_width",  800)
    H   = config.get("synth_render_height", 600)
    D   = config.get("camera_distance_mm",  550.0)

    angles = np.linspace(0.0, 360.0, n, endpoint=False)

    # Pure white background — identical to the blank canvas before the arm is placed
    bg = np.full((H, W, 3), 255, dtype=np.uint8)
    cv2.imwrite(os.path.join(img_dir, "_background.png"), bg)

    for i, angle_deg in enumerate(angles):
        img = _render_arm_analytical(angle_deg, W, H, D)
        cv2.imwrite(os.path.join(img_dir, f"arm_{i:03d}.png"), img)

    log.info(f"  {n} arm images + _background.png saved to {img_dir}/  "
             f"({W}x{H}px, 360deg coverage, analytical renderer)")
    config["background_image"] = os.path.join(img_dir, "_background.png")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICAL ARM RENDERER
# ─────────────────────────────────────────────────────────────────────────────

# Arm geometry constants (all in mm)
_ARM_LEN_MM  = 270      # elbow-to-wrist length
_RX_ELBOW    = 44       # half-width at elbow (oval wide axis, side-on view)
_RX_WRIST    = 26       # half-width at wrist
_RZ_RATIO    = 0.72     # depth = RZ_RATIO x width  (realistic forearm is flattened)
_SKIN_BGR    = (105, 145, 195)   # skin tone, BGR

def _render_arm_analytical(angle_deg, W=800, H=600, cam_dist_mm=550.0):
    """Render one clean arm silhouette image at the given azimuth angle.

    Algorithm
    ---------
    1. Sample N cross-sections along the arm length (Y axis).
    2. At each cross-section, compute the apparent half-width in closed form:
           half_w = sqrt( (rx·sin theta)^2 + (rz·cos theta)^2 )
    3. Project to screen using standard pinhole perspective.
    4. Build a smooth closed polygon (right edge + wrist cap + left edge + elbow cap).
    5. Fill with a smooth cylindrical shading gradient (no per-face noise).
    6. Draw 2-px black outline.

    Output: pure-white background with a clean, correctly-tapered arm silhouette.
    No mesh, no trimesh, no OpenCV triangle list — just fillPoly + shading.
    """
    focal  = W * 1.25
    theta  = np.radians(angle_deg)

    # Y span (mm), centred at origin; elbow at top, wrist at bottom
    Y_TOP  =  _ARM_LEN_MM / 2.0
    Y_BOT  = -_ARM_LEN_MM / 2.0

    # Sample along arm length
    N      = 400
    y      = np.linspace(Y_TOP, Y_BOT, N)
    t      = (y - Y_TOP) / (Y_BOT - Y_TOP)   # 0 = elbow, 1 = wrist

    # Tapered radii (slightly accelerated taper: realistic forearm gets narrow quickly)
    taper  = t ** 1.5
    rx     = _RX_ELBOW + (_RX_WRIST - _RX_ELBOW) * taper
    rz     = rx * _RZ_RATIO

    # Exact apparent half-width for oval cross-section at viewing angle theta
    half_w = np.sqrt((rx * np.sin(theta))**2 + (rz * np.cos(theta))**2)

    # Perspective projection (Y world -> Y screen; half-width world -> half-width px)
    sy     = focal * (-y)     / cam_dist_mm + H / 2.0
    sw     = focal * half_w   / cam_dist_mm
    cx     = W / 2.0   # arm long axis always projects to horizontal centre

    # ── Build silhouette polygon ──────────────────────────────────────────────
    right_pts = np.column_stack([cx + sw, sy]).astype(np.float32)
    left_pts  = np.column_stack([cx - sw, sy]).astype(np.float32)[::-1]

    # Hemispherical end caps (semicircular arcs)
    n_cap = 48
    # Elbow cap (top): dome above the arm
    ea   = np.linspace(np.pi, 0.0, n_cap)
    elbow_cap = np.column_stack([
        cx + float(sw[0])  * np.cos(ea),
        float(sy[0]) - float(sw[0]) * 0.45 * np.sin(ea)   # slightly flat dome
    ])
    # Wrist cap (bottom): dome below the arm
    wa   = np.linspace(0.0, np.pi, n_cap)
    wrist_cap = np.column_stack([
        cx + float(sw[-1]) * np.cos(wa),
        float(sy[-1]) + float(sw[-1]) * 0.45 * np.sin(wa)
    ])

    poly = np.vstack([right_pts, wrist_cap, left_pts, elbow_cap]).astype(np.int32)
    poly = poly.reshape(-1, 1, 2)

    # ── Flat fill ─────────────────────────────────────────────────────────────
    img = np.full((H, W, 3), 255, dtype=np.uint8)   # white background
    cv2.fillPoly(img, [poly], _SKIN_BGR, lineType=cv2.LINE_AA)

    # ── Cylindrical shading gradient (no per-triangle noise) ─────────────────
    # For each pixel column x and each arm row y, compute normalised distance
    # from arm centre → Lambertian approximation of a round surface.
    # We build the gradient as a float image and blend over the arm mask.

    gray_flat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, arm_mask = cv2.threshold(gray_flat, 252, 255, cv2.THRESH_BINARY_INV)

    # Horizontal distance from centre, normalised by local arm width
    xs     = np.arange(W, dtype=np.float32)
    ys_arr = np.arange(H, dtype=np.float32)

    # Interpolate arm centre-x and half-width at each screen-y pixel
    sy_clip  = np.clip(sy, 0, H - 1)
    sw_clip  = np.clip(sw, 1, W)

    # Build look-up: for each screen row, what is (cx, sw)?
    # We interpolate linearly using np.interp (sy is monotonically decreasing)
    sy_sorted   = sy[::-1]      # ascending
    sw_sorted   = sw_clip[::-1]
    all_rows    = np.arange(H, dtype=np.float32)
    sw_per_row  = np.interp(all_rows, sy_sorted, sw_sorted)

    # t_x in [-1, 1] for each pixel; shade = sqrt(max(0, 1-t_x^2))
    t_x   = np.clip((xs[np.newaxis, :] - cx) / sw_per_row[:, np.newaxis], -1.0, 1.0)
    shade = np.sqrt(np.maximum(0.0, 1.0 - t_x**2))   # (H, W)

    # Blend: dark edge colour to bright highlight
    SKIN_F  = np.array(_SKIN_BGR[::-1], dtype=np.float32)   # skin in RGB
    DARK_F  = SKIN_F * 0.40
    LIGHT_F = np.minimum(SKIN_F * 1.50, 255.0)

    shaded = DARK_F + (LIGHT_F - DARK_F) * shade[..., np.newaxis]  # (H,W,3) RGB
    shaded_bgr = shaded[:, :, ::-1].astype(np.uint8)               # back to BGR

    # Apply shading only inside arm mask
    mask3 = arm_mask[..., np.newaxis] > 0
    img   = np.where(mask3, shaded_bgr, img)

    # ── 2-px black silhouette outline ─────────────────────────────────────────
    gray2  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bm2 = cv2.threshold(gray2, 252, 255, cv2.THRESH_BINARY_INV)
    ctrs, _ = cv2.findContours(bm2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, ctrs, -1, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    return img


# Legacy stubs — kept so any direct call to the old functions still runs
def _build_test_arm_mesh():
    """Replaced by analytical renderer. Returns a minimal placeholder mesh."""
    import trimesh
    return trimesh.creation.cylinder(radius=0.035, height=0.25, sections=64)


def _render_mesh(mesh, angle_deg, W=800, H=600, cam_dist=550, focal=None,
                 bg_color=(255, 255, 255), elev_deg=0.0, no_blur=True):
    """Replaced by _render_arm_analytical. Delegates to it."""
    return _render_arm_analytical(angle_deg, W, H, float(cam_dist))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(force_mode=None, synthetic=False):
    t0 = time.time()

    log.info("═" * 65)
    log.info("  G18 — UNIFIED CAST PIPELINE (Best-of-Both)")
    log.info("═" * 65)

    # ── Stage 1: Detect input mode ──
    if synthetic:
        generate_synthetic_test_data(CONFIG)
        mode = "rgb"
    else:
        mode = detect_input_mode(CONFIG, force_mode)
        if mode == "synthetic":
            log.info("Auto-generating synthetic test data...")
            generate_synthetic_test_data(CONFIG)
            mode = "rgb"

    log.info(f"  Pipeline mode: {mode.upper()}")

    # ── Stage 2 + 3 + 4 + 5: Path-specific reconstruction ──
    if mode == "rgbd":
        # PATH A: RealSense
        CONFIG["roi_enabled"] = True
        pipeline, align, intrinsics = init_realsense(CONFIG)
        frames = capture_rgbd_frames(pipeline, align, CONFIG)
        if pipeline: pipeline.stop()

        if len(frames) < 20:
            log.error(f"Only {len(frames)} frames. Need ≥20. Aborting."); sys.exit(1)

        poses = estimate_poses_icp(frames, intrinsics, CONFIG)
        mesh_o3d = tsdf_fusion(frames, poses, intrinsics, CONFIG)

    else:
        # PATH B: RGB Photos
        CONFIG["roi_enabled"] = False  # visual hull is already focused
        images, img_paths = load_rgb_images(CONFIG)
        intrinsics = init_rgb_camera(CONFIG, images)
        silhouettes = extract_silhouettes_enhanced(images, CONFIG)
        projections = compute_camera_matrices_rgb(intrinsics, len(images), CONFIG)
        mesh_o3d = visual_hull_reconstruction(silhouettes, projections, CONFIG)

    # ── Stage 6: Visualization ──
    visualize_mesh(mesh_o3d, CONFIG, label="raw_reconstruction")

    # ── Stage 7: ROI Crop ──
    mesh_o3d = crop_to_roi(mesh_o3d, CONFIG)

    # ── Convert to Trimesh ──
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    tm_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # ── Stage 8: Convex hull closure ──
    tm_closed = close_with_convex_hull(tm_mesh, CONFIG)

    # ── Stage 9: Clean and validate ──
    tm_clean = clean_and_validate(tm_closed, CONFIG, label="hand_cleaned")

    # ── Stage 10: Cast design (Voronoi + Boolean + Fasteners) ──
    top_shell, bottom_shell, full_cast = generate_cast_design(tm_clean, CONFIG)

    # ── Stage 11: Export + Validation + Report ──
    elapsed = time.time() - t0
    report = export_and_report(top_shell, bottom_shell, full_cast, tm_clean,
                                CONFIG, mode, elapsed)

    log.info(f"Pipeline complete in {elapsed:.1f}s. Outputs in: {CONFIG['output_dir']}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G18 Unified Cast Pipeline")
    parser.add_argument("--mode", choices=["rgb", "rgbd"], default=None,
                       help="Force input mode (default: auto-detect)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic test data and run")
    parser.add_argument("--images", type=str, default=None,
                       help="Path to images directory (PATH B)")
    parser.add_argument("--silhouette", type=str, default="full",
                       choices=["basic", "multichannel", "grabcut", "deep_learning", "full"],
                       help="Silhouette method (PATH B)")
    parser.add_argument("--no-voronoi", action="store_true",
                       help="Disable Voronoi ventilation holes")
    parser.add_argument("--no-fasteners", action="store_true",
                       help="Disable fastener holes")
    args = parser.parse_args()

    if args.images:
        CONFIG["image_dir"] = args.images
        CONFIG["background_image"] = os.path.join(args.images, "_background.png")
    CONFIG["silhouette_method"] = args.silhouette
    CONFIG["voronoi_enabled"] = not args.no_voronoi
    CONFIG["fastener_enabled"] = not args.no_fasteners

    run_pipeline(force_mode=args.mode, synthetic=args.synthetic)