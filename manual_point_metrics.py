# manual_point_metrics.py
# -----------------------------------
# Minimal, interactive pipeline to extract key summary metrics from a video
# using user-selected frames & points. Designed for black-material cases
# where colour segmentation is not reliable.
#
# Summary CSV format matches your existing pipeline:
#   video, inflation_period_s,
#   max_lin_mm, max_lin_time,
#   max_rad_mm, max_rad_time,
#   max_area_mm, max_area_time,
#   max_bend_deg, max_bend_mm, max_bend_time,
#   mm_per_px
#
# Usage:
#   - Set INPUT_VIDEO, OUTPUT_DIR, CALIB_SECS below
#   - Run: python manual_point_metrics.py
#   - Follow on-screen prompts to choose frames and click points.
# -----------------------------------

import os
import cv2
import csv
import math
import numpy as np

# ------------- I/O (set these) -------------
INPUT_VIDEO = "raw_video_data/ExRunFinal - Protract/lin_collapse1_test1.mp4"   # <- SET
OUTPUT_DIR  = "./processed_video_data/ExRunFinal - Protract"   # <- SET
CALIB_SECS  = 2.0                               # seconds from start to try chessboard calib

# ------------- Calibration settings -------------
BOARD_SQUARES = (5, 5)               # physical squares on board
BOARD_INNER   = (BOARD_SQUARES[0]-1, BOARD_SQUARES[1]-1)
SQUARE_MM     = 10.0                 # mm per square edge
DETECTIONS_REQ   = 1
FRAME_STRIDE     = 1
MAX_CALIB_FRAMES = 300               # hard cap during calib scan
SCALE_DEF        = 0.38              # fallback mm/px if calibration fails

# Relative vs Absolute Extension 
EXTENSION_MODE = "absolute"  # "relative" or "absolute"

# ------------- Summary CSV name -------------
SUMMARY_CSV = "summary_results.csv"

# ---------------- Helpers ----------------
def _robust_mm_per_px_from_corners(corners, square_mm: float) -> float | None:
    if corners is None or len(corners) == 0:
        return None
    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])
    pts = corners.reshape(-1, 2)
    dists = []
    # horizontal neighbors
    for r in range(rows_inner):
        base = r * cols_inner
        for c in range(cols_inner - 1):
            p0 = pts[base + c]; p1 = pts[base + c + 1]
            dists.append(np.hypot(*(p1 - p0)))
    # vertical neighbors
    for c in range(cols_inner):
        for r in range(rows_inner - 1):
            p0 = pts[r * cols_inner + c]; p1 = pts[(r + 1) * cols_inner + c]
            dists.append(np.hypot(*(p1 - p0)))
    if not dists:
        return None
    px_mean = float(np.median(dists))
    return square_mm / px_mean if px_mean > 1e-6 else None

def calibrate_from_video_head(cap, seconds: float):
    """
    Try to calibrate using only the first `seconds` of video.
    Returns: (ok, mm_per_px)
    """
    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, seconds * fps))
    total = int(min(total, MAX_CALIB_FRAMES))

    objp = np.zeros((rows_inner * cols_inner, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols_inner, 0:rows_inner].T.reshape(-1, 2) * SQUARE_MM

    objpoints, imgpoints = [], []
    scale_estimates = []
    gray = None
    detections = 0
    origin = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fidx in range(0, total, FRAME_STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols_inner, rows_inner))
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)
        detections += 1

        scale_i = _robust_mm_per_px_from_corners(corners, SQUARE_MM)
        if scale_i is not None:
            scale_estimates.append(scale_i)
        if detections >= DETECTIONS_REQ:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, origin)
    if not objpoints or gray is None:
        return False, None
    scale = float(np.median(scale_estimates)) if scale_estimates else None
    return True, (scale if scale else None)

# --------- Interactive helpers (frames & clicks) ---------
def choose_frame(cap, window_name="choose_frame"):
    """
    Lightweight frame scrubber:
      - use slider to choose a frame
      - press ENTER (Return) to confirm
      - press q to cancel (returns None)
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        print("[ERROR] No frames available.")
        return None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    pos = 0

    def on_trackbar(val):
        nonlocal pos
        pos = val
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, fr = cap.read()
        if ok:
            cv2.imshow(window_name, fr)

    cv2.createTrackbar("frame", window_name, 0, max(0, total-1), on_trackbar)
    # initialize
    on_trackbar(0)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 13:  # Enter
            chosen = int(cv2.getTrackbarPos("frame", window_name))
            cv2.destroyWindow(window_name)
            return chosen
        if k == ord('q'):
            cv2.destroyWindow(window_name)
            return None

def pick_two_points(image, window_name="pick_points"):
    """
    Ask user to pick BASE (1) then TIP (2).
      - Left-click to add a point
      - Press 'r' to reset points
      - Press ENTER to confirm when two points are placed
      - Press 'q' to cancel (returns None)
    Returns: ((base_x, base_y), (tip_x, tip_y)) in image coords
    """
    pts = []

    def on_mouse(event, x, y, flags, param):
        nonlocal pts, img_disp
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 2:
                pts.append((x, y))
                draw()
    def draw():
        img_disp[:] = image
        txt = "Click BASE center then TIP (r=reset, Enter=ok, q=cancel)"
        cv2.putText(img_disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            color = (0, 255, 255) if i == 0 else (255, 255, 0)
            cv2.circle(img_disp, p, 6, color, -1)
        if len(pts) == 2:
            cv2.line(img_disp, pts[0], pts[1], (255,255,255), 2)

    img_disp = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    draw()
    cv2.imshow(window_name, img_disp)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            pts = []
            draw()
            cv2.imshow(window_name, img_disp)
        elif k == 13:  # Enter
            if len(pts) == 2:
                cv2.destroyWindow(window_name)
                return pts[0], pts[1]
        elif k == ord('q'):
            cv2.destroyWindow(window_name)
            return None

# ----------------- Main flow -----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("[ERROR] Cannot open:", INPUT_VIDEO)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt  = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # --- Calibration (first CALIB_SECS) ---
    ok, scale_est = calibrate_from_video_head(cap, CALIB_SECS)
    if ok and scale_est:
        mm_per_px = float(scale_est)
        print(f"[INFO] Calibration OK. mm/px = {mm_per_px:.6f}")
    else:
        mm_per_px = float(SCALE_DEF)
        print(f"[WARN] Calibration failed or insufficient; fallback mm/px = {mm_per_px:.6f}")

    # --- Choose MIN expansion frame ---
    print("\n[STEP] Choose MIN expansion frame (rest/baseline). Press Enter to confirm, q to cancel.")
    min_frame_idx = choose_frame(cap, "Choose MIN frame")
    if min_frame_idx is None:
        print("[ABORT] No frame chosen.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame_idx)
    ok, min_frame = cap.read()
    if not ok:
        print("[ERROR] Could not read selected MIN frame.")
        return

    # --- Choose MAX expansion frame ---
    print("\n[STEP] Choose MAX expansion frame (peak). Press Enter to confirm, q to cancel.")
    max_frame_idx = choose_frame(cap, "Choose MAX frame")
    if max_frame_idx is None:
        print("[ABORT] No frame chosen.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, max_frame_idx)
    ok, max_frame = cap.read()
    if not ok:
        print("[ERROR] Could not read selected MAX frame.")
        return

    # --- Pick BASE & TIP on MIN frame ---
    print("\n[STEP] MIN frame: Click BASE (bottom center) then TIP (end-effector).")
    picks_min = pick_two_points(min_frame.copy(), "Pick MIN points")
    if picks_min is None:
        print("[ABORT] Points not selected on MIN frame.")
        return
    (base_min_x, base_min_y), (tip_min_x, tip_min_y) = picks_min

    # --- Pick BASE & TIP on MAX frame ---
    print("\n[STEP] MAX frame: Click BASE (bottom center) then TIP (end-effector).")
    picks_max = pick_two_points(max_frame.copy(), "Pick MAX points")
    if picks_max is None:
        print("[ABORT] Points not selected on MAX frame.")
        return
    (base_max_x, base_max_y), (tip_max_x, tip_max_y) = picks_max

    # --- Compute metrics ---
    # vertical separations (screen y increases downward → use base_y - tip_y)
    sep_min_px = (base_min_y - tip_min_y)
    sep_max_px = (base_max_y - tip_max_y)
    if EXTENSION_MODE == "absolute":
        lin_extension_px = sep_max_px
    elif EXTENSION_MODE == "relative":  # relative
        lin_extension_px = (sep_max_px - sep_min_px)
    lin_extension_mm = lin_extension_px * mm_per_px

    # bending at MAX: angle of base->tip vs vertical; horizontal displacement
    dx_max_px = (tip_max_x - base_max_x)
    dy_max_px = (base_max_y - tip_max_y)
    bend_rad  = math.atan2(dx_max_px, dy_max_px)  # 0 = vertical
    bend_deg  = abs(math.degrees(bend_rad))
    horiz_disp_mm = dx_max_px * mm_per_px

    # Optional: reference stats at MIN as well (angle/displacement)
    dx_min_px = (tip_min_x - base_min_x)
    dy_min_px = (base_min_y - tip_min_y)
    bend_min_deg = abs(math.degrees(math.atan2(dx_min_px, dy_min_px)))
    horiz_min_mm = dx_min_px * mm_per_px
    bend_delta_deg = bend_deg - bend_min_deg
    horiz_delta_mm = horiz_disp_mm - horiz_min_mm

    # inflation period between chosen frames
    inflation_period_s = abs(max_frame_idx - min_frame_idx) * dt
    max_bend_time = abs(max_frame_idx - min_frame_idx) * dt  # relative to MIN as baseline

    # --- Write annotated snapshots (optional but handy) ---
    ann_min = min_frame.copy()
    cv2.circle(ann_min, (int(base_min_x), int(base_min_y)), 6, (0,255,255), -1)
    cv2.circle(ann_min, (int(tip_min_x),  int(tip_min_y)),  6, (255,255,0), -1)
    cv2.line(ann_min, (int(base_min_x), int(base_min_y)), (int(tip_min_x), int(tip_min_y)), (255,255,255), 2)
    ann_max = max_frame.copy()
    cv2.circle(ann_max, (int(base_max_x), int(base_max_y)), 6, (0,255,255), -1)
    cv2.circle(ann_max, (int(tip_max_x),  int(tip_max_y)),  6, (255,255,0), -1)
    cv2.line(ann_max, (int(base_max_x), int(base_max_y)), (int(tip_max_x), int(tip_max_y)), (255,255,255), 2)

    name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}__min_points.jpg"), ann_min)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}__max_points.jpg"), ann_max)

    # --- Summary CSV (compatible with your existing reader) ---
    csv_summary = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    header = [
        "video","inflation_period_s",
        "max_lin_mm","max_lin_time",
        "max_rad_mm","max_rad_time",
        "max_area_mm","max_area_time",
        "max_bend_deg","max_bend_mm","max_bend_time",
        "mm_per_px",
    ]
    need_hdr = not os.path.exists(csv_summary)

    with open(csv_summary, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        if need_hdr:
            wr.writeheader()
        wr.writerow(dict(
            video=name,
            inflation_period_s=inflation_period_s,
            max_lin_mm=float(lin_extension_mm),
            max_lin_time=float(inflation_period_s),  # ‘max’ at your chosen peak
            max_rad_mm=0.0,                          # unknown in manual mode
            max_rad_time=0.0,
            max_area_mm=0.0,
            max_area_time=0.0,
            max_bend_deg=float(bend_deg),
            max_bend_mm=float(horiz_disp_mm),
            max_bend_time=float(max_bend_time),
            mm_per_px=float(mm_per_px),
        ))

    # Also print a quick console summary
    print("\n===== Manual Metrics =====")
    print(f"video                 : {name}")
    print(f"mm/px (scale)        : {mm_per_px:.6f}")
    print(f"inflation period (s) : {inflation_period_s:.3f}")
    print(f"linear extension (mm): {lin_extension_mm:+.2f}")
    print(f"bend@max (deg)       : {bend_deg:.2f}  (Δ vs min: {bend_delta_deg:+.2f})")
    print(f"horiz disp@max (mm)  : {horiz_disp_mm:+.2f} (Δ vs min: {horiz_delta_mm:+.2f})")
    print("Snapshots saved to    :", OUTPUT_DIR)
    print("Summary appended to   :", csv_summary)

if __name__ == "__main__":
    main()
