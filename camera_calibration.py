#!/usr/bin/env python3
"""
camera_calibration.py  -  TAHAP 1: INTRINSIC CAMERA CALIBRATION (checkerboard)
===============================================================================

Tiga mode:
  collect (default) : kumpulkan sample checkerboard -> hitung calibration -> simpan -> preview undistort
  test              : muat hasil calibration, tampilkan ORIGINAL | UNDISTORTED
  camera-test       : diagnostik kamera SAJA (tanpa checkerboard/calibration): frame + FPS

Contoh (Logitech C270, papan 8x8 kotak = 7x7 inner corner, kotak 34 mm):
  python camera_calibration.py --camera-index 1 --cols 7 --rows 7 --square-size 34
  python camera_calibration.py --mode test --camera-index 1
  python camera_calibration.py --mode camera-test --camera-index 1

PENTING - ukuran checkerboard:
  --cols dan --rows adalah jumlah INNER CORNER (sudut dalam, titik pertemuan 4 kotak),
  BUKAN jumlah kotak. Papan 8 x 8 kotak  ->  --cols 7 --rows 7.
  --square-size = ukuran nyata satu kotak dalam mm (BUKAN pixel). Papan Anda: 34 mm.

Kontrol (mode collect):
  SPASI / s : simpan sample (ditolak bila posisi terlalu mirip sample sebelumnya)
  f         : simpan sample paksa (abaikan cek kemiripan)
  u         : hapus sample terakhir
  r         : hapus semua sample
  c         : hitung calibration + simpan + preview undistort
  q / ESC   : keluar

Kontrol (preview/test):
  g : tampilkan/sembunyikan grid (garis lurus membantu menilai koreksi distorsi)
  s : simpan screenshot PNG
  b : kembali mengumpulkan sample (hanya bila dibuka dari mode collect)
  q / ESC : keluar

Kontrol (camera-test):
  q / ESC : keluar
"""

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from calib_io import IntrinsicCalibration, Undistorter
from camera_settings import HW_SPECS, LIGHT_COLORS, light_status, open_camera
from camera_source import WEBCAM_DEFAULT_FPS, WEBCAM_DEFAULT_HEIGHT, WEBCAM_DEFAULT_WIDTH, CameraBase

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------------------------------------------------------------
# Konstanta (silakan tuning)
# --------------------------------------------------------------------------------------
WORKING_RESOLUTION = (WEBCAM_DEFAULT_WIDTH, WEBCAM_DEFAULT_HEIGHT)   # 640x480, resolusi kerja proyek
MIN_SAMPLES_HARD = 6        # di bawah ini calibration ditolak
NOT_FOUND_WARN_SEC = 3.0    # peringatan "checkerboard tidak ditemukan" setelah N detik

# Klasifikasi pose untuk checklist cakupan
EDGE_LOW, EDGE_HIGH = 0.25, 0.75   # papan menyentuh area kiri/kanan/atas/bawah citra
CENTER_RADIUS = 0.20               # pusat papan dekat tengah citra
SCALE_NEAR, SCALE_FAR = 0.50, 0.30 # sqrt(luas papan / luas citra)
TILT_MIN = 0.08                    # |log(rasio sisi)| minimum agar dianggap miring

# Batas kemiripan antar-sample (semua kondisi terpenuhi -> dianggap terlalu mirip)
SIM_POS, SIM_SCALE, SIM_TILT = 0.04, 0.06, 0.05

COVERAGE_ORDER = ["CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM", "NEAR", "FAR", "TILTED"]

# Flag deteksi. FAST_CHECK = tolak cepat frame tanpa papan (untuk preview).
FLAGS_PREVIEW = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
FLAGS_FINAL = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# --------------------------------------------------------------------------------------
# Util tampilan
# --------------------------------------------------------------------------------------
def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def put_wrapped(img, text, x, y_bottom, scale=0.5, color=(255, 255, 255), thick=1):
    """Teks yang dipotong per baris agar tidak keluar dari citra 640 px; baris terakhir di y_bottom."""
    w = img.shape[1]
    max_chars = max(20, int((w - x - 10) / (22 * scale)))
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > max_chars and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    if cur:
        lines.append(cur)
    step = int(26 * scale + 4)
    for i, ln in enumerate(reversed(lines)):
        put_text(img, ln, (x, y_bottom - i * step), scale, color, thick)


def show(win: str, img: np.ndarray, scale: float):
    if abs(scale - 1.0) > 1e-3:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(win, img)


class FpsMeter:
    """FPS dengan moving average (jendela N frame) + celah terpanjang antar frame (deteksi patah-patah)."""

    def __init__(self, window: int = 30):
        self.t = deque(maxlen=window + 1)

    def tick(self):
        self.t.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self.t) < 2:
            return 0.0
        dt = self.t[-1] - self.t[0]
        return (len(self.t) - 1) / dt if dt > 0 else 0.0

    @property
    def max_gap_ms(self) -> float:
        if len(self.t) < 2:
            return 0.0
        ts = list(self.t)
        return 1000.0 * max(b - a for a, b in zip(ts[:-1], ts[1:]))


def print_resolution_mismatch(calib_size: Tuple[int, int], current: Tuple[int, int]):
    print("\n[WARNING] CALIBRATION RESOLUTION MISMATCH")
    print(f"Calibration : {calib_size[0]}x{calib_size[1]}")
    print(f"Current     : {current[0]}x{current[1]}")
    print("Intrinsic calibration TIDAK boleh dipakai pada resolusi berbeda. Jalankan ulang dengan "
          f"--width {calib_size[0]} --height {calib_size[1]} atau lakukan calibration ulang.\n")


def print_banner(args, width: int, height: int):
    backend = "DirectShow" if sys.platform.startswith("win") else "default (bukan Windows)"
    print("=" * 40)
    print("CAMERA CALIBRATION" if args.mode != "camera-test" else "CAMERA TEST (tanpa checkerboard)")
    print("=" * 40)
    print("Camera source  : Webcam (Logitech C270 / USB RGB, tanpa depth)")
    print(f"Camera index   : {args.camera_index}")
    print(f"Backend        : {backend}")
    print(f"Resolution     : {width} x {height}")
    print(f"FPS target     : {args.fps}")
    if args.mode == "collect":
        print(f"Board squares  : {args.cols + 1} x {args.rows + 1}")
        print(f"Inner corners  : {args.cols} x {args.rows}   (dipakai OpenCV)")
        print(f"Square size    : {args.square_size:g} mm")
        print(f"Preview detect : tiap {args.detect_every} frame, skala {args.detect_scale:g} "
              f"(sample final: resolusi penuh + cornerSubPix)")
    print("=" * 40)


# --------------------------------------------------------------------------------------
# Deteksi checkerboard & deskripsi pose
# --------------------------------------------------------------------------------------
def make_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """Titik 3D checkerboard (bidang Z=0) dalam mm."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return objp


def detect_preview(gray: np.ndarray, pattern: Tuple[int, int], scale: float = 1.0):
    """Deteksi RINGAN untuk live preview: tanpa cornerSubPix; boleh pada citra yang diperkecil.
    Koordinat dikembalikan dalam ruang citra ASLI. Hasil ini TIDAK dipakai untuk calibration."""
    g = gray
    if scale < 0.999:
        g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, corners = cv2.findChessboardCorners(g, pattern, FLAGS_PREVIEW)
    if found and scale < 0.999:
        corners = (corners / scale).astype(np.float32)
    return found, corners


def find_corners(gray: np.ndarray, pattern: Tuple[int, int]):
    """Deteksi FINAL pada resolusi penuh + cornerSubPix. Dipakai saat menyimpan sample."""
    found, corners = cv2.findChessboardCorners(gray, pattern, FLAGS_FINAL)
    if found:
        cols, rows = pattern
        c = corners.reshape(rows, cols, 2)
        spacing = float(np.linalg.norm(c[:, 1:] - c[:, :-1], axis=2).min())
        half = int(np.clip(0.45 * spacing, 3, 11))      # jendela tidak boleh melewati corner tetangga
        corners = cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), SUBPIX_CRITERIA)
    return found, corners


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Urutkan 4 titik di ruang citra -> TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


@dataclass
class Pose:
    cx: float; cy: float                       # pusat papan (ternormalisasi 0..1)
    min_x: float; max_x: float                 # bounding box papan (ternormalisasi)
    min_y: float; max_y: float
    scale: float                               # sqrt(luas papan / luas citra)
    tilt_x: float; tilt_y: float               # log rasio sisi berhadapan (0 = papan tegak lurus kamera)


def describe_pose(corners: np.ndarray, pattern: Tuple[int, int], size: Tuple[int, int]) -> Pose:
    cols, _ = pattern
    w, h = size
    c = corners.reshape(-1, 2)
    quad = _order_quad(np.array([c[0], c[cols - 1], c[-1], c[-cols]], np.float32))
    tl, tr, br, bl = quad
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    area = cv2.contourArea(quad.astype(np.float32))
    return Pose(
        cx=float(c[:, 0].mean() / w), cy=float(c[:, 1].mean() / h),
        min_x=float(c[:, 0].min() / w), max_x=float(c[:, 0].max() / w),
        min_y=float(c[:, 1].min() / h), max_y=float(c[:, 1].max() / h),
        scale=float(np.sqrt(area / (w * h))),
        tilt_x=float(np.log(max(top, 1e-6) / max(bottom, 1e-6))),
        tilt_y=float(np.log(max(left, 1e-6) / max(right, 1e-6))),
    )


def pose_tags(p: Pose) -> Set[str]:
    tags = set()
    if abs(p.cx - 0.5) < CENTER_RADIUS and abs(p.cy - 0.5) < CENTER_RADIUS:
        tags.add("CENTER")
    if p.min_x < EDGE_LOW:
        tags.add("LEFT")
    if p.max_x > EDGE_HIGH:
        tags.add("RIGHT")
    if p.min_y < EDGE_LOW:
        tags.add("TOP")
    if p.max_y > EDGE_HIGH:
        tags.add("BOTTOM")
    if p.scale > SCALE_NEAR:
        tags.add("NEAR")
    if p.scale < SCALE_FAR:
        tags.add("FAR")
    if max(abs(p.tilt_x), abs(p.tilt_y)) > TILT_MIN:
        tags.add("TILTED")
    return tags


def find_similar(p: Pose, poses: List[Pose]) -> Optional[int]:
    """Index sample yang terlalu mirip dengan pose p, atau None."""
    for i, q in enumerate(poses):
        if (abs(p.cx - q.cx) < SIM_POS and abs(p.cy - q.cy) < SIM_POS and abs(p.scale - q.scale) < SIM_SCALE
                and abs(p.tilt_x - q.tilt_x) < SIM_TILT and abs(p.tilt_y - q.tilt_y) < SIM_TILT):
            return i
    return None


def coverage(poses: List[Pose]) -> Set[str]:
    covered: Set[str] = set()
    for p in poses:
        covered |= pose_tags(p)
    return covered


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def run_calibration(objpoints, imgpoints, size, board, camera_info, poses, args):
    """Jalankan cv2.calibrateCamera, hitung error, kembalikan (IntrinsicCalibration, daftar warning)."""
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

    # Error reprojection: jarak (pixel) antara corner terdeteksi dan corner hasil proyeksi model.
    per_view, all_d = [], []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        d = np.linalg.norm(imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        per_view.append(float(d.mean()))
        all_d.append(d)
    mean_err = float(np.concatenate(all_d).mean())

    calib = IntrinsicCalibration(
        camera_matrix=K, dist_coeffs=D.reshape(1, -1), image_size=size,
        rms_error=float(rms), mean_reprojection_error=mean_err, num_samples=len(objpoints),
        board=board, camera_info=camera_info, created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        per_view_errors=per_view)

    warns = []
    if len(objpoints) < args.min_samples:
        warns.append(f"Sample hanya {len(objpoints)} (disarankan >= {args.min_samples}).")
    missing = [t for t in COVERAGE_ORDER if t not in coverage(poses)]
    if missing:
        warns.append("Posisi papan belum tercakup: " + ", ".join(missing) + ".")
    if mean_err > args.max_error or rms > args.max_error:
        warns.append(f"Error calibration besar (RMS {rms:.3f}px, mean {mean_err:.3f}px > batas {args.max_error}px). "
                     f"Tekan 'b' untuk menambah sample yang lebih beragam & tajam, atau ulangi.")
    w, h = size
    if abs(calib.cx - w / 2) > 0.10 * w or abs(calib.cy - h / 2) > 0.10 * h:
        warns.append("Principal point (cx, cy) jauh dari tengah citra - kemungkinan sample kurang beragam.")
    if abs(calib.fx - calib.fy) / max(calib.fx, calib.fy) > 0.03:
        warns.append("fx dan fy berbeda > 3% (piksel kamera seharusnya hampir persegi) - periksa sample.")
    worst = sorted(range(len(per_view)), key=lambda i: per_view[i], reverse=True)[:3]
    bad = [i for i in worst if per_view[i] > max(2.0 * mean_err, 0.5)]
    if bad:
        warns.append("Sample dengan error tinggi (kemungkinan buram/bergerak): " +
                     ", ".join(f"#{i + 1} ({per_view[i]:.2f}px)" for i in bad) + ".")
    return calib, warns


def print_result(calib: IntrinsicCalibration, warns: List[str]):
    np.set_printoptions(precision=5, suppress=True)
    print("\n" + "=" * 62)
    print("HASIL INTRINSIC CALIBRATION")
    print("=" * 62)
    print(f"Resolusi            : {calib.image_size[0]} x {calib.image_size[1]}")
    print(f"Jumlah sample       : {calib.num_samples}")
    print("Camera matrix       :")
    print(calib.camera_matrix)
    print(f"fx = {calib.fx:.3f}   fy = {calib.fy:.3f}   cx = {calib.cx:.3f}   cy = {calib.cy:.3f}")
    print(f"Distortion (k1,k2,p1,p2,k3) : {calib.dist_coeffs.ravel()}")
    print(f"RMS calibration error       : {calib.rms_error:.4f} px   (dari cv2.calibrateCamera)")
    print(f"Mean reprojection error     : {calib.mean_reprojection_error:.4f} px   (rata-rata jarak corner)")
    print("Panduan: < 0.5 px bagus | 0.5 - 1.0 px cukup | > 1.0 px sebaiknya ulangi")
    print("Error per sample    : " + ", ".join(f"#{i + 1}={e:.2f}" for i, e in enumerate(calib.per_view_errors)))
    for w in warns:
        print("[WARNING]", w)
    print("=" * 62 + "\n")


# --------------------------------------------------------------------------------------
# Mode camera-test (tanpa checkerboard)
# --------------------------------------------------------------------------------------
def run_camera_test(cam: CameraBase):
    """Camera -> read -> imshow -> FPS. Tanpa findChessboardCorners / cornerSubPix / calibration."""
    win = "CAMERA TEST - tanpa checkerboard"
    cv2.namedWindow(win)
    meter, fails, read_ms = FpsMeter(), 0, 0.0
    all_fps: List[float] = []
    frame_i, light = 0, ("OK", 0.0, 0.0)
    hw = getattr(cam, "hw", None)
    if hw is not None and hw.available:                      # tampilkan status hardware yang terbaca (jujur)
        print("[INFO] Hardware settings saat ini (readback):")
        for sp in HW_SPECS:
            raw = hw.read_raw(sp.key)
            print(f"   {sp.label:<19} readback={'-' if raw is None else f'{raw:g}'}"
                  f"  ({hw.state[sp.key].status})")
    print("[INFO] CAMERA TEST: hanya read + imshow + FPS. 'q' / ESC = keluar.")
    print("[INFO] Bila di sini sudah lag -> masalah kamera/backend/resolusi. Bila lancar tetapi mode collect lag "
          "-> masalah pemrosesan checkerboard.")
    while True:
        t0 = time.perf_counter()
        fr = cam.read()
        if fr is None:
            fails += 1
            if fails > 100:
                print("[ERROR] Kamera tidak mengirim frame.")
                break
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        fails = 0
        dt_ms = (time.perf_counter() - t0) * 1000.0
        read_ms = dt_ms if read_ms == 0.0 else 0.9 * read_ms + 0.1 * dt_ms
        meter.tick()
        frame = fr.color
        h, w = frame.shape[:2]
        put_text(frame, f"FPS: {meter.fps:.1f}", (10, 28), 0.8, (0, 255, 0), 2)
        put_text(frame, f"Resolution: {w}x{h}", (10, 54), 0.55)
        put_text(frame, f"read {read_ms:.1f} ms | gap max {meter.max_gap_ms:.0f} ms", (10, 76), 0.5)
        if frame_i % 5 == 0:
            light = light_status(frame)
        frame_i += 1
        put_text(frame, f"LIGHT: {light[0]} (mean {light[1]:.0f}, clip {100 * light[2]:.1f}%)", (10, 98), 0.5,
                 LIGHT_COLORS[light[0]])
        sw = getattr(cam, "sw", None)
        if sw is not None:
            put_text(frame, f"SW bright {sw.brightness_actual:+d} contrast x{sw.contrast_factor:.2f} "
                            f"gamma {sw.gamma_value:.2f}", (10, 118), 0.45)
        put_text(frame, "CAMERA TEST (tanpa checkerboard)  |  q: keluar", (10, h - 12), 0.5, (0, 255, 255))
        cv2.imshow(win, frame)
        if meter.fps > 0:
            all_fps.append(meter.fps)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyWindow(win)
    if all_fps:
        print(f"[INFO] Camera test selesai. FPS rata-rata {sum(all_fps) / len(all_fps):.1f}, "
              f"min {min(all_fps):.1f}, max {max(all_fps):.1f}.")


# --------------------------------------------------------------------------------------
# Preview / test undistort
# --------------------------------------------------------------------------------------
def draw_grid(img: np.ndarray, step: int = 80):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)


def run_test(cam: CameraBase, calib: IntrinsicCalibration, display_scale: float, allow_back: bool) -> str:
    """Tampilkan ORIGINAL | UNDISTORTED. Return 'quit' atau 'back'."""
    win = "TEST - ORIGINAL | UNDISTORTED"
    cv2.namedWindow(win)
    und = Undistorter(calib)
    grid, fails, warned = False, 0, False
    meter = FpsMeter()
    print("[INFO] Mode preview: ORIGINAL | UNDISTORTED. 'g' = grid, 's' = screenshot, 'q' = keluar" +
          (", 'b' = kembali kumpulkan sample" if allow_back else ""))

    while True:
        fr = cam.read()
        if fr is None:
            fails += 1
            if fails > 100:
                print("[ERROR] Kamera tidak mengirim frame.")
                return "quit"
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                cv2.destroyWindow(win)
                return "quit"
            continue
        fails = 0
        meter.tick()
        frame = fr.color
        mismatch = not und.matches(frame)
        left = frame.copy()
        if mismatch:
            cur = (frame.shape[1], frame.shape[0])
            if not warned:
                print_resolution_mismatch(tuple(calib.image_size), cur)
                warned = True
            right = frame.copy()
            put_text(right, "CALIBRATION RESOLUTION MISMATCH", (10, 60), 0.6, (0, 0, 255), 2)
            put_text(right, f"Calibration : {calib.image_size[0]}x{calib.image_size[1]}", (10, 84), 0.55,
                     (0, 0, 255), 1)
            put_text(right, f"Current     : {cur[0]}x{cur[1]}", (10, 106), 0.55, (0, 0, 255), 1)
            put_text(right, "Undistort dinonaktifkan.", (10, 128), 0.55, (0, 0, 255), 1)
        else:
            right = und.apply(frame)
        if grid:
            draw_grid(left)
            draw_grid(right)
        put_text(left, "ORIGINAL", (10, 28), 0.8, (0, 255, 255), 2)
        put_text(right, "UNDISTORTED", (10, 28), 0.8, (0, 255, 0), 2)
        put_text(left, f"FPS: {meter.fps:.1f}", (10, 52), 0.55, (0, 255, 0))
        put_text(left, f"RMS {calib.rms_error:.3f}px | mean {calib.mean_reprojection_error:.3f}px | "
                       f"N={calib.num_samples}", (10, frame.shape[0] - 12), 0.5)
        combo = np.hstack([left, right])
        s = min(display_scale, 1800.0 / combo.shape[1])     # muat di layar
        show(win, combo, s)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            return "quit"
        if key == ord("g"):
            grid = not grid
        elif key == ord("s"):
            name = time.strftime("undistort_test_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, combo)
            print(f"[OK] Screenshot disimpan: {name}")
        elif key == ord("b") and allow_back:
            cv2.destroyWindow(win)
            return "back"


# --------------------------------------------------------------------------------------
# Mode collect
# --------------------------------------------------------------------------------------
def run_collect(cam: CameraBase, args, width: int, height: int):
    pattern = (args.cols, args.rows)
    objp = make_object_points(args.cols, args.rows, args.square_size)
    board = {"cols_inner_corners": args.cols, "rows_inner_corners": args.rows,
             "cols_squares": args.cols + 1, "rows_squares": args.rows + 1,
             "square_size_mm": args.square_size}
    win = "CALIBRATION - kumpulkan sample checkerboard"
    cv2.namedWindow(win)

    objpoints, imgpoints, poses = [], [], []
    size: Optional[Tuple[int, int]] = None
    msg, msg_until, msg_color = "", 0.0, (255, 255, 255)
    last_seen = time.time()
    fails = 0
    meter = FpsMeter()
    frame_idx = 0
    found, corners, pose = False, None, None     # hasil deteksi preview terakhir (dipakai ulang antar-frame)
    det_ms = 0.0

    def say(text, color=(255, 255, 255), sec=3.0):
        nonlocal msg, msg_until, msg_color
        msg, msg_until, msg_color = text, time.time() + sec, color
        print("[INFO]", text)

    print(__doc__.split("Kontrol (mode collect):")[1].split("Kontrol (preview/test):")[0].strip("\n"))

    while True:
        fr = cam.read()
        if fr is None:
            fails += 1
            if fails > 100:
                sys.exit("[ERROR] Kamera tidak mengirim frame.")
            cv2.waitKey(1)
            continue
        fails = 0
        meter.tick()
        frame = fr.color
        h, w = frame.shape[:2]
        if size is None:
            size = (w, h)
            if size != (width, height):
                print(f"[WARNING] Resolusi diminta {width}x{height} tetapi kamera memberi {w}x{h}. "
                      f"Calibration akan berlaku untuk {w}x{h}; gunakan resolusi ini di semua program.")
        if (w, h) != size:
            say(f"Resolusi frame berubah ke {w}x{h} (calibration {size[0]}x{size[1]})! Sample diabaikan.",
                (0, 0, 255))
            cv2.waitKey(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # murah di 640x480; dipakai preview & sample final

        # --- deteksi preview: ringan, tidak setiap frame, tanpa cornerSubPix ---------------------
        if frame_idx % args.detect_every == 0:
            t0 = time.perf_counter()
            found, corners = detect_preview(gray, pattern, args.detect_scale)
            dt = (time.perf_counter() - t0) * 1000.0
            det_ms = dt if det_ms == 0.0 else 0.9 * det_ms + 0.1 * dt
            if found:
                last_seen = time.time()
                pose = describe_pose(corners, pattern, size)
            else:
                corners, pose = None, None
        frame_idx += 1

        # Gambar langsung di frame (tanpa copy): sample final selalu dihitung ulang dari `gray` bersih.
        view = frame
        if found and corners is not None:
            cv2.drawChessboardCorners(view, pattern, corners, True)
        else:
            if time.time() - last_seen > NOT_FOUND_WARN_SEC:
                put_wrapped(view, f"[WARNING] Checkerboard tidak ditemukan. Cek --cols {args.cols} --rows "
                                  f"{args.rows} (INNER corner), pencahayaan, dan seluruh papan harus terlihat.",
                            10, h - 70, 0.45, (0, 0, 255))

        # --- HUD ----------------------------------------------------------------------------------
        put_text(view, f"FPS: {meter.fps:.1f}  (detect {det_ms:.1f} ms)", (10, 22), 0.55, (0, 255, 0))
        put_text(view, f"Resolution: {size[0]}x{size[1]}", (10, 44), 0.55)
        n = len(objpoints)
        put_text(view, f"Samples: {n}/{args.target_samples}", (10, 66), 0.6,
                 (0, 255, 0) if n >= args.min_samples else (0, 200, 255), 2)
        if found:
            put_text(view, "Board: DETECTED", (10, 90), 0.6, (0, 255, 0), 2)
        else:
            put_text(view, "Board: NOT DETECTED", (10, 90), 0.6, (0, 0, 255), 2)
        put_text(view, f"Squares {args.cols + 1}x{args.rows + 1} | inner {args.cols}x{args.rows} | "
                       f"{args.square_size:g} mm", (10, 112), 0.45)
        covered = coverage(poses)
        x0, y0 = w - 150, 22
        put_text(view, "Cakupan:", (x0, y0), 0.5, (0, 255, 255))
        for i, tag in enumerate(COVERAGE_ORDER):
            ok = tag in covered
            put_text(view, f"[{'x' if ok else ' '}] {tag}", (x0, y0 + 20 * (i + 1)), 0.5,
                     (0, 255, 0) if ok else (200, 200, 200))
        if pose is not None:
            put_text(view, "Pose: " + (", ".join(sorted(pose_tags(pose))) or "-"), (10, 134), 0.45, (255, 255, 0))
        if time.time() < msg_until:
            put_wrapped(view, msg, 10, h - 30, 0.5, msg_color, 1)
        put_text(view, "SPASI/s: simpan  f: paksa  u: undo  r: reset  c: calibrate  q: keluar", (10, h - 10), 0.45)
        show(win, view, args.display_scale)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            return None
        elif key in (ord(" "), ord("s"), ord("f")):
            if not found:
                say("Tidak bisa simpan: checkerboard tidak terdeteksi.", (0, 0, 255))
                continue
            # Sample FINAL: deteksi ulang pada frame resolusi penuh + cornerSubPix (bukan hasil preview).
            ok_final, final_corners = find_corners(gray, pattern)
            if not ok_final:
                say("Papan hilang/blur saat menyimpan. Tahan papan diam lalu coba lagi.", (0, 0, 255))
                continue
            fpose = describe_pose(final_corners, pattern, size)
            dup = find_similar(fpose, poses)
            if dup is not None and key != ord("f"):
                say(f"Posisi terlalu mirip sample #{dup + 1}. Ubah posisi/jarak/kemiringan (atau 'f' untuk paksa).",
                    (0, 165, 255))
                continue
            objpoints.append(objp.copy())
            imgpoints.append(final_corners.copy())
            poses.append(fpose)
            say(f"Sample #{len(objpoints)} disimpan ({', '.join(sorted(pose_tags(fpose))) or '-'}).", (0, 255, 0))
        elif key == ord("u") and objpoints:
            objpoints.pop(); imgpoints.pop(); poses.pop()
            say(f"Sample terakhir dihapus. Sisa {len(objpoints)}.")
        elif key == ord("r"):
            objpoints.clear(); imgpoints.clear(); poses.clear()
            say("Semua sample dihapus.", (0, 165, 255))
        elif key == ord("c"):
            if len(objpoints) < MIN_SAMPLES_HARD:
                say(f"Sample terlalu sedikit ({len(objpoints)}). Minimal {MIN_SAMPLES_HARD}, "
                    f"disarankan >= {args.min_samples}.", (0, 0, 255), 4.0)
                continue
            cv2.destroyWindow(win)
            calib, warns = run_calibration(objpoints, imgpoints, size, board, cam.info(), poses, args)
            print_result(calib, warns)
            npz_path, json_path = calib.save(args.output)
            print(f"[OK] Disimpan: {npz_path} dan {json_path}")
            res = run_test(cam, calib, args.display_scale, allow_back=True)
            if res == "quit":
                return calib
            cv2.namedWindow(win)        # 'b' -> lanjut kumpulkan sample


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TAHAP 1 - Intrinsic camera calibration (checkerboard)")
    p.add_argument("--mode", choices=["collect", "test", "camera-test"], default="collect")
    p.add_argument("--camera-index", type=int, default=0, help="index kamera (0, 1, 2 ...)")
    p.add_argument("--cols", type=int, default=7, help="jumlah INNER corner horizontal (papan 8x8 kotak -> 7)")
    p.add_argument("--rows", type=int, default=7, help="jumlah INNER corner vertikal (papan 8x8 kotak -> 7)")
    p.add_argument("--square-size", type=float, default=34.0, help="ukuran satu kotak checkerboard (mm), BUKAN pixel")
    p.add_argument("--width", type=int, default=None,
                   help=f"default: {WEBCAM_DEFAULT_WIDTH} (collect/camera-test) / sesuai calibration (test)")
    p.add_argument("--height", type=int, default=None,
                   help=f"default: {WEBCAM_DEFAULT_HEIGHT} (collect/camera-test) / sesuai calibration (test)")
    p.add_argument("--fps", type=int, default=WEBCAM_DEFAULT_FPS)
    p.add_argument("--output", default="camera_calibration", help="nama dasar file (.npz dan .json)")
    p.add_argument("--min-samples", type=int, default=15, help="jumlah sample minimum yang disarankan")
    p.add_argument("--target-samples", type=int, default=20, help="target jumlah sample (hanya untuk tampilan HUD)")
    p.add_argument("--max-error", type=float, default=1.0, help="batas error (px) sebelum diberi WARNING")
    p.add_argument("--no-lock-focus", action="store_true", help="biarkan autofocus aktif (TIDAK disarankan)")
    p.add_argument("--focus", type=int, default=None, help="nilai fokus manual (opsional, tergantung driver)")
    p.add_argument("--exposure", type=float, default=None, help="nilai exposure manual (opsional, tergantung driver)")
    p.add_argument("--display-scale", type=float, default=1.0, help="skala jendela tampilan")
    p.add_argument("--detect-every", type=int, default=2,
                   help="deteksi checkerboard preview tiap N frame (1 = tiap frame)")
    p.add_argument("--detect-scale", type=float, default=1.0,
                   help="skala citra HANYA untuk deteksi preview (mis. 0.75). Sample final tetap resolusi penuh")
    p.add_argument("--no-threaded", action="store_true",
                   help="matikan capture thread 'frame terbaru' (pakai read() langsung)")
    args = p.parse_args(argv)
    args.detect_every = max(1, args.detect_every)
    args.detect_scale = min(1.0, max(0.25, args.detect_scale))
    return args


def main(argv=None):
    args = parse_args(argv)
    calib = None
    if args.mode == "test":
        try:
            calib = IntrinsicCalibration.load(args.output)
        except FileNotFoundError as e:
            sys.exit(f"[ERROR] {e}")
        width = args.width or calib.image_size[0]
        height = args.height or calib.image_size[1]
    else:
        width, height = args.width or WEBCAM_DEFAULT_WIDTH, args.height or WEBCAM_DEFAULT_HEIGHT

    print_banner(args, width, height)
    if args.mode == "collect" and (width, height) != WORKING_RESOLUTION:
        print(f"[WARNING] Resolusi kerja proyek adalah {WORKING_RESOLUTION[0]}x{WORKING_RESOLUTION[1]}. "
              f"Calibration pada {width}x{height} hanya valid bila SEMUA program berikutnya "
              f"(field_calibration, vision_lapangan) memakai resolusi yang sama.")

    try:
        # OPEN -> load camera_settings.json -> apply hardware -> verify -> software -> warm-up
        cam = open_camera("webcam", args.camera_index, width, height, args.fps,
                          threaded=not args.no_threaded, lock_focus=not args.no_lock_focus,
                          focus=args.focus, exposure=args.exposure)
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")
    if cam.actual_size != (width, height):
        print(f"WARNING:\nRequested : {width}x{height} @ {args.fps}\n"
              f"Actual    : {cam.actual_size[0]}x{cam.actual_size[1]}\n")
    if not args.no_lock_focus and args.mode != "camera-test":
        print("[INFO] Autofocus dicoba dimatikan. Jangan ubah fokus lensa setelah calibration.")

    try:
        if args.mode == "camera-test":
            run_camera_test(cam)
        elif args.mode == "test":
            if cam.actual_size != tuple(calib.image_size):
                print_resolution_mismatch(tuple(calib.image_size), cam.actual_size)
            run_test(cam, calib, args.display_scale, allow_back=False)
        else:
            run_collect(cam, args, width, height)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
