"""
vision_lapangan.py
===================
CASE 2 - DETEKSI OBJEK di dalam lapangan (ROI) + utilitas vision lama.

Pipeline (mode PIXEL TEST, calibration/homography BELUM wajib):

    Camera (+ camera_settings.json)
      -> Field ROI (satu / semua field dari fields.json)
      -> HSV color mask (MERAH, BIRU, HIJAU, HITAM, PUTIH - HSV lama dipertahankan)
      -> contour -> filter bentuk: area, 4 vertex, convex, aspect, rectangularity
      -> inside-ratio (min. 95% area objek harus di dalam field)
      -> SQUARE / RECTANGLE
      -> temporal stability (DETECTING -> VALID setelah N frame berturut-turut)
      -> Object ID, Field ID, Color, Shape, Pixel X,Y, Width, Height, Status

Yang TIDAK dilakukan (disengaja):
    - tidak ada koordinat mm / homography / transform robot / PLC / UR3.
    - tidak memakai warna kulit sebagai filter; tangan/lengan ditolak oleh validasi GEOMETRI
      (4 vertex, convex, rectangularity, area, aspect, inside-ratio, stability).
    - ini BUKAN sistem keselamatan robot. `workspace_obstructed` hanya flag informasi.

Pembagian modul:
    camera_source.py    kelas kamera (webcam / RealSense)        <- sebelumnya terduplikasi di sini
    camera_settings.py  open_camera() + setting tersimpan
    field_manager.py    Field, FieldDetector, FieldStore, UI Case 1  <- sebelumnya di sini
    vision_lapangan.py  (file ini) deteksi objek Case 2 + CLI lama `python vision_lapangan.py`
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from calib_io import IntrinsicCalibration, Undistorter
from camera_settings import ConfiguredCamera, light_status, open_camera
from camera_source import CameraBase
from field_manager import (FIELDS_FILE, IMAGE_RAW, IMAGE_UNDISTORTED, Field, FieldStore, clean_mask,
                           draw_footer, draw_panel, load_calibration_if_available, prepare_frame,
                           put_text, run_field_manager)

# =====================================================================================
# 1. KONFIGURASI
# =====================================================================================

# Rentang HSV OpenCV (H: 0-179, S: 0-255, V: 0-255). Silakan tuning sesuai pencahayaan.
COLOR_RANGES = {
    "merah": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (179, 255, 255))],  # merah di 2 ujung hue
    "biru":  [((95, 100, 50), (130, 255, 255))],
    "hijau": [((35, 70, 50), (85, 255, 255))],
    "hitam": [((0, 0, 0), (179, 255, 50))],      # V sangat rendah
    "putih": [((0, 0, 180), (179, 40, 255))],    # S rendah + V tinggi
}
COLOR_ORDER = ["merah", "biru", "hijau", "hitam", "putih"]
COLOR_EN = {"merah": "RED", "biru": "BLUE", "hijau": "GREEN", "hitam": "BLACK", "putih": "WHITE"}
COLOR_DRAW = {  # warna gambar (BGR)
    "merah": (0, 0, 255), "biru": (255, 100, 0), "hijau": (0, 200, 0),
    "hitam": (150, 150, 150), "putih": (255, 255, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

AUTO_MAX_AREA_FRACTION = 0.6    # --max-area 0 -> batas atas = 60% luas field (menolak ring garis tepi field)
MOTION_DIFF_THRESHOLD = 25      # beda gray (setelah buang perubahan kecerahan global) yang dianggap "gerak"
MOTION_SCALE = 0.25             # deteksi gerak pada citra diperkecil (hemat CPU)
MAX_PANEL_OBJECTS = 8           # jumlah objek yang ditulis di panel layar


@dataclass
class Config:
    source: str = "auto"              # auto | realsense | webcam
    index: int = 1                    # index webcam (Logitech C270 = 1 pada PC ini)
    width: int = 640
    height: int = 480
    fps: int = 30
    start_case: int = 1
    field_file: str = FIELDS_FILE
    load_field: bool = False
    field_width_cm: Optional[float] = None   # ukuran lapangan nyata (opsional, hanya CLI lama)
    field_height_cm: Optional[float] = None
    min_area: float = 150             # luas minimum objek (piksel)
    margin: int = 6                   # margin dalam field agar garis tepi tidak terdeteksi
    display_scale: float = 1.5        # pembesaran tampilan jendela
    focal_px: Optional[float] = None  # (opsional, tanpa depth) focal length piksel
    object_width_cm: Optional[float] = None  # (opsional, tanpa depth) lebar objek nyata
    # ---- filter bentuk Case 2 (semua bisa diubah lewat CLI, tidak ada magic number tersembunyi) ----
    max_area: float = 0.0             # luas maksimum objek (px); 0 = otomatis 60% luas field
    min_rectangularity: float = 0.85  # luas_kontur / luas_minAreaRect
    square_tolerance: float = 0.10    # SQUARE bila sisi_panjang/sisi_pendek <= 1 + toleransi
    min_inside_ratio: float = 0.95    # minimal bagian luas objek yang berada di dalam field
    stable_frames: int = 5            # frame berturut-turut sebelum VALID
    approx_epsilon: float = 0.04      # epsilon approxPolyDP sebagai pecahan keliling
    max_aspect: float = 5.0           # sisi_panjang/sisi_pendek maksimum (menolak garis tipis)
    match_dist: float = 30.0          # jarak centroid maksimum antar frame untuk dianggap objek yang sama
    max_missed: int = 5               # frame hilang yang ditoleransi untuk objek yang sudah VALID
    outside_margin: int = 25          # objek dicari sampai N px di luar field agar yang keluar ketahuan
    obstruct_ratio: float = 0.04      # bagian field yang berubah/berisi kontur non-valid -> OBSTRUCTED
    obstruct_hold: int = 8            # frame OBSTRUCTED ditahan setelah gangguan hilang
    no_threaded: bool = False         # matikan capture thread frame-terbaru


def add_arguments(p: argparse.ArgumentParser):
    """Argumen yang sama dipakai CLI lama (python vision_lapangan.py) dan main.py."""
    p.add_argument("--source", choices=["auto", "realsense", "webcam"], default="auto")
    p.add_argument("--index", type=int, default=1, help="index webcam (C270 = 1)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--case", dest="start_case", type=int, choices=[1, 2], default=1)
    p.add_argument("--field-file", default=FIELDS_FILE)
    p.add_argument("--load-field", action="store_true", help="muat fields.json dan langsung ke Case 2")
    p.add_argument("--field-width", dest="field_width_cm", type=float, default=None)
    p.add_argument("--field-height", dest="field_height_cm", type=float, default=None)
    p.add_argument("--min-area", type=float, default=150)
    p.add_argument("--max-area", type=float, default=0.0, help="px; 0 = otomatis (60%% luas field)")
    p.add_argument("--margin", type=int, default=6)
    p.add_argument("--display-scale", type=float, default=1.5)
    p.add_argument("--focal-px", type=float, default=None, help="estimasi jarak tanpa depth (butuh --object-width)")
    p.add_argument("--object-width", dest="object_width_cm", type=float, default=None)
    p.add_argument("--min-rectangularity", type=float, default=0.85)
    p.add_argument("--square-tolerance", type=float, default=0.10)
    p.add_argument("--min-inside-ratio", type=float, default=0.95)
    p.add_argument("--stable-frames", type=int, default=5)
    p.add_argument("--approx-epsilon", type=float, default=0.04)
    p.add_argument("--max-aspect", type=float, default=5.0)
    p.add_argument("--match-dist", type=float, default=30.0)
    p.add_argument("--max-missed", type=int, default=5)
    p.add_argument("--outside-margin", type=int, default=25)
    p.add_argument("--obstruct-ratio", type=float, default=0.04)
    p.add_argument("--obstruct-hold", type=int, default=8)
    p.add_argument("--no-threaded", action="store_true", help="matikan capture thread frame-terbaru")


def config_from_namespace(ns) -> Config:
    names = set(Config.__dataclass_fields__)
    return Config(**{k: v for k, v in vars(ns).items() if k in names})


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Deteksi objek warna di dalam lapangan (CLI lama; "
                                            "gunakan main.py untuk menu lengkap)")
    add_arguments(p)
    return config_from_namespace(p.parse_args(argv))


# =====================================================================================
# 2. KAMERA (kelas kamera ada di camera_source.py; pembukaan + settings di camera_settings.py)
# =====================================================================================

def open_camera_from_config(cfg: Config) -> ConfiguredCamera:
    """RealSense bila diminta/ada (source auto), selain itu webcam. Selalu memuat camera_settings.json."""
    if cfg.source in ("auto", "realsense"):
        try:
            cam = open_camera("realsense", cfg.index, None, None, cfg.fps, verbose=False)
            print(f"[INFO] Kamera: {cam.name} (depth aktif)")
            return cam
        except Exception as e:
            if cfg.source == "realsense":
                raise RuntimeError(f"RealSense gagal: {e}")
            print(f"[INFO] RealSense tidak tersedia ({e}) -> memakai webcam.")
    cam = open_camera("webcam", cfg.index, cfg.width, cfg.height, cfg.fps, threaded=not cfg.no_threaded)
    print("[INFO] Kamera: Webcam (tanpa depth)")
    return cam


# =====================================================================================
# 3. KLASIFIKASI WARNA (HSV)  - tidak diubah
# =====================================================================================

class ColorClassifier:
    def __init__(self, ranges=COLOR_RANGES):
        self.ranges = ranges

    def mask(self, hsv: np.ndarray, name: str) -> np.ndarray:
        out = None
        for lo, hi in self.ranges[name]:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            out = m if out is None else cv2.bitwise_or(out, m)
        return out

    def dominant(self, hsv: np.ndarray, region: np.ndarray, min_share: float = 0.4) -> List[str]:
        """Warna yang menutupi >= min_share dari region (dipakai untuk mendeteksi warna dasar lapangan)."""
        total = max(1, cv2.countNonZero(region))
        return [n for n in COLOR_ORDER
                if cv2.countNonZero(cv2.bitwise_and(self.mask(hsv, n), region)) / total >= min_share]


# =====================================================================================
# 4. FILTER BENTUK (SQUARE / RECTANGLE)
# =====================================================================================

@dataclass
class ShapeParams:
    min_area: float = 150.0
    approx_epsilon: float = 0.04
    min_rectangularity: float = 0.85
    square_tolerance: float = 0.10
    max_aspect: float = 5.0


@dataclass
class ShapeResult:
    ok: bool
    reason: str = ""
    shape: str = ""             # SQUARE | RECTANGLE
    area: float = 0.0
    width: float = 0.0          # sisi panjang minAreaRect (px)
    height: float = 0.0         # sisi pendek minAreaRect (px)
    angle: float = 0.0          # orientasi sisi panjang, derajat [0,180) (untuk tahap pick nanti)
    aspect: float = 0.0
    rectangularity: float = 0.0
    trivial: bool = False       # True bila ditolak hanya karena terlalu kecil (noise) -> tidak dicatat


def evaluate_contour(c: np.ndarray, p: ShapeParams, max_area: float) -> ShapeResult:
    """Urutan validasi: area -> min/max -> arcLength -> approxPolyDP(4) -> convex -> minAreaRect
    -> aspect ratio -> rectangularity -> SQUARE/RECTANGLE."""
    area = float(cv2.contourArea(c))                                            # 1
    if area < p.min_area:                                                       # 2
        return ShapeResult(False, f"area {area:.0f} < min {p.min_area:g}", area=area, trivial=True)
    if max_area > 0 and area > max_area:                                        # 3
        return ShapeResult(False, f"area {area:.0f} > max {max_area:.0f}", area=area)
    peri = float(cv2.arcLength(c, True))                                        # 4
    if peri <= 0:
        return ShapeResult(False, "perimeter 0", area=area)
    approx = cv2.approxPolyDP(c, p.approx_epsilon * peri, True)                 # 5
    if len(approx) != 4:                                                        # 6
        return ShapeResult(False, f"vertices {len(approx)} != 4", area=area)
    if not cv2.isContourConvex(approx):                                         # 7
        return ShapeResult(False, "not convex", area=area)
    (_, _), (rw, rh), ang = cv2.minAreaRect(c)                                  # 8
    long_s, short_s = max(rw, rh), min(rw, rh)
    if short_s < 1:
        return ShapeResult(False, "degenerate rect", area=area)
    aspect = long_s / short_s                                                   # 9
    if aspect > p.max_aspect:
        return ShapeResult(False, f"aspect {aspect:.1f} > {p.max_aspect:g}", area=area, aspect=aspect)
    rectangularity = area / (rw * rh)                                           # 10
    if rectangularity < p.min_rectangularity:
        return ShapeResult(False, f"rectangularity {rectangularity:.2f} < {p.min_rectangularity:g}",
                           area=area, aspect=aspect, rectangularity=rectangularity)
    angle = (ang + (90.0 if rw < rh else 0.0)) % 180.0
    shape = "SQUARE" if aspect <= 1.0 + p.square_tolerance else "RECTANGLE"
    return ShapeResult(True, "", shape, area, long_s, short_s, angle, aspect, rectangularity)


# =====================================================================================
# 5. DETEKSI OBJEK (hanya di sekitar field)
# =====================================================================================

@dataclass
class FieldMasks:
    """Mask per field, dihitung SEKALI (field statis)."""
    field: Field
    poly: np.ndarray            # polygon penuh
    eroded: np.ndarray          # polygon diperkecil `margin` px (area valid)
    dilated: np.ndarray         # polygon diperbesar `outside_margin` px (agar yang keluar field ketahuan)
    box: Tuple[int, int, int, int]   # x0, y0, x1, y1 dari dilated
    area: float


def build_field_masks(field: Field, shape, margin: int, outside_margin: int) -> FieldMasks:
    poly = field.mask(shape)
    eroded = field.mask(shape, erode_px=margin)
    k = outside_margin * 2 + 1
    dilated = cv2.dilate(poly, np.ones((k, k), np.uint8)) if outside_margin > 0 else poly
    ys, xs = np.nonzero(dilated)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if xs.size else (0, 0, 1, 1)
    return FieldMasks(field, poly, eroded, dilated, box, max(1.0, field.area))


@dataclass
class Detection:
    field_id: int
    color: str
    shape: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    width: float
    height: float
    angle: float
    area: float
    rectangularity: float
    aspect: float
    inside_ratio: float


@dataclass
class Rejected:
    field_id: int
    color: str
    contour: np.ndarray
    reason: str
    area: float


class ObjectDetector:
    def __init__(self, classifier: ColorClassifier, shape: ShapeParams, max_area: float, min_inside_ratio: float):
        self.classifier = classifier
        self.shape = shape
        self.max_area = max_area
        self.min_inside_ratio = min_inside_ratio

    @staticmethod
    def inside_ratio(c: np.ndarray, fm: FieldMasks) -> float:
        x, y, w, h = cv2.boundingRect(c)
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1, offset=(-x, -y))
        total = cv2.countNonZero(m)
        if total == 0:
            return 0.0
        return cv2.countNonZero(cv2.bitwise_and(m, fm.eroded[y:y + h, x:x + w])) / total

    def detect(self, hsv: np.ndarray, masks: List[FieldMasks], enabled: Dict[int, Dict[str, bool]]
               ) -> Tuple[List[Detection], List[Rejected]]:
        dets: List[Detection] = []
        rejs: List[Rejected] = []
        cmask_cache: Dict[str, np.ndarray] = {}
        for fm in masks:
            fid = fm.field.id
            x0, y0, x1, y1 = fm.box
            max_area = self.max_area if self.max_area > 0 else AUTO_MAX_AREA_FRACTION * fm.area
            for name in COLOR_ORDER:
                if not enabled.get(fid, {}).get(name):
                    continue
                if name not in cmask_cache:
                    cmask_cache[name] = self.classifier.mask(hsv, name)
                sub = cv2.bitwise_and(cmask_cache[name][y0:y1, x0:x1], fm.dilated[y0:y1, x0:x1])
                sub = clean_mask(sub, 5)
                contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
                for c in contours:
                    r = evaluate_contour(c, self.shape, max_area)
                    if r.trivial:
                        continue
                    if not r.ok:
                        rejs.append(Rejected(fid, name, c, r.reason, r.area))
                        continue
                    ratio = self.inside_ratio(c, fm)
                    if ratio < self.min_inside_ratio:
                        rejs.append(Rejected(fid, name, c, f"inside {ratio:.2f} < {self.min_inside_ratio:g}", r.area))
                        continue
                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        continue
                    dets.append(Detection(fid, name, r.shape, c, cv2.boundingRect(c),
                                          (M["m10"] / M["m00"], M["m01"] / M["m00"]), r.width, r.height,
                                          r.angle, r.area, r.rectangularity, r.aspect, ratio))
        dets.sort(key=lambda d: (d.field_id, round(d.centroid[1] / 20), d.centroid[0]))   # atas->bawah, kiri->kanan
        return dets, rejs


# =====================================================================================
# 6. TEMPORAL STABILITY
# =====================================================================================

@dataclass
class Track:
    tid: int
    field_id: int
    color: str
    det: Detection
    hits: int = 1
    missed: int = 0
    oid: Optional[int] = None         # Object ID: HANYA diberikan setelah VALID (tangan/noise tidak pernah dapat ID)

    @property
    def status(self) -> str:
        return "VALID" if self.oid is not None else "DETECTING"


class ObjectTracker:
    """DETECTING (hits < stable_frames) -> VALID (konsisten stable_frames frame berturut-turut).
    Objek yang hanya muncul sebentar dibuang tanpa pernah mendapat Object ID."""

    def __init__(self, stable_frames: int, match_dist: float, max_missed: int):
        self.stable_frames = max(1, stable_frames)
        self.match_dist, self.max_missed = match_dist, max_missed
        self.reset()

    def reset(self):
        self.tracks: List[Track] = []
        self._next_tid = 1
        self._next_oid = 1

    def update(self, dets: List[Detection]) -> List[Track]:
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, d in enumerate(dets):
                if d.field_id == tr.field_id and d.color == tr.color:
                    dist = math.hypot(d.centroid[0] - tr.det.centroid[0], d.centroid[1] - tr.det.centroid[1])
                    if dist <= self.match_dist:
                        pairs.append((dist, ti, di))
        pairs.sort()
        mt, md = set(), set()
        for _, ti, di in pairs:
            if ti in mt or di in md:
                continue
            mt.add(ti)
            md.add(di)
            tr = self.tracks[ti]
            tr.det, tr.hits, tr.missed = dets[di], tr.hits + 1, 0
        keep: List[Track] = []
        for ti, tr in enumerate(self.tracks):
            if ti in mt:
                keep.append(tr)
                continue
            tr.missed += 1
            if tr.oid is not None and tr.missed <= self.max_missed:      # VALID bertahan sebentar
                keep.append(tr)
            # DETECTING yang hilang 1 frame -> dibuang (harus berturut-turut)
        for di, d in enumerate(dets):
            if di not in md:
                keep.append(Track(self._next_tid, d.field_id, d.color, d))
                self._next_tid += 1
        self.tracks = keep
        for tr in keep:
            if tr.oid is None and tr.hits >= self.stable_frames:
                tr.oid = self._next_oid
                self._next_oid += 1
        return self.visible()

    def visible(self) -> List[Track]:
        return [t for t in self.tracks if t.missed == 0 or t.oid is not None]


# =====================================================================================
# 7. WORKSPACE OBSTRUCTED  (informasi saja - BUKAN sistem keselamatan)
# =====================================================================================

class ObstructionMonitor:
    """Field dianggap OBSTRUCTED bila (a) ada perubahan/gerakan besar di dalam field, atau (b) ada
    kontur berwarna besar yang BUKAN square/rectangle valid. Ditahan `hold` frame setelah gangguan hilang."""

    def __init__(self, ratio: float, hold: int):
        self.ratio, self.hold = ratio, max(0, hold)
        self.reset()

    def reset(self):
        self.prev: Optional[np.ndarray] = None
        self.left: Dict[int, int] = {}
        self.small_masks: Dict[int, np.ndarray] = {}
        self.info: Dict[int, str] = {}

    def update(self, gray: np.ndarray, masks: List[FieldMasks], rejected: List[Rejected]) -> Dict[int, bool]:
        small = cv2.GaussianBlur(cv2.resize(gray, None, fx=MOTION_SCALE, fy=MOTION_SCALE,
                                            interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)
        small -= float(small.mean())                          # buang perubahan kecerahan global (auto-exposure)
        diff = None
        if self.prev is not None and self.prev.shape == small.shape:
            diff = (np.abs(small - self.prev) > MOTION_DIFF_THRESHOLD).astype(np.uint8)
        self.prev = small
        out: Dict[int, bool] = {}
        for fm in masks:
            fid = fm.field.id
            if fid not in self.small_masks:
                self.small_masks[fid] = cv2.resize(fm.eroded, (small.shape[1], small.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
            sm = self.small_masks[fid]
            denom = max(1, cv2.countNonZero(sm))
            motion = cv2.countNonZero(cv2.bitwise_and(diff, sm)) / denom if diff is not None else 0.0
            rej_area = sum(r.area for r in rejected if r.field_id == fid)
            rej = rej_area / fm.area
            flag = motion > self.ratio or rej > self.ratio
            if flag:
                self.left[fid] = self.hold
                self.info[fid] = f"motion {100 * motion:.1f}% | non-valid contour {100 * rej:.1f}%"
            else:
                self.left[fid] = max(0, self.left.get(fid, 0) - 1)
            out[fid] = self.left[fid] > 0 or flag
        return out


# =====================================================================================
# 8. GEOMETRI FIELD & JARAK (fitur lama, tetap dipertahankan; hanya aktif bila ada ukuran/depth)
# =====================================================================================

def sample_depth(depth: np.ndarray, x: float, y: float, r: int = 4) -> Optional[float]:
    """Median depth valid di sekitar piksel (x, y)."""
    h, w = depth.shape
    roi = depth[max(0, int(y) - r):min(h, int(y) + r + 1), max(0, int(x) - r):min(w, int(x) + r + 1)]
    v = roi[(roi > 0.1) & (roi < 10)]
    return float(np.median(v)) if v.size else None


class FieldGeometry:
    """
    Menghitung jarak objek ke 4 sisi lapangan.
    Lapangan dianggap bidang datar berbentuk persegi panjang. Posisi objek dipetakan
    (homography) ke koordinat normal (u, v) 0..1, lalu dikali ukuran lapangan.
    Ukuran lapangan diambil dari: argumen --field-width/--field-height, atau dihitung
    dari depth (jarak 3D antar sudut), atau tidak ada -> hasil dalam persen.
    (Dalam menu main.py fitur ini tidak aktif kecuali RealSense/depth ada.)
    """

    def __init__(self, field: Field, cfg: Config):
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        self.H = cv2.getPerspectiveTransform(field.corners.astype(np.float32), dst)
        self.field = field
        self.size_m: Optional[Tuple[float, float]] = None   # (lebar, tinggi) meter
        self.fixed = False
        if cfg.field_width_cm and cfg.field_height_cm:
            self.size_m = (cfg.field_width_cm / 100.0, cfg.field_height_cm / 100.0)
            self.fixed = True

    def update_size(self, cam: CameraBase, depth: Optional[np.ndarray], fmask: np.ndarray):
        """Perkiraan ukuran lapangan dari depth (di-smoothing agar stabil)."""
        if self.fixed or depth is None or not cam.has_depth:
            return
        valid = depth[(fmask > 0) & (depth > 0.1) & (depth < 10)]
        fallback = float(np.median(valid)) if valid.size else None
        pts = []
        for x, y in self.field.corners:
            z = sample_depth(depth, x, y, 5) or fallback
            p = cam.deproject(x, y, z) if z else None
            if p is None:
                return
            pts.append(p)
        tl, tr, br, bl = pts
        w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        new = (float(w), float(h))
        self.size_m = new if self.size_m is None else (0.9 * self.size_m[0] + 0.1 * new[0],
                                                       0.9 * self.size_m[1] + 0.1 * new[1])

    def side_distances(self, pt: Tuple[float, float]) -> Tuple[Dict[str, float], str]:
        """Return ({'left','right','top','bottom'}, satuan). Satuan 'cm' atau '%'."""
        uv = cv2.perspectiveTransform(np.array([[pt]], np.float32), self.H)[0][0]
        u, v = float(np.clip(uv[0], 0, 1)), float(np.clip(uv[1], 0, 1))
        if self.size_m:
            W, Hh = self.size_m[0] * 100, self.size_m[1] * 100
            unit = "cm"
        else:
            W, Hh, unit = 100.0, 100.0, "%"
        return {"left": u * W, "right": (1 - u) * W, "top": v * Hh, "bottom": (1 - v) * Hh}, unit


def camera_distance_cm(cam: CameraBase, depth: Optional[np.ndarray], obj, cfg: Config) -> Tuple[Optional[float], str]:
    """
    Jarak objek ke kamera (cm). `obj` = Detection (punya contour, bbox, centroid).
    - Ada depth  : median depth di dalam kontur objek -> jarak 3D Euclidean ke kamera.
    - Tanpa depth: estimasi pinhole  Z = focal_px * lebar_nyata / lebar_piksel  (jika diberikan).
    Return (jarak_cm | None, metode).
    """
    if depth is not None and cam.has_depth:
        x, y, w, h = obj.bbox
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [obj.contour], -1, 255, -1, offset=(-x, -y))
        roi = depth[y:y + h, x:x + w]
        v = roi[(m > 0) & (roi > 0.1) & (roi < 10)]
        if v.size >= 5:
            z = float(np.median(v))
            p = cam.deproject(obj.centroid[0], obj.centroid[1], z)
            return (float(np.linalg.norm(p)) if p is not None else z) * 100.0, "depth"
        return None, "depth-kosong"
    if cfg.focal_px and cfg.object_width_cm and obj.bbox[2] > 0:
        return cfg.focal_px * cfg.object_width_cm / obj.bbox[2], "estimasi"
    return None, "n/a"


# =====================================================================================
# 9. FORMAT OUTPUT + VISUALISASI
# =====================================================================================

def format_object_row(t: Track, pick_ok: bool) -> str:
    """Satu baris ringkas untuk panel layar (mode PIXEL)."""
    d = t.det
    return (f"Object ID: {t.oid} | Field ID: {t.field_id} | Color: {COLOR_EN[t.color]} | Shape: {d.shape} | "
            f"Pixel: ({d.centroid[0]:.0f},{d.centroid[1]:.0f}) | Width: {d.width:.0f} px | "
            f"Height: {d.height:.0f} px | Status: {t.status}" + ("" if pick_ok else " | NO PICK"))


def format_object_block(t: Track, pick_ok: bool, extra: str = "") -> str:
    d = t.det
    s = (f"Object ID : {t.oid}\nField ID  : {t.field_id}\nColor     : {COLOR_EN[t.color]}\n"
         f"Shape     : {d.shape}\nPixel     : ({d.centroid[0]:.0f},{d.centroid[1]:.0f})\n"
         f"Width     : {d.width:.0f} px\nHeight    : {d.height:.0f} px\nStatus    : {t.status}\n"
         f"Pick cand.: {'YES' if pick_ok else 'NO (WORKSPACE OBSTRUCTED)'}")
    return s + (("\n" + extra) if extra else "")


def format_object_line(o, sides: Dict[str, float], unit: str, cam_cm: Optional[float], method: str) -> str:
    """Format lama: Object 1 | Warna: .. | Left: .. cm | Right: .. | Top: .. | Bottom: .. | Camera: .. cm"""
    cam_txt = "N/A" if cam_cm is None else f"{'~' if method == 'estimasi' else ''}{cam_cm:.1f} cm"
    return (f"Object {o.id} | Warna: {COLOR_EN.get(o.color, o.color)} | Left: {sides['left']:.1f} {unit} | "
            f"Right: {sides['right']:.1f} {unit} | Top: {sides['top']:.1f} {unit} | "
            f"Bottom: {sides['bottom']:.1f} {unit} | Camera: {cam_txt}")


class Visualizer:
    def __init__(self, scale: float):
        self.scale = max(0.5, scale)

    def enlarge(self, view: np.ndarray) -> np.ndarray:
        if abs(self.scale - 1.0) < 1e-3:
            return view
        return cv2.resize(view, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

    def draw_field(self, view, field: Field, obstructed: bool):
        col = (0, 0, 255) if obstructed else (255, 255, 0)
        cv2.polylines(view, [field.corners.astype(np.int32)], True, col, 2)
        x, y, _, _ = field.rect
        put_text(view, f"FIELD {field.id}" + ("  WORKSPACE OBSTRUCTED" if obstructed else ""),
                 (x + 4, max(14, y - 6)), 0.5, col)

    def draw_tracks(self, view, tracks: List[Track], pick_ok: Dict[int, bool], stable_frames: int):
        for t in tracks:
            d = t.det
            x, y, w, h = d.bbox
            cx, cy = int(d.centroid[0]), int(d.centroid[1])
            if t.oid is None:                                                   # DETECTING
                cv2.drawContours(view, [d.contour], -1, (170, 170, 170), 1)
                put_text(view, f"DETECTING {t.hits}/{stable_frames}", (x, max(14, y - 6)), 0.45, (170, 170, 170))
                continue
            col = COLOR_DRAW[t.color]
            cv2.drawContours(view, [d.contour], -1, col, 2)
            cv2.rectangle(view, (x, y), (x + w, y + h), col, 1)
            cv2.circle(view, (cx, cy), 4, (0, 0, 255), -1)
            put_text(view, f"#{t.oid} F{t.field_id} {COLOR_EN[t.color]} {d.shape}"
                           + ("" if pick_ok.get(t.field_id, True) else " (NO PICK)"),
                     (x, max(14, y - 6)), 0.5, col)
            put_text(view, f"({cx},{cy})", (cx + 6, cy + 14), 0.4, (255, 255, 255))

    def draw_rejected(self, view, rejs: List[Rejected]):
        for r in rejs:
            cv2.drawContours(view, [r.contour], -1, (255, 0, 255), 1)
            x, y, _, _ = cv2.boundingRect(r.contour)
            put_text(view, r.reason, (x, y + 12), 0.35, (255, 0, 255))

    panel = staticmethod(draw_panel)
    footer = staticmethod(draw_footer)


# =====================================================================================
# 10. CASE 2 - DETEKSI OBJEK
# =====================================================================================

def run_case2(cam: CameraBase, cfg: Config, viz: Visualizer, classifier: ColorClassifier,
              fields: List[Field], calib: Optional[IntrinsicCalibration] = None,
              image_space: str = IMAGE_RAW) -> str:
    """CASE 2 (mode PIXEL): deteksi objek square/rectangle di dalam field terpilih.
    Return 'back' (Q/ESC) atau 'fields' (F: ke Select / Manage Field)."""
    if not fields:
        print("[WARNING] No field selected.")
        return "back"
    win = "CASE 2 - Deteksi Objek di Lapangan (PIXEL TEST MODE)"

    und = Undistorter(calib) if calib is not None else None
    use_undistort = image_space == IMAGE_UNDISTORTED
    if use_undistort and und is None:
        print("[ERROR] fields.json dibuat pada citra UNDISTORTED, tetapi camera_calibration.npz tidak tersedia.\n"
              "        Pilih ulang field di Select / Manage Field (tanpa undistort) atau lakukan calibration.")
        return "back"
    if und is not None and use_undistort and tuple(calib.image_size) != tuple(cam.actual_size):
        print(f"[ERROR] CALIBRATION RESOLUTION MISMATCH: calibration {calib.image_size[0]}x{calib.image_size[1]}, "
              f"kamera {cam.actual_size[0]}x{cam.actual_size[1]}. Undistort tidak dipakai -> dibatalkan.")
        return "back"

    cv2.namedWindow(win)
    shape_params = ShapeParams(cfg.min_area, cfg.approx_epsilon, cfg.min_rectangularity,
                               cfg.square_tolerance, cfg.max_aspect)
    detector = ObjectDetector(classifier, shape_params, cfg.max_area, cfg.min_inside_ratio)
    tracker = ObjectTracker(cfg.stable_frames, cfg.match_dist, cfg.max_missed)
    monitor = ObstructionMonitor(cfg.obstruct_ratio, cfg.obstruct_hold)

    masks: List[FieldMasks] = []
    enabled: Dict[int, Dict[str, bool]] = {}
    geometries: Dict[int, FieldGeometry] = {}
    debug, fails, last_sig = False, 0, None
    t_prev, fps = time.perf_counter(), 0.0
    print("[INFO] CASE 2 - PIXEL TEST MODE. Koordinat dalam PIXEL (bukan mm). Vision ini BUKAN sistem keselamatan.")
    print(f"[INFO] Filter: min_area={cfg.min_area:g} max_area={cfg.max_area:g}(0=auto) "
          f"min_rect={cfg.min_rectangularity:g} sq_tol={cfg.square_tolerance:g} "
          f"inside>={cfg.min_inside_ratio:g} stable={cfg.stable_frames}")

    def print_table(tracks: List[Track], obstructed: Dict[int, bool], header: str):
        print(f"--- {time.strftime('%H:%M:%S')} {header} ---")
        valid = [t for t in tracks if t.oid is not None]
        if not valid:
            print("(tidak ada objek VALID)")
        for t in valid:
            extra = ""
            g = geometries.get(t.field_id)
            if g is not None and g.size_m:
                sides, unit = g.side_distances(t.det.centroid)
                extra = (f"Sides     : L {sides['left']:.1f} R {sides['right']:.1f} "
                         f"T {sides['top']:.1f} B {sides['bottom']:.1f} {unit}")
            if cam.has_depth:
                cam_cm, method = camera_distance_cm(cam, last_depth[0], t.det, cfg)
                if cam_cm is not None:
                    extra += ("\n" if extra else "") + f"Camera    : {cam_cm:.1f} cm ({method})"
            print(format_object_block(t, not obstructed.get(t.field_id, False), extra) + "\n")
        for fid, ob in obstructed.items():
            if ob:
                print(f"[FIELD {fid}] WORKSPACE OBSTRUCTED ({monitor.info.get(fid, '')}) - no NEW PICK candidate")

    last_depth = [None]
    try:
        while True:
            fr = cam.read()
            if fr is None:
                fails += 1
                if fails > 100:
                    print("[ERROR] Kamera tidak mengirim frame.")
                    return "back"
                cv2.waitKey(1)
                continue
            fails = 0
            last_depth[0] = fr.depth
            color = prepare_frame(fr.color, und, use_undistort)
            h, w = color.shape[:2]

            if not masks:                                         # inisialisasi sekali (field statis)
                scaled = []
                for f in fields:
                    if f.frame_size != (w, h):
                        print(f"[WARNING] FIELD {f.id} dibuat pada {f.frame_size[0]}x{f.frame_size[1]}, "
                              f"frame sekarang {w}x{h} -> polygon diskalakan.")
                    scaled.append(f.scaled_to(w, h))
                fields = scaled
                masks = [build_field_masks(f, color.shape, cfg.margin, cfg.outside_margin) for f in fields]
                enabled = {f.id: {c: c not in f.ignore_colors for c in COLOR_ORDER} for f in fields}
                if (cfg.field_width_cm and cfg.field_height_cm) or cam.has_depth:
                    geometries = {f.id: FieldGeometry(f, cfg) for f in fields}

            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            if cam.has_depth:
                for fm in masks:
                    if fm.field.id in geometries:
                        geometries[fm.field.id].update_size(cam, fr.depth, fm.poly)

            dets, rejs = detector.detect(hsv, masks, enabled)
            tracks = tracker.update(dets)
            obstructed = monitor.update(gray, masks, rejs)
            pick_ok = {fid: not ob for fid, ob in obstructed.items()}      # workspace_obstructed -> tidak ada NEW PICK

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 / dt if fps and dt > 0 else (1.0 / dt if dt > 0 else 0.0)

            view = color.copy()
            for fm in masks:
                viz.draw_field(view, fm.field, obstructed.get(fm.field.id, False))
            if debug:
                viz.draw_rejected(view, rejs)
            viz.draw_tracks(view, tracks, pick_ok, cfg.stable_frames)

            valid = [t for t in tracks if t.oid is not None]
            n_pick = sum(1 for t in valid if pick_ok.get(t.field_id, True))
            lines: List[Tuple[str, Tuple[int, int, int]]] = []
            for t in valid[:MAX_PANEL_OBJECTS]:
                lines.append((format_object_row(t, pick_ok.get(t.field_id, True)), COLOR_DRAW[t.color]))
            if len(valid) > MAX_PANEL_OBJECTS:
                lines.append((f"... +{len(valid) - MAX_PANEL_OBJECTS} objek lain (lihat console, tombol P)",
                              (200, 200, 200)))
            if not valid:
                lines.append(("Belum ada objek VALID (square/rectangle stabil di dalam field).", (200, 200, 200)))
            for fid, ob in obstructed.items():
                if ob:
                    lines.append((f"FIELD {fid}: WORKSPACE OBSTRUCTED ({monitor.info.get(fid, '')}) - no NEW PICK",
                                  (0, 0, 255)))
            lab, _, _ = light_status(view)
            head = (f"CASE 2 | PIXEL TEST MODE | image: {'UNDISTORTED' if use_undistort else 'RAW'} | "
                    f"fields: {','.join(str(f.id) for f in fields)} | valid: {len(valid)} | pick cand.: {n_pick} | "
                    f"LIGHT: {lab} | FPS {fps:.0f}")
            out = viz.enlarge(view)
            viz.panel(out, head, lines)
            flags = " ".join(f"{i + 1}:{COLOR_EN[c]}{'*' if any(enabled[f.id][c] for f in fields) else '-'}"
                             for i, c in enumerate(COLOR_ORDER))
            viz.footer(out, [f"{flags} | R: reset tracking | D: debug ditolak | P: print | F: pilih field | Q/ESC: menu",
                             "Vision test saja - BUKAN sistem keselamatan robot"])
            cv2.imshow(win, out)

            sig = (tuple(sorted((t.oid, t.field_id, t.color) for t in valid)),
                   tuple(sorted(fid for fid, ob in obstructed.items() if ob)))
            if sig != last_sig:                                  # log konsol hanya saat ada perubahan
                last_sig = sig
                print_table(tracks, obstructed, "perubahan")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return "back"
            elif key in (ord("f"), ord("F")):
                return "fields"
            elif key in (ord("r"), ord("R")):
                tracker.reset()
                monitor.reset()
                last_sig = None
                print("[INFO] Tracking & stabilitas di-reset (Object ID mulai dari 1).")
            elif key in (ord("d"), ord("D")):
                debug = not debug
                print(f"[INFO] Debug kontur ditolak: {'ON' if debug else 'OFF'}")
            elif key in (ord("p"), ord("P")):
                print_table(tracks, obstructed, "snapshot")
            elif ord("1") <= key <= ord("5"):
                c = COLOR_ORDER[key - ord("1")]
                new = not any(enabled[f.id][c] for f in fields)
                for f in fields:
                    enabled[f.id][c] = new
                print(f"[INFO] Warna {COLOR_EN[c]}: {'ON' if new else 'OFF'}")
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass


# =====================================================================================
# 11. CLI LAMA (python vision_lapangan.py) - memakai fungsi yang SAMA dengan main.py
# =====================================================================================

def main(argv=None):
    cfg = parse_args(argv)
    try:
        cam = open_camera_from_config(cfg)
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")
    viz = Visualizer(cfg.display_scale)
    classifier = ColorClassifier()
    calib = load_calibration_if_available()
    if not cam.has_depth:
        print("[INFO] Tanpa depth: jarak kamera = N/A "
              "(kecuali --focal-px dan --object-width diberikan untuk estimasi).")
    try:
        store = FieldStore.load(cfg.field_file)
        case = cfg.start_case if not cfg.load_field else 2
        while True:
            if case == 1 or not store.active_fields():
                run_field_manager(cam, store, classifier, calib, cfg.display_scale, cfg.field_file)
                if not store.active_fields():
                    break
                case = 2
            else:
                res = run_case2(cam, cfg, viz, classifier, store.active_fields(), calib, store.image_space)
                if res == "fields":
                    case = 1
                else:
                    break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
