"""
field_manager.py
=================
CASE 1 - SELECT / MANAGE FIELD.  Mendukung LEBIH DARI SATU lapangan.

Isi file ini (dipindah dari vision_lapangan.py, perilaku deteksi kandidat TIDAK diubah):
    order_corners, clean_mask, Candidate, FieldDetector   pencari kandidat lapangan (edge/otsu/warna)
    Field                                                 satu lapangan (polygon dalam PIXEL)
Baru:
    FieldStore       banyak Field + fields.json (+ impor otomatis field.json lama)
    FieldManagerUI   mesin status pilih kandidat -> ENTER konfirmasi -> simpan/hapus/clear
    run_field_manager(...)   loop kamera + mouse + keyboard (dipanggil dari main.py)

Alur status sebuah area:
    CANDIDATE --klik--> SELECTED --ENTER--> CONFIRMED (belum disimpan) --S--> ACTIVE (tersimpan)
                           |ESC
                           v
                        CANDIDATE
Klik TIDAK PERNAH langsung menyimpan field. Hapus field / clear all selalu minta konfirmasi.

Koordinat: semua polygon disimpan dalam PIXEL (tanpa homography, tanpa mm).
Image space ("raw" atau "undistorted") ikut disimpan di fields.json supaya Case 2 memakai ruang
citra yang sama dengan saat lapangan dipilih.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from calib_io import IntrinsicCalibration, Undistorter

FIELDS_FILE = "fields.json"
LEGACY_FIELD_FILE = "field.json"
FIELDS_VERSION = 2
FIELD_METHODS = ["edge", "otsu", "warna"]      # metode pencarian kandidat lapangan
IMAGE_RAW, IMAGE_UNDISTORTED = "raw", "undistorted"

REDETECT_EVERY = 6          # kandidat dihitung ulang tiap N frame saat tidak ada pilihan (mode LIVE)
DUPLICATE_IOU = 0.8         # kandidat yang hampir sama dengan field terkonfirmasi disembunyikan
IGNORE_ERODE_PX = 7         # erosi mask saat mengukur warna dasar lapangan

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Warna overlay (BGR). Status SELALU juga ditulis sebagai teks, tidak hanya warna.
COL_CANDIDATE = (0, 220, 255)     # kuning
COL_SELECTED = (0, 255, 0)        # hijau
COL_CONFIRMED = (0, 165, 255)     # oranye (belum disimpan)
COL_ACTIVE = (255, 200, 0)        # biru muda (tersimpan, aktif)
COL_WARN = (0, 0, 255)            # merah


# ======================================================================================
# Helper tampilan (dipakai juga oleh vision_lapangan.py)
# ======================================================================================
def put_text(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_panel(view, header: str, lines: List[Tuple[str, Tuple[int, int, int]]]):
    """Panel teks semi-transparan di atas video (font otomatis mengecil bila terlalu panjang)."""
    w = view.shape[1]
    row = 20
    n = len(lines) + 1
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w, 8 + row * n), (0, 0, 0), -1)
    view[:] = cv2.addWeighted(overlay, 0.6, view, 0.4, 0)
    y = 18
    for text, col in [(header, (0, 255, 255))] + lines:
        scale = 0.5
        tw = cv2.getTextSize(text, FONT, scale, 1)[0][0]
        if tw > w - 12:
            scale *= (w - 12) / tw
        put_text(view, text, (6, y), scale, col)
        y += row


def draw_footer(view, text):
    lines = [text] if isinstance(text, str) else list(text)
    h, w = view.shape[:2]
    for i, t in enumerate(reversed(lines)):
        scale = 0.45
        tw = cv2.getTextSize(t, FONT, scale, 1)[0][0]
        if tw > w - 12:
            scale *= (w - 12) / tw
        put_text(view, t, (8, h - 10 - 18 * i), scale)


# ======================================================================================
# Geometri & pencari kandidat (dipindah dari vision_lapangan.py)
# ======================================================================================
def order_corners(pts) -> np.ndarray:
    """Urutkan 4 titik -> kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def clean_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Buang noise kecil (open) lalu tutup lubang kecil (close)."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker), cv2.MORPH_CLOSE, ker)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Candidate:
    corners: np.ndarray
    area: float
    rect: Tuple[int, int, int, int]


class FieldDetector:
    """Mencari kandidat area lapangan (kontur besar yang berbentuk segi-4)."""

    def __init__(self, max_candidates: int = 9):
        self.max_candidates = max_candidates

    def find(self, bgr: np.ndarray, method: str) -> List[Candidate]:
        h, w = bgr.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        if method == "edge":                       # berdasarkan tepi
            m = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
        elif method == "otsu":                     # berdasarkan terang/gelap
            _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m = clean_mask(m, 7)
        else:                                      # berdasarkan area berwarna (saturasi tinggi)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            m = clean_mask(cv2.inRange(hsv, (0, 60, 40), (179, 255, 255)), 9)

        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_a, max_a = 0.02 * w * h, 0.95 * w * h
        cands: List[Candidate] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_a <= area <= max_a):
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
            else:                                  # bukan segi-4 -> pakai kotak miring terkecil
                pts = cv2.boxPoints(cv2.minAreaRect(c))
            corners = order_corners(pts)
            cands.append(Candidate(corners, area, cv2.boundingRect(corners.astype(np.int32))))

        cands.sort(key=lambda c: c.area, reverse=True)
        unique: List[Candidate] = []
        for c in cands:                            # buang kandidat yang hampir kembar
            if all(_iou(c.rect, u.rect) < 0.8 for u in unique):
                unique.append(c)
            if len(unique) >= self.max_candidates:
                break
        return unique


# ======================================================================================
# Field + FieldStore
# ======================================================================================
@dataclass
class Field:
    """Satu lapangan (ROI) dalam PIXEL. `corners` = 4x2 (TL, TR, BR, BL)."""
    corners: np.ndarray
    frame_size: Tuple[int, int]               # (w, h) citra tempat polygon ini dipilih
    ignore_colors: List[str]                  # warna dasar lapangan yang diabaikan di Case 2
    id: int = 0
    active: bool = True                       # False -> dilewati oleh "ALL FIELDS" di Case 2
    saved: bool = False                       # True = sudah ada di fields.json

    def mask(self, shape, erode_px: int = 0) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(m, self.corners.astype(np.int32), 255)
        if erode_px > 0:
            m = cv2.erode(m, np.ones((erode_px * 2 + 1,) * 2, np.uint8))
        return m

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return cv2.boundingRect(self.corners.astype(np.int32))

    @property
    def area(self) -> float:
        return float(cv2.contourArea(self.corners.astype(np.float32)))

    def contains(self, pt) -> bool:
        return cv2.pointPolygonTest(self.corners.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0

    def scaled_to(self, w: int, h: int) -> "Field":
        fw, fh = self.frame_size
        if (fw, fh) == (w, h):
            return self
        return Field(self.corners * np.array([w / fw, h / fh], np.float32), (w, h), self.ignore_colors,
                     self.id, self.active, self.saved)

    def to_dict(self) -> Dict:
        return {"id": self.id, "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in self.corners],
                "active": self.active, "ignore_colors": list(self.ignore_colors)}


class FieldStore:
    """Kumpulan lapangan + file fields.json. Tidak pernah menulis file kecuali save() dipanggil."""

    def __init__(self, fields: Optional[List[Field]] = None, image_space: str = IMAGE_RAW,
                 frame_size: Optional[Tuple[int, int]] = None):
        self.fields: List[Field] = fields or []
        self.image_space = image_space
        self.frame_size = frame_size
        self.source = "none"                  # "fields.json" | "legacy field.json" | "none"
        self.imported_legacy = False
        self.load_error = ""

    # ---------------------------------------------------------------- muat
    @staticmethod
    def load(path: str = FIELDS_FILE, legacy_path: str = LEGACY_FIELD_FILE) -> "FieldStore":
        store = FieldStore()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                fs = tuple(d.get("frame_size") or (0, 0))
                store.frame_size = fs if fs != (0, 0) else None
                store.image_space = d.get("image_space", IMAGE_RAW)
                for e in d.get("fields", []):
                    store.fields.append(Field(np.array(e["polygon"], np.float32), store.frame_size or (0, 0),
                                              list(e.get("ignore_colors", [])), int(e["id"]),
                                              bool(e.get("active", True)), saved=True))
                store.source = path
            except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError) as e:
                store.fields, store.load_error = [], f"{path} tidak bisa dibaca ({e})"
            return store
        if os.path.exists(legacy_path):                       # kompatibel dengan format lama (1 lapangan)
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                fs = tuple(d["frame_size"])
                store.fields = [Field(np.array(d["corners"], np.float32), fs, list(d.get("ignore_colors", [])),
                                      1, True, saved=False)]
                store.frame_size, store.source, store.imported_legacy = fs, "legacy " + legacy_path, True
            except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError) as e:
                store.load_error = f"{legacy_path} tidak bisa dibaca ({e})"
        return store

    # ---------------------------------------------------------------- simpan
    def save(self, path: str = FIELDS_FILE) -> int:
        d = {"version": FIELDS_VERSION, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "coordinate_mode": "pixel", "image_space": self.image_space,
             "frame_size": list(self.frame_size) if self.frame_size else None,
             "fields": [f.to_dict() for f in sorted(self.fields, key=lambda f: f.id)]}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)
        for f in self.fields:
            f.saved = True
        self.imported_legacy = False
        return len(self.fields)

    # ---------------------------------------------------------------- kelola
    def next_free_id(self) -> int:
        used = {f.id for f in self.fields}
        i = 1
        while i in used:
            i += 1
        return i

    def add(self, f: Field) -> Field:
        f.id = self.next_free_id()
        f.saved = False
        self.fields.append(f)
        self.fields.sort(key=lambda x: x.id)
        if self.frame_size is None:
            self.frame_size = f.frame_size
        return f

    def get(self, fid: int) -> Optional[Field]:
        return next((f for f in self.fields if f.id == fid), None)

    def remove(self, fid: int) -> bool:
        n = len(self.fields)
        self.fields = [f for f in self.fields if f.id != fid]
        return len(self.fields) < n

    def clear(self):
        self.fields = []

    def active_fields(self) -> List[Field]:
        return [f for f in sorted(self.fields, key=lambda f: f.id) if f.active]

    @property
    def unsaved(self) -> bool:
        return self.imported_legacy or any(not f.saved for f in self.fields)


# ======================================================================================
# Kalibrasi opsional + ruang citra
# ======================================================================================
def load_calibration_if_available(path: str = "camera_calibration", quiet: bool = False
                                  ) -> Optional[IntrinsicCalibration]:
    """Muat camera_calibration.npz bila ada. Tanpa file -> None (PIXEL TEST MODE), tidak error."""
    try:
        calib = IntrinsicCalibration.load(path)
    except FileNotFoundError:
        if not quiet:
            print("[INFO] Camera calibration not loaded.")
            print("[INFO] Running vision in PIXEL TEST MODE.")
        return None
    except Exception as e:                                   # file rusak: jangan crash
        if not quiet:
            print(f"[WARNING] Camera calibration gagal dimuat ({e}).")
            print("[INFO] Running vision in PIXEL TEST MODE.")
        return None
    if not quiet:
        print(f"[INFO] Camera calibration loaded: {calib.image_size[0]}x{calib.image_size[1]}, "
              f"RMS {calib.rms_error:.3f}px, {calib.num_samples} sample.")
    return calib


def prepare_frame(color: np.ndarray, und: Optional[Undistorter], use_undistort: bool) -> np.ndarray:
    """Terapkan undistort HANYA bila diminta dan resolusi sesuai calibration."""
    if use_undistort and und is not None and und.matches(color):
        return und.apply(color)
    return color


# ======================================================================================
# UI state machine
# ======================================================================================
class FieldManagerUI:
    """Logika Select/Manage Field tanpa ketergantungan pada window (mudah diuji)."""

    def __init__(self, store: FieldStore, classifier, calib: Optional[IntrinsicCalibration] = None,
                 path: str = FIELDS_FILE):
        self.store, self.classifier, self.calib, self.path = store, classifier, calib, path
        self.finder = FieldDetector()
        self.method_i = 0
        self.cands: List[Candidate] = []
        self.sel: Optional[int] = None
        self.frozen = False
        self.mode = "normal"        # normal | remove_pick | remove_confirm | clear_confirm | quit_confirm
        self.pending_remove: Optional[int] = None
        self.msg, self.msg_until, self.msg_color = "", 0.0, (255, 255, 255)
        self.frame_i = 0
        self.dirty = store.unsaved
        self.exit_requested = False
        self.use_undistort = bool(calib is not None and store.image_space == IMAGE_UNDISTORTED)
        self.space_warning = ""
        if store.image_space == IMAGE_UNDISTORTED and calib is None and store.fields:
            self.space_warning = "fields.json dibuat pada citra UNDISTORTED tetapi calibration tidak ada!"

    # ---------------------------------------------------------------- util
    def say(self, text: str, color=(255, 255, 255), sec: float = 3.5):
        self.msg, self.msg_until, self.msg_color = text, time.time() + sec, color
        print("[INFO]", text)

    @property
    def method(self) -> str:
        return FIELD_METHODS[self.method_i]

    def redetect(self, bgr: np.ndarray):
        found = self.finder.find(bgr, self.method)
        existing = [f.rect for f in self.store.fields]
        self.cands = [c for c in found if all(_iou(c.rect, r) < DUPLICATE_IOU for r in existing)]

    def tick(self, bgr: np.ndarray):
        """Dipanggil tiap frame: hitung ulang kandidat (mode LIVE) tiap REDETECT_EVERY frame."""
        if not self.frozen and (self.frame_i % REDETECT_EVERY == 0):
            self.redetect(bgr)
        self.frame_i += 1

    # ---------------------------------------------------------------- mouse
    def click(self, pt: Tuple[float, float]):
        if self.mode == "normal":
            hits = [i for i, c in enumerate(self.cands) if cv2.pointPolygonTest(c.corners, pt, False) >= 0]
            if hits:
                self.sel = min(hits, key=lambda i: self.cands[i].area)
                self.frozen = True                  # kandidat dibekukan selama ada pilihan
                self.say(f"SELECTED CANDIDATE {self.sel + 1}  (ENTER = konfirmasi, ESC = batal)",
                         COL_SELECTED, 6.0)
            else:
                hit_field = next((f for f in self.store.fields if f.contains(pt)), None)
                self.say("Tidak ada candidate di titik ini." +
                         (f" (FIELD {hit_field.id}: pakai X untuk menghapus)" if hit_field else ""),
                         (200, 200, 200))
        elif self.mode == "remove_pick":
            hits = [f for f in self.store.fields if f.contains(pt)]
            if hits:
                f = min(hits, key=lambda f: f.area)
                self.pending_remove, self.mode = f.id, "remove_confirm"
                self.say(f"Remove FIELD {f.id}?  ENTER = YES, ESC = CANCEL", COL_WARN, 8.0)
            else:
                self.say("Klik di dalam FIELD yang ingin dihapus (ESC = batal).", (200, 200, 200))

    # ---------------------------------------------------------------- keyboard
    def key(self, k: int, bgr: np.ndarray) -> Optional[str]:
        """Proses satu tombol. Return 'exit' bila harus kembali ke Main Menu."""
        if k in (-1, 255):
            return None
        enter, esc = k in (13, 10), k == 27
        ch = chr(k).lower() if 32 <= k < 127 else ""
        m = self.mode

        if m == "clear_confirm":
            if enter:
                n = len(self.store.fields)
                self.store.clear()
                self.dirty, self.mode = True, "normal"
                self.sel, self.frozen = None, False
                self.say(f"{n} field dihapus (belum disimpan - tekan S untuk menyimpan).", COL_WARN)
            elif esc or ch == "q":
                self.mode = "normal"
                self.say("Clear all dibatalkan.")
        elif m == "remove_confirm":
            if enter:
                ok = self.store.remove(self.pending_remove)
                self.dirty, self.mode = True, "normal"
                self.say(f"FIELD {self.pending_remove} removed (belum disimpan - tekan S)." if ok
                         else "Field tidak ditemukan.", COL_WARN)
                self.pending_remove = None
            elif esc or ch == "q":
                self.mode, self.pending_remove = "normal", None
                self.say("Remove dibatalkan.")
        elif m == "quit_confirm":
            if ch == "s":
                self.save()
                return "exit"
            if enter:
                return "exit"
            if esc or ch == "q":
                self.mode = "normal"
                self.say("Tetap di Select / Manage Field.")
        elif m == "remove_pick":
            if esc or ch in ("x", "q"):
                self.mode = "normal"
                self.say("Remove mode dibatalkan.")
        else:                                                   # normal
            if esc:
                if self.sel is not None:
                    n = self.sel + 1
                    self.sel, self.frozen = None, False
                    self.say(f"Selection CANDIDATE {n} dibatalkan.")
                else:
                    self.say("Tidak ada selection. (Tekan Q untuk kembali ke Main Menu)", (200, 200, 200))
            elif enter:
                if self.sel is None:
                    self.say("Belum ada candidate yang dipilih (klik candidate dulu).", (200, 200, 200))
                else:
                    self.confirm_selection(bgr)
            elif ch == "q":
                if self.dirty or self.store.unsaved:
                    self.mode = "quit_confirm"
                    self.say("Ada field BELUM disimpan: S = simpan & keluar | ENTER = keluar tanpa simpan "
                             "| ESC = tetap", COL_WARN, 10.0)
                else:
                    return "exit"
            elif "1" <= ch <= "9":
                i = int(ch) - 1
                if i < len(self.cands):
                    self.sel, self.frozen = i, True
                    self.say(f"SELECTED CANDIDATE {i + 1}  (ENTER = konfirmasi, ESC = batal)", COL_SELECTED, 6.0)
            elif ch == " ":
                self.frozen = not self.frozen
                if not self.frozen:
                    self.sel = None
            elif ch == "m":
                self.method_i, self.frozen, self.sel = (self.method_i + 1) % len(FIELD_METHODS), False, None
                self.redetect(bgr)
            elif ch == "r":
                self.frozen, self.sel = False, None
                self.redetect(bgr)
                self.say(f"Candidate dideteksi ulang ({len(self.cands)} ditemukan).")
            elif ch == "s":
                self.save()
            elif ch == "x":
                if not self.store.fields:
                    self.say("Belum ada FIELD yang dikonfirmasi.", (200, 200, 200))
                else:
                    self.mode, self.sel, self.frozen = "remove_pick", None, False
                    self.say("REMOVE FIELD MODE: klik FIELD yang ingin dihapus (ESC = batal).", COL_WARN, 8.0)
            elif ch == "c":
                if not self.store.fields:
                    self.say("Tidak ada FIELD untuk dihapus.", (200, 200, 200))
                else:
                    self.mode, self.sel, self.frozen = "clear_confirm", None, False
                    self.say("CLEAR ALL FIELD?  ENTER = YES, ESC = CANCEL", COL_WARN, 8.0)
            elif ch == "u":
                self.toggle_undistort()
        return None

    # ---------------------------------------------------------------- aksi
    def confirm_selection(self, bgr: np.ndarray):
        c = self.cands[self.sel]
        h, w = bgr.shape[:2]
        fld = Field(c.corners.copy(), (w, h), [])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # warna dasar lapangan (mis. lantai putih) otomatis diabaikan di Case 2
        fld.ignore_colors = self.classifier.dominant(hsv, fld.mask(bgr.shape, erode_px=IGNORE_ERODE_PX))
        self.store.add(fld)
        self.store.image_space = IMAGE_UNDISTORTED if self.use_undistort else IMAGE_RAW
        self.dirty = True
        n, self.sel, self.frozen = self.sel + 1, None, False
        self.say(f"FIELD {fld.id} CONFIRMED (dari candidate {n}). Tekan S untuk menyimpan.", COL_CONFIRMED, 5.0)
        if fld.ignore_colors:
            print(f"[INFO] FIELD {fld.id}: warna dasar diabaikan: {', '.join(fld.ignore_colors)} "
                  f"(aktifkan lagi di Case 2 dengan tombol 1-5)")
        self.redetect(bgr)

    def save(self):
        try:
            n = self.store.save(self.path)
            self.dirty = False
            self.say(f"[OK] {n} fields saved to {self.path}", (0, 255, 0), 4.0)
        except OSError as e:
            self.say(f"Gagal menyimpan {self.path}: {e}", COL_WARN, 5.0)

    def toggle_undistort(self):
        if self.calib is None:
            self.say("Undistort tidak tersedia: camera_calibration.npz belum ada.", (200, 200, 200))
        elif self.store.fields:
            self.say("Ganti image space butuh semua field dihapus dulu (C), karena polygon "
                     "dipilih pada ruang citra tertentu.", COL_WARN, 6.0)
        else:
            self.use_undistort = not self.use_undistort
            self.store.image_space = IMAGE_UNDISTORTED if self.use_undistort else IMAGE_RAW
            self.say(f"Image space: {self.store.image_space.upper()}")

    # ---------------------------------------------------------------- gambar
    def draw(self, view: np.ndarray, fps: float = 0.0):
        h, w = view.shape[:2]
        pend = self.pending_remove if self.mode == "remove_confirm" else None
        for f in self.store.fields:                                    # field terkonfirmasi / aktif
            col = COL_ACTIVE if f.saved else COL_CONFIRMED
            if pend == f.id:
                col = COL_WARN
            cv2.polylines(view, [f.corners.astype(np.int32)], True, col, 3)
            cx, cy = f.corners.mean(axis=0).astype(int)
            tag = "REMOVE?" if pend == f.id else ("ACTIVE" if f.saved else "CONFIRMED (unsaved)")
            if f.saved and not f.active:
                tag = "INACTIVE"
            put_text(view, f"FIELD {f.id} {tag}", (max(4, cx - 70), cy), 0.5, col, 1)
        if self.mode in ("normal",):
            for i, c in enumerate(self.cands):                          # kandidat
                is_sel = i == self.sel
                col = COL_SELECTED if is_sel else COL_CANDIDATE
                cv2.polylines(view, [c.corners.astype(np.int32)], True, col, 4 if is_sel else 2)
                x, y, bw, bh = c.rect
                cv2.rectangle(view, (x, y), (x + bw, y + bh), col, 1)
                cx, cy = c.corners.mean(axis=0).astype(int)
                cv2.circle(view, (cx, cy), 16, (0, 0, 0), -1)
                put_text(view, str(i + 1), (cx - 8, cy + 7), 0.7, col, 2)
                label = f"SELECTED CANDIDATE {i + 1}" if is_sel else f"CANDIDATE {i + 1}"
                put_text(view, label, (x + 4, max(16, y + 16)), 0.5, col)

        n_unsaved = sum(1 for f in self.store.fields if not f.saved)
        head = (f"FIELD MANAGER | fields: {len(self.store.fields)} ({n_unsaved} unsaved) | "
                f"candidates: {len(self.cands)} | metode: {self.method} | {'BEKU' if self.frozen else 'LIVE'} | "
                f"image: {'UNDISTORTED' if self.use_undistort else 'RAW'} | COORD: PIXEL | FPS {fps:.0f}")
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        if self.mode == "normal":
            if self.sel is not None:
                lines.append((f"SELECTED CANDIDATE {self.sel + 1} -> ENTER = jadikan FIELD {self.store.next_free_id()}"
                              f"  |  ESC = batal", COL_SELECTED))
            else:
                lines.append(("Klik candidate (atau tekan 1-9) untuk memilih. Belum dianggap FIELD sebelum ENTER.",
                              (255, 255, 255)))
        elif self.mode == "remove_pick":
            lines.append(("REMOVE FIELD MODE: klik FIELD yang ingin dihapus   |   ESC = batal", COL_WARN))
        elif self.mode == "remove_confirm":
            lines.append((f"Remove FIELD {self.pending_remove}?    ENTER = YES    ESC = CANCEL", COL_WARN))
        elif self.mode == "clear_confirm":
            lines.append(("CLEAR ALL FIELD?    ENTER = YES    ESC = CANCEL", COL_WARN))
        elif self.mode == "quit_confirm":
            lines.append(("FIELD BELUM DISIMPAN!   S = simpan & keluar | ENTER = keluar tanpa simpan | "
                          "ESC = tetap", COL_WARN))
        if self.space_warning:
            lines.append(("[WARNING] " + self.space_warning, COL_WARN))
        if time.time() < self.msg_until:
            lines.append((self.msg, self.msg_color))
        draw_panel(view, head, lines)
        draw_footer(view, ["Klik: pilih | ENTER: konfirmasi | ESC: batal | S: simpan | X: hapus | C: clear all | "
                           "R: deteksi ulang | M: metode | SPASI: beku | U: undistort | Q: menu",
                           "Status: CANDIDATE(kuning) SELECTED(hijau) CONFIRMED(oranye, belum disimpan) "
                           "ACTIVE(biru, tersimpan)"])


def run_field_manager(cam, store: FieldStore, classifier, calib: Optional[IntrinsicCalibration] = None,
                      display_scale: float = 1.5, path: str = FIELDS_FILE) -> str:
    """Loop UI Select / Manage Field. Return 'back'. Kamera dikelola oleh pemanggil (main.py)."""
    win = "SELECT / MANAGE FIELD (Case 1)"
    cv2.namedWindow(win)
    ui = FieldManagerUI(store, classifier, calib, path)
    und = Undistorter(calib) if calib is not None else None
    scale = max(0.5, display_scale)
    clicks: List[Tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x / scale, y / scale))              # ke koordinat citra asli

    cv2.setMouseCallback(win, on_mouse)
    if store.load_error:
        print("[WARNING]", store.load_error)
    if store.imported_legacy:
        print(f"[INFO] {store.source} diimpor sebagai FIELD 1 - tekan S untuk menyimpannya ke {path}.")
    print(f"[INFO] Select / Manage Field: {len(store.fields)} field dimuat dari {store.source}.")

    fails, rescaled = 0, False
    t_prev, fps = time.perf_counter(), 0.0
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
            bgr = prepare_frame(fr.color, und, ui.use_undistort)
            h, w = bgr.shape[:2]
            if not rescaled:                                    # polygon dari resolusi lain -> skala + peringatan
                rescaled = True
                if store.fields and any(f.frame_size != (w, h) for f in store.fields):
                    print(f"[WARNING] Field dibuat pada resolusi berbeda dari frame sekarang {w}x{h}. "
                          f"Polygon diskalakan - periksa hasilnya lalu simpan ulang (S).")
                    store.fields = [f.scaled_to(w, h) for f in store.fields]
                    store.frame_size = (w, h)
                    ui.dirty = True
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 / dt if fps and dt > 0 else (1.0 / dt if dt > 0 else 0.0)

            while clicks:
                ui.click(clicks.pop(0))
            ui.tick(bgr)
            view = bgr.copy()
            ui.draw(view, fps)
            if abs(scale - 1.0) > 1e-3:
                view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(win, view)
            if ui.key(cv2.waitKey(1) & 0xFF, bgr) == "exit":
                return "back"
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
