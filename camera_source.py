"""
camera_source.py
=================
Lapisan kamera yang dipakai bersama oleh camera_calibration.py, field_calibration.py
dan vision_lapangan.py.

Semua program hanya bergantung pada antarmuka `CameraBase.read() -> Frame`.
Untuk mengganti kamera (mis. ke Intel RealSense D455) cukup ganti kelas yang dibuat
oleh `create_camera()`; program lain tidak perlu ditulis ulang.

Catatan penting Logitech C270:
- Working resolution proyek ini: 640x480 @ 30 FPS (lebih ringan dari 1280x720).
- Memakai DirectShow (cv2.CAP_DSHOW) di Windows agar kamera terbuka stabil.
- MJPG dipakai agar USB tidak kehabisan bandwidth (YUY2 jauh lebih lambat).
- Autofocus DIMATIKAN secara default. Intrinsic calibration hanya berlaku untuk
  satu setting fokus. Jika fokus berubah-ubah (autofocus), hasil calibration tidak valid.
- Properti kamera (FPS, autofocus, exposure, buffer) TIDAK selalu didukung driver.
  Setiap properti di-set, dibaca kembali, dan bila tidak cocok hanya muncul WARNING
  (program tidak crash).
- DirectShow sering mengabaikan CAP_PROP_BUFFERSIZE. Karena itu capture dijalankan di
  thread terpisah yang selalu menyimpan frame TERBARU (frame lama dibuang), sehingga
  preview tidak tertinggal walau pemrosesan sesaat lebih lambat dari kamera.
"""

import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Working resolution Logitech C270 untuk seluruh pipeline (calibration -> field -> vision).
WEBCAM_DEFAULT_WIDTH = 640
WEBCAM_DEFAULT_HEIGHT = 480
WEBCAM_DEFAULT_FPS = 30

# Default RealSense dibiarkan seperti sebelumnya.
REALSENSE_DEFAULT_WIDTH = 1280
REALSENSE_DEFAULT_HEIGHT = 720


@dataclass
class Frame:
    color: np.ndarray                      # citra BGR
    depth: Optional[np.ndarray] = None     # peta depth (meter, float32); None bila kamera tanpa depth


class CameraBase:
    """Antarmuka kamera. Kamera baru cukup menurunkan kelas ini dan mengisi read()."""
    name = "base"
    has_depth = False
    actual_size: Tuple[int, int] = (0, 0)  # (width, height) yang benar-benar dipakai kamera
    intrinsics: Optional[Tuple[float, float, float, float]] = None  # (fx, fy, cx, cy) citra warna

    def read(self) -> Optional[Frame]:
        raise NotImplementedError

    def deproject(self, px: float, py: float, z: float) -> Optional[np.ndarray]:
        """Piksel + kedalaman (m) -> titik 3D (x, y, z) meter. None bila intrinsics tidak tersedia."""
        if self.intrinsics is None:
            return None
        fx, fy, cx, cy = self.intrinsics
        return np.array([(px - cx) / fx * z, (py - cy) / fy * z, z])

    def info(self) -> Dict:
        """Informasi perangkat (disimpan ke file calibration)."""
        return {"name": self.name}

    def close(self):
        pass


def _fourcc_to_str(value: float) -> str:
    try:
        v = int(value)
        s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
        return s if s.isprintable() and s.strip() else "?"
    except Exception:
        return "?"


class WebcamCamera(CameraBase):
    """Kamera USB biasa (Logitech C270). Tanpa depth."""

    def __init__(self, index: int = 0,
                 width: int = WEBCAM_DEFAULT_WIDTH, height: int = WEBCAM_DEFAULT_HEIGHT,
                 fps: int = WEBCAM_DEFAULT_FPS,
                 lock_focus: bool = True, focus: Optional[int] = None, warmup: int = 10,
                 exposure: Optional[float] = None, threaded: bool = True):
        # cv2.VideoCapture TIDAK thread-safe: read() (thread capture) dan set()/get() (UI settings)
        # harus bergantian. Semua akses cap setelah konstruktor melalui _io_lock.
        self._io_lock = threading.RLock()
        self.supports_hw_controls = True
        self._closed = False
        use_dshow = sys.platform.startswith("win")
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Kamera index {index} tidak bisa dibuka. Coba --camera-index lain (0, 1, 2 ...) "
                f"dan pastikan kamera tidak dipakai program lain.")

        self.index = index
        self.backend = "DSHOW" if use_dshow else "default"
        self.requested = (width, height, fps)
        self.warnings: List[str] = []
        self.threaded = threaded
        self._thread: Optional[threading.Thread] = None

        # --- format & ukuran (MJPG dulu, baru ukuran, baru FPS) -------------------------------
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # --- buffer kecil (banyak backend, termasuk DSHOW, mengabaikan ini -> hanya WARNING) ---
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.buffer_reported = self._get(cv2.CAP_PROP_BUFFERSIZE)
        if self.buffer_reported is None or abs(self.buffer_reported - 1) > 0.5:
            self.warnings.append(
                f"CAP_PROP_BUFFERSIZE=1 tidak bisa dipastikan (backend melaporkan {self.buffer_reported}). "
                + ("Capture thread 'frame terbaru' dipakai untuk menghindari delay."
                   if threaded else "Aktifkan capture thread (hapus --no-threaded) bila preview delay."))

        # --- fokus / exposure: set -> baca kembali -> warning bila tidak cocok -----------------
        if lock_focus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            af = self._get(cv2.CAP_PROP_AUTOFOCUS)
            if af is None or abs(af) > 0.5:
                self.warnings.append(
                    f"Autofocus tidak terbukti mati (CAP_PROP_AUTOFOCUS terbaca {af}). Fokus bisa berubah "
                    f"-> calibration tidak valid. Matikan autofocus lewat Logitech Capture / driver bila perlu.")
            if focus is not None:
                self.cap.set(cv2.CAP_PROP_FOCUS, float(focus))
                fo = self._get(cv2.CAP_PROP_FOCUS)
                if fo is None or abs(fo - float(focus)) > 1.0:
                    self.warnings.append(f"CAP_PROP_FOCUS diminta {focus}, terbaca {fo} (tidak didukung driver?).")
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            ex = self._get(cv2.CAP_PROP_EXPOSURE)
            if ex is None or abs(ex - float(exposure)) > 0.5:
                self.warnings.append(f"CAP_PROP_EXPOSURE diminta {exposure}, terbaca {ex} (tidak didukung driver?).")

        # --- warmup + ukur FPS nyata -----------------------------------------------------------
        for _ in range(warmup):            # buang frame awal (auto-exposure belum stabil)
            self.cap.read()
        n_meas, t0, last = 15, time.perf_counter(), None
        for _ in range(n_meas):
            ok, img = self.cap.read()
            if ok and img is not None:
                last = img
        dt = time.perf_counter() - t0
        self.measured_fps = n_meas / dt if dt > 0 else 0.0
        if last is None:
            self.cap.release()
            raise RuntimeError("Kamera terbuka tetapi tidak mengirim frame.")
        self.actual_size = (last.shape[1], last.shape[0])
        self.reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fourcc = _fourcc_to_str(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.name = f"Webcam (index {index}, {self.backend})"

        # --- warning konfigurasi aktual ---------------------------------------------------------
        if self.actual_size != (width, height):
            self.warnings.append(f"Resolusi diminta {width}x{height}, kamera memberi "
                                 f"{self.actual_size[0]}x{self.actual_size[1]}.")
        if self.reported_fps and abs(self.reported_fps - fps) > 1.0:
            self.warnings.append(f"FPS diminta {fps}, driver melaporkan {self.reported_fps:g}.")
        if self.measured_fps < 0.8 * fps:
            self.warnings.append(
                f"FPS terukur saat start hanya {self.measured_fps:.1f} (target {fps}). Penyebab umum: cahaya "
                f"redup (auto-exposure menurunkan FPS), kabel/port USB lambat, atau program lain memakai kamera.")
        if self.fourcc != "MJPG":
            self.warnings.append(f"Format kamera {self.fourcc}, bukan MJPG (bandwidth USB lebih berat).")

        # --- thread capture: selalu simpan frame terbaru ----------------------------------------
        self._cond = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self._read_seq = 0
        self._stop = threading.Event()
        if self.threaded:
            self._latest, self._seq = last, 1
            self._thread = threading.Thread(target=self._capture_loop, name="webcam-capture", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------------------------- util
    def _get(self, prop) -> Optional[float]:
        try:
            with self._io_lock:
                v = float(self.cap.get(prop))
            return None if v != v else v        # NaN -> None
        except Exception:
            return None

    def get_prop(self, prop) -> Optional[float]:
        """Baca properti kamera (None bila gagal). Aman dipanggil saat thread capture berjalan."""
        return self._get(prop)

    def set_prop(self, prop, value) -> bool:
        """Set properti kamera. Return nilai balik cap.set() (False = ditolak backend).
        Hanya menandakan 'diterima', BUKAN 'berhasil' -> selalu verifikasi dengan get_prop()."""
        try:
            with self._io_lock:
                return bool(self.cap.set(prop, value))
        except Exception:
            return False

    def _capture_loop(self):
        while not self._stop.is_set():
            with self._io_lock:
                ok, img = self.cap.read()
            if not ok or img is None:
                self._stop.wait(0.005)          # hanya di thread capture, bukan di loop preview
                continue
            with self._cond:
                self._latest = img              # cap.read() membuat array baru -> aman tanpa copy
                self._seq += 1
                self._cond.notify_all()

    def print_report(self):
        """Tampilkan konfigurasi kamera aktual + semua warning."""
        req_w, req_h, req_fps = self.requested
        print("\n[CAMERA]")
        print(f"Name       : {self.name}")
        print(f"Index      : {self.index}")
        print(f"Requested  : {req_w}x{req_h} @ {req_fps} FPS")
        print(f"Actual     : {self.actual_size[0]}x{self.actual_size[1]}")
        print(f"Actual FPS : {self.reported_fps:g} (driver) | {self.measured_fps:.1f} (terukur)")
        print(f"FOURCC     : {self.fourcc}")
        print(f"Backend    : {self.backend}")
        print(f"Capture    : {'thread, frame terbaru' if self.threaded else 'langsung (blocking)'}")
        for w in self.warnings:
            print("[WARNING]", w)
        print()

    # ---------------------------------------------------------------------------------- API
    def read(self) -> Optional[Frame]:
        if not self.threaded:
            with self._io_lock:
                ok, img = self.cap.read()
            return Frame(img, None) if ok and img is not None else None
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._read_seq, timeout=1.0)
            if self._seq == self._read_seq:     # timeout: tidak ada frame baru
                return None
            self._read_seq = self._seq
            img = self._latest
        return Frame(img, None)

    def info(self) -> Dict:
        return {
            "name": self.name,
            "type": "webcam",
            "index": self.index,
            "backend": self.backend,
            "requested_width_height_fps": list(self.requested),
            "actual_width": self.actual_size[0],
            "actual_height": self.actual_size[1],
            "reported_fps": float(self._get(cv2.CAP_PROP_FPS) or 0.0),
            "measured_fps": float(self.measured_fps),
            "fourcc": self.fourcc,
            "autofocus_reported": self._get(cv2.CAP_PROP_AUTOFOCUS),
            "focus_reported": self._get(cv2.CAP_PROP_FOCUS),
            "threaded_capture": bool(self.threaded),
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        with self._io_lock:
            self.cap.release()


class RealSenseCamera(CameraBase):
    """Intel RealSense (D455, dst). Disiapkan untuk masa depan; depth di-align ke citra warna.
    (Belum diuji pada proyek ini - kamera yang dipakai sekarang adalah C270.)"""
    has_depth = True
    supports_hw_controls = False      # kontrol exposure/focus RealSense (rs.option) belum diimplementasikan

    def __init__(self, width: int = REALSENSE_DEFAULT_WIDTH, height: int = REALSENSE_DEFAULT_HEIGHT,
                 fps: int = 30):
        import pyrealsense2 as rs
        if len(rs.context().query_devices()) == 0:
            raise RuntimeError("Tidak ada perangkat RealSense terdeteksi.")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipe.start(cfg)
        dev = profile.get_device()
        self.name = "RealSense " + dev.get_info(rs.camera_info.name)
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.factory_intrinsics = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        self.intrinsics = self.factory_intrinsics
        self.actual_size = (width, height)
        for _ in range(20):
            self.pipe.wait_for_frames()

    def read(self) -> Optional[Frame]:
        frames = self.align.process(self.pipe.wait_for_frames())
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        if not d or not c:
            return None
        color = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        return Frame(color, depth)

    def info(self) -> Dict:
        return {"name": self.name, "type": "realsense",
                "actual_width": self.actual_size[0], "actual_height": self.actual_size[1],
                "factory_intrinsics_fx_fy_cx_cy": list(self.factory_intrinsics)}

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


# CameraBase default untuk atribut yang dibaca modul lain (supaya aman untuk kamera apa pun)
CameraBase.supports_hw_controls = False


def create_camera(source: str = "webcam", index: int = 0,
                  width: Optional[int] = None, height: Optional[int] = None,
                  fps: int = 30, lock_focus: bool = True, focus: Optional[int] = None,
                  exposure: Optional[float] = None, threaded: bool = True) -> CameraBase:
    """Pabrik kamera. source: 'webcam' | 'realsense'.

    width/height None -> default per kamera:
      webcam (Logitech C270) : 640x480
      realsense              : 1280x720 (tidak diubah)
    """
    if source == "realsense":
        return RealSenseCamera(width or REALSENSE_DEFAULT_WIDTH, height or REALSENSE_DEFAULT_HEIGHT, fps)
    return WebcamCamera(index, width or WEBCAM_DEFAULT_WIDTH, height or WEBCAM_DEFAULT_HEIGHT, fps,
                        lock_focus, focus, exposure=exposure, threaded=threaded)
