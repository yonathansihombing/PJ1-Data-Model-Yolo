"""
calib_io.py
============
Penyimpanan / pemuatan hasil INTRINSIC calibration dan kelas Undistorter.

Dipakai oleh:
  - camera_calibration.py  (menyimpan hasil)
  - field_calibration.py   (memuat + undistort)
  - vision_lapangan.py     (memuat + undistort)

ATURAN PENTING (agar koordinat pixel konsisten di semua program):
  Semua program setelah camera_calibration.py HARUS memakai Undistorter dari file ini
  dengan nilai alpha yang sama (DEFAULT_UNDISTORT_ALPHA). Titik pixel di field_calibration
  dan vision didefinisikan pada citra yang SUDAH di-undistort tersebut.
"""

import json
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

# alpha=0 -> citra hasil undistort dipotong/diperbesar sedikit sehingga tidak ada tepi hitam.
DEFAULT_UNDISTORT_ALPHA = 0.0


def _base_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return root if ext.lower() in (".npz", ".json") else path


@dataclass
class IntrinsicCalibration:
    camera_matrix: np.ndarray                  # 3x3
    dist_coeffs: np.ndarray                    # (k1, k2, p1, p2, k3)
    image_size: Tuple[int, int]                # (width, height) saat calibration
    rms_error: float                           # RMS dari cv2.calibrateCamera (pixel)
    mean_reprojection_error: float             # rata-rata jarak titik (pixel)
    num_samples: int = 0
    board: Dict = field(default_factory=dict)  # cols, rows, square_size_mm
    camera_info: Dict = field(default_factory=dict)
    created_at: str = ""
    per_view_errors: List[float] = field(default_factory=list)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    # ------------------------------------------------------------------ simpan
    def _meta(self) -> Dict:
        return {
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "rms_error": float(self.rms_error),
            "mean_reprojection_error": float(self.mean_reprojection_error),
            "num_samples": int(self.num_samples),
            "board": self.board,
            "camera_info": self.camera_info,
            "created_at": self.created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "per_view_errors": [float(e) for e in self.per_view_errors],
            "undistort_alpha": DEFAULT_UNDISTORT_ALPHA,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
        }

    def save(self, path: str) -> Tuple[str, str]:
        """Simpan ke <path>.npz dan <path>.json. Return (npz_path, json_path)."""
        base = _base_path(path)
        meta = self._meta()
        npz_path, json_path = base + ".npz", base + ".json"

        np.savez(
            npz_path,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.dist_coeffs,
            image_width=meta["image_width"],
            image_height=meta["image_height"],
            rms_error=meta["rms_error"],
            mean_reprojection_error=meta["mean_reprojection_error"],
            meta_json=json.dumps(meta),
        )
        out = dict(meta)
        out["camera_matrix"] = self.camera_matrix.tolist()
        out["distortion_coefficients"] = self.dist_coeffs.ravel().tolist()
        out["fx"], out["fy"], out["cx"], out["cy"] = self.fx, self.fy, self.cx, self.cy
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return npz_path, json_path

    # ------------------------------------------------------------------ muat
    @staticmethod
    def load(path: str) -> "IntrinsicCalibration":
        base = _base_path(path)
        npz_path, json_path = base + ".npz", base + ".json"

        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=False)
            meta = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
            return IntrinsicCalibration(
                camera_matrix=np.array(d["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(d["distortion_coefficients"], dtype=np.float64).reshape(1, -1),
                image_size=(int(d["image_width"]), int(d["image_height"])),
                rms_error=float(d["rms_error"]),
                mean_reprojection_error=float(d["mean_reprojection_error"]),
                num_samples=int(meta.get("num_samples", 0)),
                board=meta.get("board", {}),
                camera_info=meta.get("camera_info", {}),
                created_at=meta.get("created_at", ""),
                per_view_errors=meta.get("per_view_errors", []),
            )
        if os.path.exists(json_path):
            m = json.load(open(json_path, "r", encoding="utf-8"))
            return IntrinsicCalibration(
                camera_matrix=np.array(m["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(m["distortion_coefficients"], dtype=np.float64).reshape(1, -1),
                image_size=(int(m["image_width"]), int(m["image_height"])),
                rms_error=float(m["rms_error"]),
                mean_reprojection_error=float(m["mean_reprojection_error"]),
                num_samples=int(m.get("num_samples", 0)),
                board=m.get("board", {}),
                camera_info=m.get("camera_info", {}),
                created_at=m.get("created_at", ""),
                per_view_errors=m.get("per_view_errors", []),
            )
        raise FileNotFoundError(
            f"File calibration tidak ditemukan: {npz_path} (atau {json_path}). "
            f"Jalankan camera_calibration.py terlebih dahulu.")


class Undistorter:
    """Koreksi distorsi lensa. Peta remap dihitung SEKALI lalu dipakai setiap frame."""

    def __init__(self, calib: IntrinsicCalibration, alpha: float = DEFAULT_UNDISTORT_ALPHA):
        self.size = calib.image_size
        w, h = self.size
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            calib.camera_matrix, calib.dist_coeffs, (w, h), alpha, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            calib.camera_matrix, calib.dist_coeffs, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)

    def matches(self, img: np.ndarray) -> bool:
        """True bila resolusi citra sama dengan resolusi saat calibration."""
        return (img.shape[1], img.shape[0]) == tuple(self.size)

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.matches(img):
            raise ValueError(
                f"Resolusi frame {img.shape[1]}x{img.shape[0]} berbeda dari resolusi calibration "
                f"{self.size[0]}x{self.size[1]}.")
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
