"""Capture gambar dari kamera project untuk dataset YOLO.

Gambar disimpan ke datasets/raw_yolo/images/ dan TIDAK diberi label otomatis.
Setiap gambar harus dianotasi sebelum dipakai untuk supervised YOLO training.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2

import yolo_config as yc

WINDOW = "Camera Dataset Capture"


def capture_dir() -> Path:
    path = yc.RAW_DATASET_DIR / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_image_path(folder: Path) -> Path:
    """Pilih nomor berikutnya tanpa menimpa gambar lama."""
    highest = 0
    for p in folder.glob("camera_*.jpg"):
        try:
            highest = max(highest, int(p.stem.split("_")[-1]))
        except (ValueError, IndexError):
            continue
    return folder / f"camera_{highest + 1:06d}.jpg"


def save_frame(frame, folder: Optional[Path] = None) -> Path:
    """Simpan satu frame BGR ke JPG; error ditampilkan sebagai exception."""
    folder = Path(folder) if folder else capture_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = next_image_path(folder)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("OpenCV gagal mengodekan frame menjadi JPG.")
    encoded.tofile(str(path))  # aman untuk path Windows/non-ASCII
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"File tidak berhasil ditulis: {path}")
    return path


def _read_color(cam):
    result = cam.read()
    if result is None:
        return None
    return getattr(result, "color", result)


def run_capture(open_camera_fn: Callable[[], object], input_fn=None) -> None:
    """Buka preview kamera; tekan S untuk menyimpan, Q/ESC untuk keluar."""
    folder = capture_dir()
    cam = None
    saved = 0
    try:
        cam = open_camera_fn()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("\n[INFO] Pengambilan gambar dataset dimulai.")
        print(f"Lokasi penyimpanan: {folder}")
        print("Tombol: S = simpan gambar | Q/ESC = selesai")
        while True:
            frame = _read_color(cam)
            if frame is None:
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            view = frame.copy()
            cv2.putText(view, f"Saved: {saved} | S: Save | Q: Quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("S")):
                try:
                    path = save_frame(frame, folder)
                    saved += 1
                    print(f"[OK] Gambar tersimpan: {path.name} ({path.stat().st_size} bytes)")
                except OSError as exc:
                    print(f"[ERROR] Gagal menyimpan gambar: {exc}")
    except KeyboardInterrupt:
        print("\n[INFO] Pengambilan gambar dibatalkan.")
    except Exception as exc:
        print(f"[ERROR] Kamera capture gagal: {exc}")
    finally:
        if cam is not None:
            try:
                cam.close()
            except Exception as exc:
                print(f"[WARNING] Gagal melepas kamera: {exc}")
        cv2.destroyAllWindows()
        print(f"[INFO] Selesai. Gambar baru disimpan sesi ini: {saved}")
        print(f"[INFO] Folder gambar: {folder}")
        if saved:
            print("[LANGKAH BERIKUTNYA] Anotasi setiap gambar dengan bounding box/class ID YOLO; "
                  "jangan langsung training gambar yang belum berlabel.")


def run_capture_menu(open_camera_fn: Callable[[], object], input_fn=None) -> None:
    input_fn = input_fn or input
    while True:
        folder = capture_dir()
        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        print("\n================================================")
        print("CAMERA DATASET CAPTURE")
        print("================================================")
        print(f"Folder: {folder}")
        print(f"Jumlah gambar: {len(images)}")
        print("[1] Buka kamera dan ambil gambar")
        print("[2] Tampilkan lokasi dataset")
        print("[0] Kembali ke Main Menu")
        try:
            choice = input_fn("Choose: ").strip()
        except EOFError:
            choice = "0"
        if choice == "0":
            return
        if choice == "1":
            run_capture(open_camera_fn, input_fn)
        elif choice == "2":
            print(f"[INFO] Lokasi: {folder.resolve()}")
            print(f"[INFO] Jumlah gambar: {len(images)}")
        else:
            print("[WARNING] Pilihan tidak valid.")
