"""
yolo_detector.py
================
Deteksi objek YOLO (Ultralytics) pada kamera project, file video, atau file gambar.
Modul BARU dan terpisah: tidak mengubah deteksi warna/field lama (vision_lapangan.py).

Dipanggil dari main.py menu [6] lewat run_yolo_menu(). Kamera TIDAK dibuka oleh modul ini sendiri:
main.py memberikan fungsi pembuka kamera project (vl.open_camera_from_config) sehingga YOLO memakai
mekanisme yang sama dengan mode lain (camera_settings.json dimuat, resolusi/FPS/brightness/contrast/gamma
yang sama). Kamera hanya dibuka SETELAH model berhasil dimuat, dan SELALU dilepas di blok finally.

Catatan koordinat: hasil YOLO adalah bounding box dalam PIXEL pada frame kamera. Ini BUKAN koordinat
dunia / mm. Konversi ke koordinat lapangan hanya boleh dilakukan dengan homography/kalibrasi yang valid
(belum diimplementasikan di project ini).

ultralytics diimpor malas (hanya saat model dimuat), sehingga main.py dan menu lama tetap jalan walau
ultralytics/torch belum terpasang.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

import yolo_config as yc
from yolo_config import YoloSetupError

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "YOLO Object Detection"
CONF_STEP = 0.05
CONF_MIN, CONF_MAX = 0.05, 0.95
MAX_CAMERA_FAILURES = 60                   # frame kosong berturut-turut sebelum dianggap kamera bermasalah
PALETTE = [(0, 200, 0), (0, 140, 255), (255, 80, 0), (200, 0, 200), (0, 200, 200), (60, 60, 255), (200, 200, 0)]


# ----------------------------------------------------------------------------------------
# Detektor
# ----------------------------------------------------------------------------------------
@dataclass
class YoloDetection:
    class_id: int
    name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        """Titik tengah bounding box dalam PIXEL (bukan mm)."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def checkpoint_help(path: Path) -> str:
    """Instruksi bila checkpoint tidak ada (tidak diganti diam-diam dengan bobot lain)."""
    avail = yc.list_checkpoints()
    lines = [f"Checkpoint tidak ditemukan: {path}",
             "Model yang diperlukan: hasil training Anda (mis. runs/e1_pretrained/weights/best.pt).",
             "Cara menghasilkannya:",
             "  1) python prepare_dataset.py coco-subset --classes person car cup --download   (atau dataset Anda)",
             "  2) python prepare_dataset.py check --root datasets/raw_yolo ; python prepare_dataset.py split",
             "  3) python train_yolo.py --exp e1",
             "Atau tempatkan file .pt Anda di folder weights/ / runs/<nama>/weights/ lalu pilih lewat menu [4]."]
    if avail:
        lines.append("Checkpoint yang saat ini ADA di komputer ini:")
        lines += [f"    - {label}" for label, _ in avail]
    else:
        lines.append("Saat ini belum ada file .pt di runs/*/weights/ maupun weights/.")
    return "\n".join(lines)


class YoloDetector:
    def __init__(self, weights, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
                 device: Optional[str] = None, max_det: int = 100, yolo_cls=None):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise YoloSetupError(checkpoint_help(self.weights))
        self.conf, self.iou, self.imgsz, self.device, self.max_det = float(conf), float(iou), int(imgsz), device, int(max_det)
        if yolo_cls is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise YoloSetupError(
                    "Paket 'ultralytics' belum terpasang di Python ini.\n"
                    "  Pasang PyTorch (lihat README bagian B) lalu: pip install ultralytics\n"
                    "  (atau: pip install -r requirements.txt). Menu lain tidak terpengaruh.") from e
            yolo_cls = YOLO
        try:
            self.model = yolo_cls(str(self.weights))
        except Exception as e:
            raise YoloSetupError(f"Gagal memuat model '{self.weights}': {e}\n"
                                 f"Pastikan file .pt tidak rusak dan dibuat oleh versi Ultralytics yang kompatibel.") from e
        self.names: Dict[int, str] = dict(getattr(self.model, "names", {}) or {})

    def predict(self, frame_bgr: np.ndarray) -> Tuple[List[YoloDetection], float]:
        """Return (deteksi, waktu_ms). Koordinat dalam pixel pada frame yang diberikan."""
        kw = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, max_det=self.max_det, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        t0 = time.perf_counter()
        result = self.model.predict(frame_bgr, **kw)[0]
        ms = (time.perf_counter() - t0) * 1000.0
        dets: List[YoloDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            names = getattr(result, "names", None) or self.names
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                k = int(k)
                dets.append(YoloDetection(k, str(names.get(k, k)), float(c), float(x1), float(y1), float(x2), float(y2)))
        return dets, ms


def draw_detections(img: np.ndarray, dets: List[YoloDetection]) -> np.ndarray:
    """Gambar bounding box, nama kelas, dan confidence pada salinan citra."""
    out = img.copy()
    for d in dets:
        col = PALETTE[d.class_id % len(PALETTE)]
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, col, 2)
        label = f"{d.name} {d.conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, FONT, 0.5, 1)
        y0 = max(p1[1], th + 6)
        cv2.rectangle(out, (p1[0], y0 - th - 6), (p1[0] + tw + 4, y0 + base - 4), col, -1)
        cv2.putText(out, label, (p1[0] + 2, y0 - 5), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _text(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# ----------------------------------------------------------------------------------------
# Sumber frame
# ----------------------------------------------------------------------------------------
class CameraSource:
    """Membungkus kamera project (ConfiguredCamera.read() -> Frame). Tidak memiliki / menutup kameranya."""
    kind = "camera"

    def __init__(self, cam):
        self.cam = cam
        self.label = f"Kamera: {getattr(cam, 'name', 'camera')}"

    def read(self) -> Optional[np.ndarray]:
        fr = self.cam.read()
        return None if fr is None else fr.color

    def close(self):
        pass


class VideoFileSource:
    kind = "video"

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise YoloSetupError(f"File video tidak bisa dibuka: {path}")
        self.label = f"Video: {Path(path).name}"

    def read(self) -> Optional[np.ndarray]:
        ok, img = self.cap.read()
        return img if ok else None

    def close(self):
        self.cap.release()


class ImageSource:
    kind = "image"

    def __init__(self, path: Path):
        data = np.fromfile(str(path), dtype=np.uint8) if Path(path).exists() else np.empty(0, np.uint8)
        self.img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if self.img is None:
            raise YoloSetupError(f"File gambar tidak ada / tidak bisa dibaca: {path}")
        self.label = f"Gambar: {Path(path).name}"

    def read(self) -> Optional[np.ndarray]:
        return self.img.copy()

    def close(self):
        pass


# ----------------------------------------------------------------------------------------
# Loop deteksi
# ----------------------------------------------------------------------------------------
def save_snapshot(img: np.ndarray) -> Optional[Path]:
    folder = yc.OUTPUTS_DIR / "yolo_snapshots"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"yolo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            buf.tofile(str(p))
            return p
    except OSError as e:
        print(f"[WARNING] Gagal menyimpan snapshot: {e}")
    return None


def run_detection_loop(source, detector: YoloDetector, display_scale: float = 1.0, window: str = WINDOW) -> str:
    """Tampilkan deteksi. Q/ESC keluar, +/- confidence, SPASI jeda, S simpan snapshot.
    Return alasan berhenti: 'quit' | 'end' | 'error'. Window selalu ditutup."""
    cv2.namedWindow(window)
    paused, last_frame, last_view = False, None, None
    fails, fps, t_prev = 0, 0.0, time.perf_counter()
    static = getattr(source, "kind", "") == "image"
    need_predict, last_log = True, 0.0
    dets: List[YoloDetection] = []
    ms, note, note_t = 0.0, "", 0.0
    reason = "quit"
    print(f"[INFO] YOLO aktif | {source.label} | model {detector.weights.name} | conf {detector.conf:.2f} | "
          f"imgsz {detector.imgsz}. Koordinat = PIXEL (bukan mm). Tekan Q/ESC untuk kembali.")
    try:
        while True:
            if (not paused and not static) or last_frame is None:
                frame = source.read()
                if frame is None:
                    if source.kind == "video":
                        print("[INFO] Video selesai.")
                        reason = "end"
                        break
                    fails += 1
                    if fails >= MAX_CAMERA_FAILURES:
                        print("[ERROR] Kamera tidak mengirim frame. Periksa koneksi / pastikan kamera tidak dipakai program lain.")
                        reason = "error"
                        break
                    cv2.waitKey(10)
                    continue
                fails = 0
                last_frame = frame
                need_predict = True
            if need_predict and last_frame is not None:
                try:
                    dets, ms = detector.predict(last_frame)
                except Exception as e:
                    print(f"[ERROR] Inference gagal: {e}")
                    reason = "error"
                    break
                need_predict = False                       # frame berikutnya (atau ganti conf) memicu inferensi lagi
                last_view = draw_detections(last_frame, dets)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            view = last_view.copy()
            if abs(display_scale - 1.0) > 1e-3:
                view = cv2.resize(view, None, fx=display_scale, fy=display_scale)
            counts: Dict[str, int] = {}
            for d in dets:
                counts[d.name] = counts.get(d.name, 0) + 1
            summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "tidak ada deteksi"
            h, w = view.shape[:2]
            ov = view.copy()
            cv2.rectangle(ov, (0, 0), (w, 64), (0, 0, 0), -1)
            cv2.rectangle(ov, (0, h - 22), (w, h), (0, 0, 0), -1)
            view = cv2.addWeighted(ov, 0.55, view, 0.45, 0)
            _text(view, f"YOLO | {source.label} | model {detector.weights.name}" + ("  [JEDA]" if paused else ""),
                  (6, 16), 0.5, (0, 255, 255))
            _text(view, f"FPS {fps:4.1f} | inferensi {ms:5.1f} ms | conf >= {detector.conf:.2f} | imgsz {detector.imgsz}",
                  (6, 34), 0.5)
            _text(view, f"Deteksi: {summary}" + (f"   [{note}]" if note and time.time() - note_t < 3 else ""), (6, 52), 0.5, (0, 255, 0))
            _text(view, "Q/ESC kembali | +/- confidence | SPASI jeda | S simpan | koordinat = PIXEL", (6, h - 7), 0.42)
            cv2.imshow(window, view)

            if time.time() - last_log >= 2.0 and dets:
                last_log = time.time()
                print(f"[YOLO] {summary} | FPS {fps:.1f} | {ms:.1f} ms")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("+"), ord("=")):
                detector.conf = min(CONF_MAX, round(detector.conf + CONF_STEP, 2))
                note, note_t, need_predict = f"conf {detector.conf:.2f}", time.time(), True
                print(f"[INFO] confidence threshold = {detector.conf:.2f}")
            elif key in (ord("-"), ord("_")):
                detector.conf = max(CONF_MIN, round(detector.conf - CONF_STEP, 2))
                note, note_t, need_predict = f"conf {detector.conf:.2f}", time.time(), True
                print(f"[INFO] confidence threshold = {detector.conf:.2f}")
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("s"), ord("S")) and last_view is not None:
                p = save_snapshot(last_view)
                note, note_t = ("snapshot tersimpan" if p else "snapshot gagal"), time.time()
                if p:
                    print(f"[OK] Snapshot: {p}")
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
        for _ in range(3):
            cv2.waitKey(1)
    return reason


# ----------------------------------------------------------------------------------------
# Menu [6]
# ----------------------------------------------------------------------------------------
def imgsz_from_meta(checkpoint: Path) -> Optional[int]:
    """imgsz saat training bila ada experiment_meta.json di folder run checkpoint tersebut."""
    meta = Path(checkpoint).parent.parent / "experiment_meta.json"
    if meta.exists():
        try:
            import json
            return int(json.loads(meta.read_text(encoding="utf-8"))["train_config"]["imgsz"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return None


def run_yolo_menu(cfg, open_camera_fn: Callable[[], object], input_fn: Optional[Callable[[str], str]] = None,
                  yolo_cls=None) -> None:
    """Sub-menu YOLO. cfg: objek konfigurasi vision (dipakai display_scale). open_camera_fn(): membuka kamera
    project dan mengembalikan objek kamera yang punya read() dan close()."""
    input_fn = input_fn or input                       # dicari saat dipanggil (mudah diganti pada pengujian)
    try:
        icfg = yc.load_inference_config()
    except YoloSetupError as e:
        print(f"[WARNING] {e}\n[INFO] Memakai pengaturan inference bawaan.")
        icfg = yc.InferenceConfig()
    checkpoint = yc.resolve_path(icfg.checkpoint)
    scale = float(getattr(cfg, "display_scale", 1.0) or 1.0)

    def ask(prompt: str) -> str:
        try:
            return input_fn(prompt).strip()
        except EOFError:
            return "0"

    def make_detector() -> Optional[YoloDetector]:
        try:
            return YoloDetector(checkpoint, icfg.conf, icfg.iou, icfg.imgsz, icfg.device, icfg.max_det, yolo_cls)
        except YoloSetupError as e:
            print(f"\n[ERROR] {e}\n")
            return None

    while True:
        print("\n================================================")
        print("YOLO OBJECT DETECTION")
        print("================================================")
        print(f"Checkpoint : {checkpoint}  [{'ADA' if checkpoint.exists() else 'TIDAK ADA'}]")
        print(f"Confidence : {icfg.conf:.2f} | imgsz {icfg.imgsz} | device {icfg.device or 'otomatis'}")
        print("[1] Deteksi dari Webcam")
        print("[2] Deteksi dari file video")
        print("[3] Deteksi dari file gambar")
        print("[4] Pilih checkpoint (.pt)")
        print("[5] Ubah confidence threshold")
        print("[6] Status dependensi & petunjuk")
        print("[0] Kembali ke Main Menu\n")
        choice = ask("Choose: ")

        if choice in ("0", "", "b", "B"):
            return
        if choice == "1":
            det = make_detector()
            if det is None:
                continue
            cam = None
            try:
                cam = open_camera_fn()
                run_detection_loop(CameraSource(cam), det, scale)
            except KeyboardInterrupt:
                print("\n[INFO] Dihentikan (Ctrl+C).")
            except Exception as e:
                print(f"[ERROR] Webcam/YOLO gagal: {e}\n"
                      f"        Pastikan kamera terhubung, index benar (--index), dan tidak dipakai program lain.")
            finally:
                if cam is not None:
                    try:
                        cam.close()
                    except Exception as e:
                        print(f"[WARNING] Gagal melepas kamera: {e}")
                cv2.destroyAllWindows()
                print("[INFO] Kamera dilepas. Kembali ke menu YOLO.")
        elif choice in ("2", "3"):
            path = Path(ask("Path file " + ("video" if choice == "2" else "gambar") + ": ").strip('"').strip("'"))
            det = make_detector()
            if det is None:
                continue
            src = None
            try:
                src = VideoFileSource(path) if choice == "2" else ImageSource(path)
                run_detection_loop(src, det, scale)
            except YoloSetupError as e:
                print(f"[ERROR] {e}")
            except KeyboardInterrupt:
                print("\n[INFO] Dihentikan (Ctrl+C).")
            finally:
                if src is not None:
                    src.close()
                cv2.destroyAllWindows()
        elif choice == "4":
            cps = yc.list_checkpoints()
            print("\nCheckpoint yang tersedia:")
            for i, (label, _) in enumerate(cps, 1):
                print(f"  [{i}] {label}")
            print("  [P] ketik path .pt sendiri")
            sel = ask("Pilih: ")
            new = None
            if sel.lower() == "p":
                new = Path(ask("Path file .pt: ").strip('"').strip("'"))
                new = new if new.is_absolute() else yc.PROJECT_ROOT / new
            elif sel.isdigit() and 1 <= int(sel) <= len(cps):
                new = cps[int(sel) - 1][1]
            if new is None or not new.exists():
                print("[WARNING] Pilihan tidak valid / file tidak ada. Checkpoint tidak diubah.")
                continue
            checkpoint = new
            m = imgsz_from_meta(new)
            if m:
                icfg.imgsz = m
                print(f"[INFO] imgsz diatur ke {m} (sesuai training model ini).")
            print(f"[OK] Checkpoint untuk sesi ini: {checkpoint} (tidak mengubah configs/inference.yaml)")
        elif choice == "5":
            raw = ask(f"Confidence threshold baru ({CONF_MIN}-{CONF_MAX}) [sekarang {icfg.conf:.2f}]: ")
            try:
                icfg.conf = min(CONF_MAX, max(CONF_MIN, float(raw)))
                print(f"[OK] confidence = {icfg.conf:.2f}")
            except ValueError:
                print("[WARNING] Bukan angka. Tidak diubah.")
        elif choice == "6":
            print("\n--- Status ---")
            print(f"ultralytics : {yc.package_version('ultralytics') or 'BELUM TERPASANG (pip install ultralytics)'}")
            print(f"torch       : {yc.package_version('torch') or 'BELUM TERPASANG (lihat README bagian B)'}")
            print(f"OpenCV      : {cv2.__version__}")
            print(f"Perangkat   : {yc.cuda_status()}")
            try:
                dcfg = yc.load_data_config()
                print(f"Kelas data  : {dcfg['names_list']}")
                print(f"Dataset     : {yc.dataset_counts(dcfg) or 'belum disiapkan (lihat README bagian D)'}")
            except YoloSetupError as e:
                print(f"Dataset     : {e}")
            cps = yc.list_checkpoints()
            print("Checkpoint  : " + (", ".join(label for label, _ in cps) if cps else "belum ada"))
            if not checkpoint.exists():
                print("\n" + checkpoint_help(checkpoint))
        else:
            print("[WARNING] Pilihan tidak valid.")
