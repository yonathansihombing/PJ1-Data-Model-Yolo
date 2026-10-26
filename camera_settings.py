"""
camera_settings.py
===================
Pengaturan kamera yang dipakai bersama oleh SEMUA mode (main.py, camera_calibration.py,
vision_lapangan.py).  Alur pembukaan kamera selalu sama:

    OPEN CAMERA -> LOAD camera_settings.json -> APPLY hardware -> VERIFY (readback)
                -> APPLY software -> WARM UP -> START

Dua jenis setting
-----------------
SOFTWARE (selalu tersedia, tidak menyentuh kamera, hanya mengubah frame lewat 1 LUT):
    Brightness  UI 0..200    actual = UI - 100         (100 = tanpa koreksi)
    Contrast    UI 0..300    factor = UI / 100         (100 = 1.0)
    Gamma       UI 10..300   gamma  = UI / 100         (100 = 1.0)

HARDWARE (Logitech C270 + DirectShow, hanya bila driver benar-benar menerimanya):
    Auto Exposure, Exposure, Gain, Autofocus, Focus, Auto White Balance, White Balance
    Setiap property: SET -> READ BACK -> VERIFY.  Status per property:
        OK           readback cocok dengan yang diminta (toleransi per property)
        MISMATCH     driver menerima set() tetapi readback berbeda  -> TIDAK dianggap berhasil
        UNSUPPORTED  backend menolak set() / readback tidak bisa dibaca
        NOT SET      belum pernah diubah
    Hanya property berstatus OK yang disimpan sebagai nilai (nilai = ACTUAL hasil readback)
    dan di-apply ulang saat program dibuka lagi.

Catatan jujur tentang OpenCV + DirectShow:
    - Rentang nilai property TIDAK bisa ditanyakan lewat OpenCV; rentang di bawah adalah rentang
      umum C270 dan hanya dipakai untuk slider.  Nilai di luar rentang akan muncul sebagai MISMATCH.
    - Pada C270, beberapa property (terutama autofocus / auto exposure) sering tidak terbaca balik
      dengan benar lewat OpenCV.  Bila itu terjadi tombol D membuka dialog driver DirectShow
      (jendela properti bawaan Windows) sebagai jalan keluar.
    - cap.set() hanya dipanggil saat nilai slider BERUBAH (bukan tiap frame) dan dibatasi
      HW_APPLY_MIN_INTERVAL detik antar-set agar video tidak tersendat.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_source import (WEBCAM_DEFAULT_FPS, WEBCAM_DEFAULT_HEIGHT, WEBCAM_DEFAULT_WIDTH,
                           CameraBase, Frame, create_camera)

SETTINGS_FILE = "camera_settings.json"
SETTINGS_VERSION = 1

# ---- konstanta yang boleh di-tuning ----------------------------------------------------------
SETTLE_FRAMES = 8              # frame yang dibuang setelah hardware settings diterapkan
HW_APPLY_MIN_INTERVAL = 0.08   # detik minimum antar cap.set() saat slider digeser
PANEL_REFRESH_FRAMES = 15      # panel status digambar ulang tiap N frame (atau saat ada perubahan)
LIGHT_EVERY_N_FRAMES = 5       # status cahaya dihitung tiap N frame
LIGHT_DARK_MEAN = 60.0         # rata-rata gray < ini  -> TOO DARK
LIGHT_BRIGHT_MEAN = 190.0      # rata-rata gray > ini  -> TOO BRIGHT
LIGHT_CLIP_PIXEL = 250         # piksel >= nilai ini dianggap clipping (overexposed)
LIGHT_CLIP_RATIO = 0.08        # bagian piksel clipping > ini -> TOO BRIGHT

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN_PREVIEW = "Camera Preview"
WIN_SETTINGS = "Setting Kamera"


def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# ======================================================================================
# SOFTWARE SETTINGS
# ======================================================================================
SW_RANGES = {"brightness": (0, 200), "contrast": (0, 300), "gamma": (10, 300)}
SW_DEFAULT = 100


@dataclass
class SoftwareSettings:
    """Nilai disimpan dalam satuan UI (100 = netral)."""
    brightness: int = SW_DEFAULT
    contrast: int = SW_DEFAULT
    gamma: int = SW_DEFAULT

    def __post_init__(self):
        self._lut_key = None
        self._lut: Optional[np.ndarray] = None
        self.clamp()

    def clamp(self):
        for k, (lo, hi) in SW_RANGES.items():
            setattr(self, k, int(min(hi, max(lo, int(getattr(self, k))))))

    @property
    def brightness_actual(self) -> int:       # UI 0..200 -> -100..+100
        return self.brightness - 100

    @property
    def contrast_factor(self) -> float:       # UI 0..300 -> 0.0..3.0
        return self.contrast / 100.0

    @property
    def gamma_value(self) -> float:           # UI 10..300 -> 0.1..3.0
        return max(10, self.gamma) / 100.0

    @property
    def is_identity(self) -> bool:
        return self.brightness == 100 and self.contrast == 100 and self.gamma == 100

    def lut(self) -> np.ndarray:
        """Satu LUT 256 entri untuk brightness+contrast+gamma (dihitung ulang hanya bila berubah)."""
        key = (self.brightness, self.contrast, self.gamma)
        if key != self._lut_key or self._lut is None:
            x = np.arange(256, dtype=np.float32)
            y = np.clip(x * self.contrast_factor + self.brightness_actual, 0, 255)
            y = 255.0 * np.power(y / 255.0, 1.0 / self.gamma_value)      # gamma > 1 -> lebih terang
            self._lut = np.clip(np.rint(y), 0, 255).astype(np.uint8)
            self._lut_key = key
        return self._lut

    def apply(self, img: np.ndarray) -> np.ndarray:
        return img if self.is_identity else cv2.LUT(img, self.lut())

    def to_dict(self) -> Dict:
        return {"brightness": self.brightness, "contrast": self.contrast, "gamma": self.gamma,
                "brightness_actual": self.brightness_actual, "contrast_factor": self.contrast_factor,
                "gamma_value": self.gamma_value}

    def load_dict(self, d: Dict):
        for k in SW_RANGES:
            if k in d:
                setattr(self, k, int(d[k]))
        self.clamp()


# ======================================================================================
# HARDWARE SETTINGS (set -> readback -> verify)
# ======================================================================================
@dataclass(frozen=True)
class HwSpec:
    key: str
    label: str
    prop: int
    kind: str                                   # "toggle" | "value"
    on_values: Tuple[float, ...] = ()           # nilai property untuk ON/auto (dicoba berurutan)
    off_values: Tuple[float, ...] = ()          # nilai property untuk OFF/manual
    parent: Optional[str] = None                # toggle auto yang harus OFF agar nilai ini berlaku
    tol: float = 0.5                            # toleransi verifikasi readback
    ui_min: int = 0
    ui_max: int = 255
    to_value: Callable[[int], float] = lambda u: float(u)
    to_ui: Callable[[float], int] = lambda v: int(round(v))


# Rentang UI = rentang umum C270 (OpenCV tidak bisa menanyakan rentang). Exposure DirectShow
# memakai skala log2 detik (-13 .. -1); slider menampilkan n dan nilai kamera = -n.
HW_SPECS: List[HwSpec] = [
    HwSpec("auto_exposure", "Auto Exposure", cv2.CAP_PROP_AUTO_EXPOSURE, "toggle",
           on_values=(0.75, 3.0), off_values=(0.25, 1.0), tol=0.01),
    HwSpec("exposure", "Exposure", cv2.CAP_PROP_EXPOSURE, "value", parent="auto_exposure",
           tol=0.5, ui_min=1, ui_max=13, to_value=lambda u: -float(u), to_ui=lambda v: int(round(-v))),
    HwSpec("gain", "Gain", cv2.CAP_PROP_GAIN, "value", tol=3.0, ui_max=255),
    HwSpec("autofocus", "Autofocus", cv2.CAP_PROP_AUTOFOCUS, "toggle",
           on_values=(1.0,), off_values=(0.0,), tol=0.01),
    HwSpec("focus", "Focus", cv2.CAP_PROP_FOCUS, "value", parent="autofocus", tol=6.0, ui_max=255),
    HwSpec("auto_wb", "Auto White Balance", cv2.CAP_PROP_AUTO_WB, "toggle",
           on_values=(1.0,), off_values=(0.0,), tol=0.01),
    HwSpec("wb_temp", "White Balance", cv2.CAP_PROP_WB_TEMPERATURE, "value", parent="auto_wb",
           tol=150.0, ui_max=45, to_value=lambda u: 2000.0 + 100.0 * u,
           to_ui=lambda v: int(round((v - 2000.0) / 100.0))),
]
HW_BY_KEY = {s.key: s for s in HW_SPECS}


@dataclass
class HwState:
    status: str = "NOT SET"                     # NOT SET | OK | MISMATCH | UNSUPPORTED
    requested: Optional[float] = None           # yang diminta (toggle: 1.0 = ON/auto, 0.0 = OFF/manual)
    actual: Optional[float] = None              # hasil readback mentah dari driver
    value: Optional[object] = None              # nilai TERVERIFIKASI (bool / float); hanya bila OK
    note: str = ""


class HardwareControls:
    """Menerapkan & memverifikasi property hardware. Tidak pernah melempar exception."""

    def __init__(self, cam: Optional[CameraBase]):
        self.cam = cam
        self.available = bool(getattr(cam, "supports_hw_controls", False)) and hasattr(cam, "set_prop")
        self.state: Dict[str, HwState] = {s.key: HwState() for s in HW_SPECS}
        self.applied: Dict[str, object] = {}     # nilai terakhir yang di-REQUEST (cegah set berulang)
        self._last_apply = 0.0
        if not self.available:
            for st in self.state.values():
                st.status, st.note = "UNSUPPORTED", "kamera ini tidak punya kontrol hardware via OpenCV"

    # ---------------------------------------------------------------- baca
    def read_raw(self, key: str) -> Optional[float]:
        if not self.available:
            return None
        v = self.cam.get_prop(HW_BY_KEY[key].prop)
        return v

    def interpret_toggle(self, spec: HwSpec, actual: Optional[float]) -> Optional[bool]:
        if actual is None:
            return None
        if any(abs(actual - v) <= spec.tol for v in spec.on_values):
            return True
        if any(abs(actual - v) <= spec.tol for v in spec.off_values):
            return False
        return None

    def read_current(self) -> Dict[str, Optional[object]]:
        """Nilai hardware SAAT INI (bool untuk toggle, float untuk value; None bila tidak terbaca)."""
        out: Dict[str, Optional[object]] = {}
        for s in HW_SPECS:
            raw = self.read_raw(s.key)
            if self.state[s.key].status == "NOT SET":
                self.state[s.key].actual = raw
            out[s.key] = self.interpret_toggle(s, raw) if s.kind == "toggle" else raw
        return out

    # ---------------------------------------------------------------- set + verify
    def apply(self, key: str, value) -> HwState:
        """SET -> READ BACK -> VERIFY untuk satu property."""
        spec, st = HW_BY_KEY[key], self.state[key]
        if not self.available:
            return st
        accepted_any, actual, matched = False, None, False
        if spec.kind == "toggle":
            st.requested = 1.0 if value else 0.0
            for cand in (spec.on_values if value else spec.off_values):
                accepted_any |= self.cam.set_prop(spec.prop, cand)
                actual = self.cam.get_prop(spec.prop)
                if actual is not None and abs(actual - cand) <= spec.tol:
                    matched = True
                    break
        else:
            st.requested = float(value)
            accepted_any = self.cam.set_prop(spec.prop, float(value))
            actual = self.cam.get_prop(spec.prop)
            matched = actual is not None and abs(actual - float(value)) <= spec.tol
        st.actual = actual
        if matched:
            st.status, st.note = "OK", ""
            st.value = bool(value) if spec.kind == "toggle" else float(actual)
        else:
            st.value = None
            if actual is None or not accepted_any:
                st.status = "UNSUPPORTED"
                st.note = "backend menolak set()" if not accepted_any else "readback gagal"
            else:
                st.status = "MISMATCH"
                st.note = f"diminta {st.requested:g}, readback {actual:g}"
        self.applied[key] = value
        return st

    def poll(self, desired: Dict[str, object]) -> List[str]:
        """Terapkan SATU property yang nilainya berubah sejak terakhir (maks. 1 set per
        HW_APPLY_MIN_INTERVAL). Return daftar key yang baru di-apply."""
        if not self.available or time.monotonic() - self._last_apply < HW_APPLY_MIN_INTERVAL:
            return []
        for spec in sorted(HW_SPECS, key=lambda s: s.kind != "toggle"):       # toggle dulu
            if spec.key not in desired:
                continue
            want = desired[spec.key]
            if spec.parent and desired.get(spec.parent) is True:               # parent = auto ON
                self.state[spec.key].note = "tidak berlaku (auto aktif)"
                continue
            if self.applied.get(spec.key) == want:
                continue
            self.apply(spec.key, want)
            self._last_apply = time.monotonic()
            if spec.kind == "toggle":                                          # anak harus di-apply ulang
                for child in HW_SPECS:
                    if child.parent == spec.key:
                        self.applied.pop(child.key, None)
            return [spec.key]
        return []

    def refresh_actuals(self):
        """Baca ulang semua nilai hardware (mis. setelah dialog driver) dan nilai ulang statusnya."""
        for s in HW_SPECS:
            st = self.state[s.key]
            raw = self.read_raw(s.key)
            st.actual = raw
            if st.requested is not None and st.status in ("OK", "MISMATCH"):
                req = (s.on_values[0] if st.requested else s.off_values[0]) if s.kind == "toggle" else st.requested
                tol_ok = raw is not None and (
                    (s.kind == "toggle" and self.interpret_toggle(s, raw) == bool(st.requested)) or
                    (s.kind == "value" and abs(raw - req) <= s.tol))
                st.status = "OK" if tol_ok else "MISMATCH"

    # ---------------------------------------------------------------- simpan / muat
    def snapshot(self) -> Dict:
        out = {}
        for s in HW_SPECS:
            st = self.state[s.key]
            e = {"label": s.label, "status": st.status, "requested": st.requested, "actual": st.actual}
            if st.status == "OK":
                e["value"] = st.value                 # ACTUAL terverifikasi (bool untuk toggle)
            out[s.key] = e
        return out

    def apply_saved(self, saved_hw: Dict) -> List[str]:
        """Terapkan hanya entri berstatus OK dari file. Return daftar peringatan."""
        warns: List[str] = []
        todo = [s for s in HW_SPECS if saved_hw.get(s.key, {}).get("status") == "OK"
                and "value" in saved_hw[s.key]]
        for spec in sorted(todo, key=lambda s: s.kind != "toggle"):
            val = saved_hw[spec.key]["value"]
            if spec.parent and saved_hw.get(spec.parent, {}).get("value") is True:
                continue                                                       # parent auto ON -> lewati
            st = self.apply(spec.key, bool(val) if spec.kind == "toggle" else float(val))
            if st.status != "OK":
                warns.append(f"{spec.label}: nilai tersimpan tidak bisa diverifikasi ({st.status}: {st.note})")
        return warns


def fmt_hw(spec: HwSpec, st: HwState) -> str:
    """Satu baris status hardware: Requested vs Actual."""
    if spec.kind == "toggle":
        req = "-" if st.requested is None else ("ON" if st.requested else "OFF")
    else:
        req = "-" if st.requested is None else f"{st.requested:g}"
    act = "-" if st.actual is None else f"{st.actual:g}"
    return f"{spec.label:<19} req={req:<6} actual={act:<7} {st.status}" + (f" ({st.note})" if st.note else "")


# ======================================================================================
# PERSISTENCE
# ======================================================================================
def load_settings(path: str = SETTINGS_FILE) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict) or "hardware" not in d or "software" not in d:
            print(f"[WARNING] {path} formatnya tidak dikenali -> diabaikan.")
            return None
        return d
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARNING] Gagal membaca {path}: {e} -> diabaikan.")
        return None


def save_settings(cam: "ConfiguredCamera", path: str = SETTINGS_FILE) -> str:
    base = cam.base.info() if hasattr(cam.base, "info") else {}
    d = {
        "version": SETTINGS_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": {k: base.get(k) for k in ("name", "type", "index", "backend", "requested_width_height_fps",
                                            "actual_width", "actual_height", "fourcc", "measured_fps")},
        "hardware": cam.hw.snapshot(),
        "software": cam.sw.to_dict(),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:       # tulis sekali (hanya saat P), bukan tiap frame
        json.dump(d, f, indent=2)
    os.replace(tmp, path)
    return path


# ======================================================================================
# CAMERA WRAPPER + open_camera()
# ======================================================================================
class ConfiguredCamera(CameraBase):
    """Kamera + software settings. read() sudah menerapkan brightness/contrast/gamma."""

    def __init__(self, base: CameraBase, sw: SoftwareSettings, hw: HardwareControls,
                 settings_path: str, loaded: bool):
        self.base, self.sw, self.hw = base, sw, hw
        self.settings_path, self.settings_loaded = settings_path, loaded

    # atribut yang dibaca modul lain -> diteruskan ke kamera asli
    @property
    def name(self):
        return self.base.name

    @property
    def has_depth(self):
        return self.base.has_depth

    @property
    def actual_size(self):
        return self.base.actual_size

    @property
    def intrinsics(self):
        return self.base.intrinsics

    def read(self) -> Optional[Frame]:
        fr = self.base.read()
        if fr is None or self.sw.is_identity:
            return fr
        return Frame(self.sw.apply(fr.color), fr.depth)

    def info(self) -> Dict:
        d = dict(self.base.info())
        d["camera_settings"] = {"loaded_from_file": self.settings_loaded, "software": self.sw.to_dict(),
                                "hardware": self.hw.snapshot()}
        return d

    def print_report(self):
        if hasattr(self.base, "print_report"):
            self.base.print_report()

    def close(self):
        self.base.close()


def open_camera(source: str = "webcam", index: int = 1,
                width: Optional[int] = WEBCAM_DEFAULT_WIDTH, height: Optional[int] = WEBCAM_DEFAULT_HEIGHT,
                fps: int = WEBCAM_DEFAULT_FPS, settings_path: str = SETTINGS_FILE, threaded: bool = True,
                lock_focus: bool = True, focus: Optional[int] = None, exposure: Optional[float] = None,
                apply_saved: bool = True, attempts: int = 3, verbose: bool = True) -> ConfiguredCamera:
    """OPEN -> LOAD settings -> APPLY hardware -> VERIFY -> APPLY software -> WARM UP."""
    base, last_err = None, None
    for i in range(attempts):                       # kamera yang baru dilepas mode lain bisa butuh jeda singkat
        try:
            base = create_camera(source, index, width, height, fps, lock_focus, focus, exposure, threaded)
            break
        except RuntimeError as e:
            last_err = e
            if i + 1 < attempts:
                time.sleep(0.3)
    if base is None:
        raise RuntimeError(f"{last_err}  (pastikan tidak ada mode/program lain yang sedang memakai kamera)")
    if verbose and hasattr(base, "print_report"):
        base.print_report()

    sw, hw = SoftwareSettings(), HardwareControls(base)
    saved = load_settings(settings_path) if apply_saved else None
    settle = 2
    if saved:
        cam_saved = saved.get("camera", {})
        if cam_saved.get("index") not in (None, index) or \
                (cam_saved.get("actual_width"), cam_saved.get("actual_height")) not in (
                    (None, None), tuple(base.actual_size)):
            print(f"[WARNING] {settings_path} disimpan untuk index {cam_saved.get('index')} "
                  f"{cam_saved.get('actual_width')}x{cam_saved.get('actual_height')}, sekarang index {index} "
                  f"{base.actual_size[0]}x{base.actual_size[1]}.")
        warns = hw.apply_saved(saved.get("hardware", {})) if hw.available else []
        sw.load_dict(saved.get("software", {}))
        settle = SETTLE_FRAMES
        if verbose:
            print(f"[CAMERA SETTINGS] {settings_path} dimuat (disimpan {saved.get('saved_at', '?')})")
            for s in HW_SPECS:
                if hw.state[s.key].status != "NOT SET":
                    print("   [HW]", fmt_hw(s, hw.state[s.key]))
            print(f"   [SW] brightness={sw.brightness} contrast={sw.contrast} gamma={sw.gamma}")
        for w in warns:
            print("[WARNING]", w)
    elif verbose:
        print(f"[CAMERA SETTINGS] {settings_path} tidak ada -> setting default kamera (software netral).")

    for _ in range(settle):                         # buang frame awal setelah setting berubah
        base.read()
    return ConfiguredCamera(base, sw, hw, settings_path, loaded=bool(saved))


# ======================================================================================
# LIGHT STATUS
# ======================================================================================
def light_status(img: np.ndarray) -> Tuple[str, float, float]:
    """(label, mean_gray, bagian_piksel_clipping). Hanya peringatan; tidak mengubah setting."""
    g = cv2.cvtColor(np.ascontiguousarray(img[::4, ::4]), cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    clip = float((g >= LIGHT_CLIP_PIXEL).mean())
    if mean < LIGHT_DARK_MEAN:
        return "TOO DARK", mean, clip
    if mean > LIGHT_BRIGHT_MEAN or clip > LIGHT_CLIP_RATIO:
        return "TOO BRIGHT", mean, clip
    return "OK", mean, clip


LIGHT_COLORS = {"OK": (0, 255, 0), "TOO DARK": (255, 160, 0), "TOO BRIGHT": (0, 0, 255)}


# ======================================================================================
# UI: Camera Preview + Setting Kamera
# ======================================================================================
# (nama trackbar, key hardware / software, maks)
SW_TRACKBARS = [("Brightness", "brightness", 200), ("Contrast", "contrast", 300), ("Gamma", "gamma", 300)]
HW_TRACKBAR_NAMES = {"auto_exposure": "AutoExposure", "exposure": "Exposure(-n)", "gain": "Gain",
                     "autofocus": "Autofocus", "focus": "Focus", "auto_wb": "AutoWB", "wb_temp": "WB(2000+100n)"}


def _panel_image(cam: ConfiguredCamera, msg: str) -> np.ndarray:
    lines: List[Tuple[str, Tuple[int, int, int]]] = [("SOFTWARE (selalu aktif)", (0, 255, 255))]
    sw = cam.sw
    lines.append((f"Brightness UI={sw.brightness} -> actual {sw.brightness_actual:+d}", (255, 255, 255)))
    lines.append((f"Contrast   UI={sw.contrast} -> x{sw.contrast_factor:.2f}", (255, 255, 255)))
    lines.append((f"Gamma      UI={sw.gamma} -> {sw.gamma_value:.2f}", (255, 255, 255)))
    lines.append(("", (0, 0, 0)))
    lines.append(("HARDWARE  (Requested vs Actual - hasil READBACK)", (0, 255, 255)))
    col = {"OK": (0, 255, 0), "MISMATCH": (0, 165, 255), "UNSUPPORTED": (0, 0, 255), "NOT SET": (190, 190, 190)}
    for s in HW_SPECS:
        st = cam.hw.state[s.key]
        lines.append((fmt_hw(s, st), col.get(st.status, (255, 255, 255))))
    lines.append(("", (0, 0, 0)))
    lines.append(("P: SAVE  R: reset software  D: dialog driver  G: baca ulang HW  Q: kembali", (255, 255, 0)))
    if msg:
        lines.append((msg, (0, 255, 0)))
    img = np.zeros((26 + 19 * len(lines), 640, 3), np.uint8)
    for i, (t, c) in enumerate(lines):
        _put(img, t, (8, 20 + 19 * i), 0.45, c)
    return img


def run_camera_settings(cam: ConfiguredCamera, display_scale: float = 1.0) -> None:
    """Window 'Camera Preview' + 'Setting Kamera'. Slider -> efek langsung, tanpa restart kamera."""
    sw, hw = cam.sw, cam.hw
    cv2.namedWindow(WIN_PREVIEW)
    cv2.namedWindow(WIN_SETTINGS)

    # --- posisi awal slider: dari setting tersimpan / readback kamera (BUKAN angka contoh) --------
    cur = hw.read_current()
    init: Dict[str, int] = {}
    unknown: List[str] = []
    for s in HW_SPECS:
        st = hw.state[s.key]
        if st.status == "OK" and st.value is not None:
            val = st.value
        else:
            val = cur.get(s.key)
        if s.kind == "toggle":
            if val is None:
                unknown.append(s.label)
            # nilai tak terbaca -> posisi awal ON (default kamera); menggeser ke OFF pasti memicu set()
            init[s.key] = 1 if (val is None or val) else 0
        else:
            if val is None:
                unknown.append(s.label)
                init[s.key] = (s.ui_min + s.ui_max) // 2
            else:
                init[s.key] = int(min(s.ui_max, max(s.ui_min, s.to_ui(float(val)))))
    noop = lambda v: None
    for label, attr, mx in SW_TRACKBARS:
        cv2.createTrackbar(label, WIN_SETTINGS, getattr(sw, attr), mx, noop)
    for s in HW_SPECS:
        cv2.createTrackbar(HW_TRACKBAR_NAMES[s.key], WIN_SETTINGS, init[s.key],
                           1 if s.kind == "toggle" else s.ui_max, noop)

    def read_hw_desired() -> Dict[str, object]:
        d: Dict[str, object] = {}
        for s in HW_SPECS:
            pos = cv2.getTrackbarPos(HW_TRACKBAR_NAMES[s.key], WIN_SETTINGS)
            if s.kind == "toggle":
                d[s.key] = bool(pos)
            else:
                clamped = min(s.ui_max, max(s.ui_min, pos))
                if clamped != pos:
                    cv2.setTrackbarPos(HW_TRACKBAR_NAMES[s.key], WIN_SETTINGS, clamped)
                d[s.key] = s.to_value(clamped)
        return d

    desired = read_hw_desired()
    hw.applied.update(desired)               # nilai awal dianggap sudah berlaku -> TIDAK ada set saat startup
    if unknown:
        print("[INFO] Nilai awal tidak terbaca dari driver (slider di posisi tengah/default, tidak di-apply): "
              + ", ".join(unknown))
    print("[INFO] Camera Settings: geser slider di 'Setting Kamera'. P = SAVE, Q = kembali ke menu.")

    def state_sig():
        """Isi setting yang relevan untuk 'sudah disimpan?' (tanpa readback yang bisa berfluktuasi)."""
        return (sw.brightness, sw.contrast, sw.gamma,
                tuple((k, st.status, st.requested) for k, st in hw.state.items()))

    msg, msg_until = "", 0.0
    saved_sig = state_sig()          # baseline = kondisi saat menu dibuka (hasil auto-load)
    quit_pending = False
    frame_i, light, fails = 0, ("OK", 0.0, 0.0), 0
    t_prev, fps = time.perf_counter(), 0.0
    last_panel = None
    panel_sig = None

    def signature():
        return (sw.brightness, sw.contrast, sw.gamma,
                tuple((k, st.status, st.requested, st.actual) for k, st in hw.state.items()))

    def say(text, sec=3.0):
        nonlocal msg, msg_until
        msg, msg_until = text, time.time() + sec
        print("[INFO]", text)

    while True:
        # --- slider software: efek langsung (hanya LUT, tidak menyentuh kamera) -----------------
        for label, attr, mx in SW_TRACKBARS:
            pos = cv2.getTrackbarPos(label, WIN_SETTINGS)
            lo = SW_RANGES[attr][0]
            if pos < lo:
                pos = lo
                cv2.setTrackbarPos(label, WIN_SETTINGS, lo)
            setattr(sw, attr, pos)
        # --- slider hardware: set hanya bila berubah ---------------------------------------------
        desired = read_hw_desired()
        hw.poll(desired)

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
        view = fr.color
        h, w = view.shape[:2]
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps and dt > 0 else (1.0 / dt if dt > 0 else 0.0)
        if frame_i % LIGHT_EVERY_N_FRAMES == 0:
            light = light_status(view)
        frame_i += 1

        _put(view, f"FPS: {fps:.1f}   {w}x{h}", (8, 20), 0.55, (0, 255, 0))
        _put(view, f"LIGHT: {light[0]}  (mean {light[1]:.0f}, clip {100 * light[2]:.1f}%)", (8, 42), 0.55,
             LIGHT_COLORS[light[0]])
        _put(view, f"SW: bright {sw.brightness_actual:+d} | contrast x{sw.contrast_factor:.2f} | "
                   f"gamma {sw.gamma_value:.2f}", (8, 62), 0.45)
        if time.time() < msg_until:
            _put(view, msg, (8, h - 28), 0.5, (0, 255, 0))
        _put(view, "P: SAVE  R: reset SW  D: dialog driver  G: baca ulang HW  Q: kembali", (8, h - 8), 0.45)
        if abs(display_scale - 1.0) > 1e-3:
            view = cv2.resize(view, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WIN_PREVIEW, view)

        sig = signature()
        if last_panel is None or sig != panel_sig or frame_i % PANEL_REFRESH_FRAMES == 0 or msg_until > 0:
            last_panel = _panel_image(cam, msg if time.time() < msg_until else "")
            panel_sig = sig
        cv2.imshow(WIN_SETTINGS, last_panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            if state_sig() != saved_sig and not quit_pending:
                quit_pending = True
                say("Ada perubahan yang BELUM disimpan. P = simpan, Q lagi = keluar tanpa menyimpan.", 4.0)
                continue
            break
        elif key != 255:
            quit_pending = False
        if key in (ord("p"), ord("P")):
            try:
                path = save_settings(cam, cam.settings_path)
                saved_sig = state_sig()
                n_ok = sum(1 for st in hw.state.values() if st.status == "OK")
                say(f"SAVED {path} (hardware OK: {n_ok}, software: bright {sw.brightness} "
                    f"contrast {sw.contrast} gamma {sw.gamma})", 4.0)
                for s in HW_SPECS:
                    if hw.state[s.key].status in ("MISMATCH", "UNSUPPORTED"):
                        print(f"[WARNING] {s.label}: {hw.state[s.key].status} -> TIDAK disimpan sebagai nilai.")
            except OSError as e:
                say(f"Gagal menyimpan: {e}", 4.0)
        elif key in (ord("r"), ord("R")):
            for label, attr, mx in SW_TRACKBARS:
                cv2.setTrackbarPos(label, WIN_SETTINGS, SW_DEFAULT)
            say("Software settings di-reset ke 100/100/100.")
        elif key in (ord("d"), ord("D")):
            if hw.available and hasattr(cam.base, "set_prop"):
                prop = getattr(cv2, "CAP_PROP_SETTINGS", 37)
                ok = cam.base.set_prop(prop, 1)
                say("Dialog driver DirectShow dibuka. Setelah ditutup tekan G." if ok
                    else "Dialog driver tidak didukung backend ini.", 4.0)
            else:
                say("Dialog driver hanya untuk webcam DirectShow.", 3.0)
        elif key in (ord("g"), ord("G")):
            hw.refresh_actuals()
            say("Nilai hardware dibaca ulang dari kamera.")

    for name in (WIN_PREVIEW, WIN_SETTINGS):
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass
