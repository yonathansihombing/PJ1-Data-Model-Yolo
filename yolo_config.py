"""
yolo_config.py
==============
Konfigurasi dan utilitas bersama untuk fitur YOLO (tugas RET503).

Dipakai oleh: yolo_detector.py, train_yolo.py, evaluate_yolo.py, prepare_dataset.py.
TIDAK dipakai oleh modul kamera/vision lama, dan TIDAK mengimpor ultralytics / torch / PyYAML
saat modul ini diimpor (semuanya diimpor malas di dalam fungsi), sehingga menu lama tetap
berjalan walaupun paket YOLO belum terpasang.

Isi:
  - lokasi folder project (stabil, tidak bergantung folder terminal)
  - pembacaan YAML, penggabungan konfigurasi eksperimen (extends / override)
  - validasi desain eksperimen E1 / E2 / E3
  - pembacaan data.yaml dan pemeriksaan ringan dataset
  - konfigurasi inference (configs/inference.yaml)
  - pencarian checkpoint (.pt)
"""

import copy
import hashlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------------------
# Lokasi (selalu relatif terhadap folder project, bukan folder tempat terminal berada)
# ----------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATASET_DIR = DATASETS_DIR / "raw_yolo"              # hasil konversi/pengambilan data, sebelum dibagi
DATASET_DIR = DATASETS_DIR / "object_detection"          # hasil pembagian train/val/test (dipakai training)
REPORTS_DIR = DATASETS_DIR / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"                   # tempat bobot pretrained (.pt) yang Anda unduh
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_YAML = CONFIGS_DIR / "data.yaml"
INFERENCE_YAML = CONFIGS_DIR / "inference.yaml"

EXPERIMENT_FILES = {"e1": "e1_pretrained.yaml", "e2": "e2_scratch.yaml", "e3": "e3_finetuning.yaml"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "val", "test")


class YoloSetupError(RuntimeError):
    """Kesalahan persiapan (dependensi, file, dataset, konfigurasi) dengan pesan yang siap ditampilkan."""


# ----------------------------------------------------------------------------------------
# YAML
# ----------------------------------------------------------------------------------------
def _yaml_module():
    try:
        import yaml
        return yaml
    except ImportError as e:                                            # pragma: no cover
        raise YoloSetupError("Paket PyYAML belum terpasang. Jalankan: pip install pyyaml "
                             "(atau pip install -r requirements.txt)") from e


def load_yaml(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise YoloSetupError(f"File konfigurasi tidak ditemukan: {path}")
    yaml = _yaml_module()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise YoloSetupError(f"File YAML tidak valid ({path}): {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise YoloSetupError(f"Isi {path} harus berupa pasangan kunci: nilai (dictionary).")
    return data


def dump_yaml(data: Dict[str, Any], path) -> None:
    yaml = _yaml_module()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def resolve_path(p) -> Path:
    """Path relatif dianggap relatif terhadap folder project."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ----------------------------------------------------------------------------------------
# Konfigurasi eksperimen
# ----------------------------------------------------------------------------------------
REQUIRED_TRAIN_KEYS = ("epochs", "imgsz", "batch", "seed")


def _load_with_extends(path: Path, _seen: Optional[Tuple[Path, ...]] = None) -> Dict[str, Any]:
    _seen = _seen or ()
    path = path.resolve()
    if path in _seen:
        raise YoloSetupError(f"'extends' melingkar di {path}")
    cfg = load_yaml(path)
    parent = cfg.pop("extends", None)
    if parent:
        base = _load_with_extends((path.parent / parent), _seen + (path,))
        cfg = deep_merge(base, cfg)
    return cfg


def load_experiment(key_or_path) -> Dict[str, Any]:
    """Muat satu eksperimen (e1 / e2 / e3 atau path YAML) lengkap dengan penggabungan 'extends'."""
    key = str(key_or_path)
    path = CONFIGS_DIR / EXPERIMENT_FILES[key] if key in EXPERIMENT_FILES else resolve_path(key)
    cfg = _load_with_extends(path)
    cfg.setdefault("description", "")
    cfg.setdefault("data", "configs/data.yaml")
    cfg.setdefault("train", {})
    cfg["_source"] = str(path)
    validate_experiment(cfg)
    return cfg


def validate_experiment(cfg: Dict[str, Any]) -> None:
    errs = []
    for k in ("name", "model"):
        if not cfg.get(k):
            errs.append(f"kunci '{k}' wajib diisi")
    if "pretrained" not in cfg or not isinstance(cfg.get("pretrained"), bool):
        errs.append("kunci 'pretrained' wajib bernilai true/false")
    model = str(cfg.get("model", ""))
    if cfg.get("pretrained") is True and not model.endswith(".pt"):
        errs.append(f"pretrained: true membutuhkan model berekstensi .pt, bukan '{model}'")
    if cfg.get("pretrained") is False and not model.endswith((".yaml", ".yml")):
        errs.append(f"pretrained: false (dari nol) membutuhkan model berekstensi .yaml, bukan '{model}'. "
                    "File .pt berisi bobot pretrained dan tidak boleh dipakai untuk training dari awal.")
    train = cfg.get("train", {})
    for k in REQUIRED_TRAIN_KEYS:
        if k not in train:
            errs.append(f"train.{k} wajib diisi")
    if errs:
        raise YoloSetupError(f"Konfigurasi eksperimen tidak valid ({cfg.get('_source', '?')}):\n  - " + "\n  - ".join(errs))


def diff_experiments(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """Perbedaan yang berarti secara ilmiah antara dua eksperimen (model, pretrained, data, train.*)."""
    out: Dict[str, Tuple[Any, Any]] = {}
    for k in ("model", "pretrained", "data"):
        if a.get(k) != b.get(k):
            out[k] = (a.get(k), b.get(k))
    ta, tb = a.get("train", {}), b.get("train", {})
    for k in sorted(set(ta) | set(tb)):
        if ta.get(k) != tb.get(k):
            out[f"train.{k}"] = (ta.get(k), tb.get(k))
    return out


def check_experiment_design(e1: Dict[str, Any], e2: Dict[str, Any], e3: Dict[str, Any]) -> List[str]:
    """Aturan tugas: E2 = bobot acak saja yang berbeda; E3 = tepat SATU hyperparameter berbeda dari E1."""
    errs: List[str] = []
    if e1.get("pretrained") is not True:
        errs.append("E1 harus memakai pretrained weights (pretrained: true, model .pt).")
    d2 = diff_experiments(e1, e2)
    if e2.get("pretrained") is not False:
        errs.append("E2 harus training dari awal (pretrained: false, model .yaml).")
    extra2 = [k for k in d2 if k not in ("model", "pretrained")]
    if extra2:
        errs.append(f"E2 hanya boleh berbeda dari E1 pada model/pretrained; ternyata juga berbeda: {extra2}")
    d3 = diff_experiments(e1, e3)
    if any(k in d3 for k in ("model", "pretrained", "data")):
        errs.append(f"E3 harus memakai model/pretrained/data yang sama dengan E1; berbeda: "
                    f"{[k for k in ('model', 'pretrained', 'data') if k in d3]}")
    hyper = [k for k in d3 if k.startswith("train.")]
    if len(hyper) != 1:
        errs.append(f"E3 harus mengubah TEPAT SATU hyperparameter terhadap E1; terdeteksi {len(hyper)}: {hyper}")
    return errs


# ----------------------------------------------------------------------------------------
# data.yaml dan dataset
# ----------------------------------------------------------------------------------------
def normalize_names(names) -> List[str]:
    """names di data.yaml boleh list atau dict {0: a, 1: b}; hasil: list berurutan indeks 0..n-1."""
    if isinstance(names, dict):
        try:
            idx = sorted(int(k) for k in names)
        except (TypeError, ValueError) as e:
            raise YoloSetupError("Kunci 'names' di data.yaml harus angka 0,1,2,...") from e
        if idx != list(range(len(idx))):
            raise YoloSetupError(f"Indeks kelas di data.yaml harus berurutan mulai dari 0, ditemukan: {idx}")
        return [str(names[i] if i in names else names[str(i)]) for i in idx]
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    raise YoloSetupError("data.yaml tidak memiliki daftar 'names' yang valid.")


def load_data_config(path=DATA_YAML) -> Dict[str, Any]:
    """Baca data.yaml; 'path' diubah menjadi ABSOLUT (relatif -> terhadap folder project)."""
    cfg = load_yaml(path)
    for k in ("train", "val", "names"):
        if k not in cfg:
            raise YoloSetupError(f"data.yaml ({path}) tidak memiliki kunci '{k}'.")
    out = dict(cfg)
    out["path"] = str(resolve_path(cfg.get("path", "datasets/object_detection")).resolve())
    out["names_list"] = normalize_names(cfg["names"])
    return out


def write_resolved_data_yaml(path=DATA_YAML, dest: Optional[Path] = None) -> Path:
    """Tulis salinan data.yaml dengan path absolut (dibaca Ultralytics) dan kembalikan lokasinya."""
    cfg = load_data_config(path)
    dest = Path(dest or OUTPUTS_DIR / "resolved_data.yaml")
    data = {"path": Path(cfg["path"]).as_posix(), "train": cfg["train"], "val": cfg["val"]}
    if cfg.get("test"):
        data["test"] = cfg["test"]
    data["names"] = {i: n for i, n in enumerate(cfg["names_list"])}
    dump_yaml(data, dest)
    return dest


def _images_in(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path) -> Path:
    """Aturan YOLO: ganti '/images/' (komponen terakhir yang cocok) menjadi '/labels/', ekstensi .txt."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def dataset_counts(data_cfg: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Jumlah gambar dan label per split (berdasarkan data.yaml)."""
    root = Path(data_cfg["path"])
    out = {}
    for split in SPLITS:
        rel = data_cfg.get(split)
        if not rel:
            continue
        imgs = _images_in(root / rel)
        labels = sum(1 for p in imgs if label_path_for(p).exists())
        out[split] = {"images": len(imgs), "labels": labels}
    return out


def ensure_dataset_ready(data_cfg: Dict[str, Any], need_test: bool = False) -> Dict[str, Dict[str, int]]:
    """Gagal dengan pesan jelas bila dataset belum disiapkan. Tidak mengunduh apa pun."""
    counts = dataset_counts(data_cfg)
    problems = []
    for split in ("train", "val") + (("test",) if need_test else ()):
        c = counts.get(split)
        if not c or c["images"] == 0:
            problems.append(f"split '{split}' kosong / tidak ditemukan di {Path(data_cfg['path']) / str(data_cfg.get(split))}")
        elif c["labels"] < c["images"]:
            problems.append(f"split '{split}': {c['images'] - c['labels']} gambar tidak punya file label")
    if problems:
        raise YoloSetupError(
            "Dataset belum siap:\n  - " + "\n  - ".join(problems) + "\n"
            "Siapkan dataset terlebih dahulu (lihat README bagian 'Menyiapkan dataset'):\n"
            "  python prepare_dataset.py coco-subset --classes person car cup --download   (atau letakkan data Anda)\n"
            "  python prepare_dataset.py check --root datasets/raw_yolo\n"
            "  python prepare_dataset.py split")
    return counts


def dataset_fingerprint(root: Path) -> str:
    """Hash isi semua file label -> dataset yang sama menghasilkan kode yang sama (seperti program dosen)."""
    h = hashlib.sha256()
    for f in sorted((Path(root) / "labels").rglob("*.txt")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12].upper()


# ----------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    checkpoint: str = "runs/e1_pretrained/weights/best.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None            # None = otomatis oleh Ultralytics; "cpu", "0", ...
    max_det: int = 100


def load_inference_config(path=INFERENCE_YAML) -> InferenceConfig:
    cfg = InferenceConfig()
    p = Path(path)
    if not p.exists():
        return cfg
    data = load_yaml(p)
    for k in cfg.__dataclass_fields__:
        if k in data and data[k] is not None:
            setattr(cfg, k, data[k])
    cfg.conf = min(max(float(cfg.conf), 0.01), 0.99)
    return cfg


def list_checkpoints() -> List[Tuple[str, Path]]:
    """Checkpoint .pt yang ADA: hasil training (runs/*/weights) dan bobot pretrained (weights/*.pt)."""
    found: List[Tuple[str, Path]] = []
    if RUNS_DIR.exists():
        for run in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for name in ("best.pt", "last.pt"):
                f = run / "weights" / name
                if f.exists():
                    found.append((f"{run.name}/{name}", f))
    if WEIGHTS_DIR.exists():
        for f in sorted(WEIGHTS_DIR.glob("*.pt")):
            found.append((f"weights/{f.name} (pretrained)", f))
    return found


def find_pretrained_weights(file_name: str) -> Optional[Path]:
    """Cari bobot pretrained yang sudah Anda taruh sendiri: weights/<nama> lalu folder project."""
    for base in (WEIGHTS_DIR, PROJECT_ROOT):
        f = base / file_name
        if f.exists():
            return f
    return None


# ----------------------------------------------------------------------------------------
# Informasi lingkungan (untuk meta eksperimen / troubleshooting)
# ----------------------------------------------------------------------------------------
def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:                                                   # pragma: no cover
        return None


def environment_info() -> Dict[str, Any]:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "ultralytics": package_version("ultralytics"), "torch": package_version("torch"),
            "opencv": package_version("opencv-python") or package_version("opencv-python-headless")}


def cuda_status() -> str:
    """Teks status CUDA. Mengimpor torch (bisa lambat); hanya dipanggil saat diminta."""
    try:
        import torch
    except ImportError:
        return "torch belum terpasang"
    if torch.cuda.is_available():
        return f"CUDA tersedia: {torch.cuda.get_device_name(0)} (torch {torch.__version__})"
    return f"CUDA TIDAK tersedia (torch {torch.__version__}) -> training/inference memakai CPU"
