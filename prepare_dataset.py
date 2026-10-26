#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Pipeline dataset YOLO untuk tugas RET503 (menggabungkan program dosen 00-05 dengan pemeriksaan
yang lebih ketat). Semua langkah eksplisit; TIDAK ada unduhan otomatis kecuali Anda memberi --download.

Sub-perintah:
  coco-subset   Ambil subset kelas pilihan dari COCO val2017 -> datasets/raw_yolo/
  coco-to-yolo  Konversi satu file COCO JSON -> label YOLO (+ laporan konversi)
  yolo-to-coco  Konversi dataset YOLO -> satu file COCO JSON
  check         Periksa dataset (gambar rusak, label hilang/tidak valid, kelas, kotak di luar batas,
                duplikat, distribusi kelas) + visualisasi kotak + laporan
  split         Bagi train/val/test (seed tetap, anti-kebocoran) -> datasets/object_detection/
                dan perbarui configs/data.yaml

Alur umum:
  python prepare_dataset.py coco-subset --classes person car cup --count 200 --download
  python prepare_dataset.py check --root datasets/raw_yolo --visualize 12
  python prepare_dataset.py split
  python prepare_dataset.py check --root datasets/object_detection

Format label YOLO: satu .txt per gambar, tiap baris "class_id cx cy w h" (titik tengah, lebar, tinggi,
dinormalisasi 0-1, class_id mulai 0). Gambar tanpa objek memakai file .txt kosong (gambar latar).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from yolo_config import (DATA_YAML, DATASET_DIR, DATASETS_DIR, IMAGE_EXTS, PROJECT_ROOT, RAW_DATASET_DIR,
                         REPORTS_DIR, SPLITS, YoloSetupError, dataset_fingerprint, label_path_for,
                         load_data_config, resolve_path)

# URL berikut berasal dari program dosen (00_ambil_subset_coco.py); tidak diverifikasi dari sini.
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_DIR = DATASETS_DIR / "coco_download"

MIN_SIDE_PX = 8              # kotak lebih kecil dari ini patut dicurigai (peringatan)
BOX_TOLERANCE = 1e-4         # toleransi pembulatan saat memeriksa kotak di luar batas gambar
IMBALANCE_RATIO = 3.0        # kelas dianggap "jauh lebih sedikit" bila jumlah kotaknya <= terbanyak / 3
VISUAL_COLORS = [(0, 200, 0), (0, 140, 255), (255, 80, 0), (200, 0, 200), (0, 200, 200), (120, 120, 255)]


# ========================================================================================
# Utilitas
# ========================================================================================
def imread(path: Path) -> Optional[np.ndarray]:
    """cv2.imread yang aman untuk path non-ASCII (Windows)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def imwrite(path: Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_classes_file(path: Path) -> List[str]:
    return [k.strip() for k in path.read_text(encoding="utf-8").splitlines() if k.strip()]


def classes_for_root(root: Path, classes_file: Optional[str] = None) -> List[str]:
    """classes.txt di folder dataset; bila tidak ada, pakai names dari configs/data.yaml."""
    if classes_file:
        return read_classes_file(resolve_path(classes_file))
    if (root / "classes.txt").exists():
        return read_classes_file(root / "classes.txt")
    if DATA_YAML.exists():
        return load_data_config(DATA_YAML)["names_list"]
    raise YoloSetupError(f"Tidak ada daftar kelas: {root / 'classes.txt'} dan {DATA_YAML} tidak tersedia.")


def split_of(image: Path, images_root: Path) -> str:
    try:
        first = image.relative_to(images_root).parts[0]
    except ValueError:
        return "all"
    return first if first in SPLITS else "all"


def parse_label(path: Path, n_classes: int):
    """Return (rows, issues). rows = [(cid, cx, cy, w, h)]; issues = [(severity, pesan)]."""
    rows, issues = [], []
    seen = set()
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(("ERROR", f"{path.name} baris {i}: harus 5 nilai, ditemukan {len(parts)}"))
            continue
        try:
            cid_f = float(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append(("ERROR", f"{path.name} baris {i}: ada nilai bukan angka"))
            continue
        if cid_f != int(cid_f):
            issues.append(("ERROR", f"{path.name} baris {i}: class_id harus bilangan bulat"))
            continue
        cid = int(cid_f)
        if not 0 <= cid < n_classes:
            issues.append(("ERROR", f"{path.name} baris {i}: class_id {cid} di luar 0..{n_classes - 1}"))
            continue
        if not all(np.isfinite(v) for v in (cx, cy, w, h)):
            issues.append(("ERROR", f"{path.name} baris {i}: nilai tidak hingga (NaN/inf)"))
            continue
        if not all(-BOX_TOLERANCE <= v <= 1 + BOX_TOLERANCE for v in (cx, cy, w, h)):
            issues.append(("ERROR", f"{path.name} baris {i}: nilai di luar 0-1 (belum dinormalisasi?)"))
            continue
        if w <= 0 or h <= 0:
            issues.append(("ERROR", f"{path.name} baris {i}: lebar/tinggi kotak <= 0"))
            continue
        if cx - w / 2 < -BOX_TOLERANCE or cx + w / 2 > 1 + BOX_TOLERANCE or \
                cy - h / 2 < -BOX_TOLERANCE or cy + h / 2 > 1 + BOX_TOLERANCE:
            issues.append(("ERROR", f"{path.name} baris {i}: kotak keluar dari batas gambar"))
            continue
        key = (cid, round(cx, 5), round(cy, 5), round(w, 5), round(h, 5))
        if key in seen:
            issues.append(("WARNING", f"{path.name} baris {i}: kotak duplikat"))
        seen.add(key)
        rows.append((cid, cx, cy, w, h))
    return rows, issues


# ========================================================================================
# check
# ========================================================================================
def check_dataset(root: Path, classes: Sequence[str], visualize: int = 0, seed: int = 0,
                  min_side: int = MIN_SIDE_PX, name: Optional[str] = None) -> Dict:
    root = Path(root)
    images_root = root / "images"
    images = list_images(images_root)
    n_classes = len(classes)
    issues: List[Tuple[str, str]] = []
    per_split_images, per_split_objs = Counter(), defaultdict(Counter)
    total_objs = Counter()
    empty_labels = 0
    hashes: Dict[str, List[Path]] = defaultdict(list)
    parsed: Dict[Path, list] = {}
    sizes: Dict[Path, Tuple[int, int]] = {}

    if not images:
        issues.append(("ERROR", f"Tidak ada gambar di {images_root}"))

    for img_path in images:
        split = split_of(img_path, images_root)
        per_split_images[split] += 1
        img = imread(img_path)
        if img is None:
            issues.append(("ERROR", f"{img_path.name}: gambar rusak / tidak bisa dibaca"))
            continue
        h, w = img.shape[:2]
        sizes[img_path] = (w, h)
        hashes[file_md5(img_path)].append(img_path)
        lbl = label_path_for(img_path)
        if not lbl.exists():
            issues.append(("ERROR", f"{img_path.name}: tidak ada file label ({lbl.name}); "
                                    f"gambar tanpa objek harus punya file .txt KOSONG"))
            continue
        rows, row_issues = parse_label(lbl, n_classes)
        issues += row_issues
        parsed[img_path] = rows
        if not rows and not row_issues:
            empty_labels += 1
        for cid, cx, cy, bw, bh in rows:
            per_split_objs[split][classes[cid]] += 1
            total_objs[classes[cid]] += 1
            if bw * w < min_side or bh * h < min_side:
                issues.append(("WARNING", f"{lbl.name}: kotak sangat kecil ({bw * w:.0f}x{bh * h:.0f} px)"))

    # label yatim (tanpa gambar)
    labels_root = root / "labels"
    if labels_root.exists():
        img_stems = {(split_of(p, images_root), p.stem) for p in images}
        for lbl in sorted(labels_root.rglob("*.txt")):
            if lbl.name == "classes.txt":
                continue
            if (split_of(lbl, labels_root), lbl.stem) not in img_stems:
                issues.append(("WARNING", f"{lbl.name}: label tanpa gambar pasangan"))

    # duplikat gambar (identik byte-per-byte); lintas split = kebocoran data
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    cross_split = 0
    for grp in dup_groups:
        splits = {split_of(p, images_root) for p in grp}
        names = ", ".join(p.name for p in grp[:4])
        if len(splits) > 1:
            cross_split += 1
            issues.append(("ERROR", f"KEBOCORAN DATA: gambar identik ada di split berbeda ({sorted(splits)}): {names}"))
        else:
            issues.append(("WARNING", f"gambar identik (duplikat): {names}"))

    # distribusi kelas
    counts = {c: total_objs.get(c, 0) for c in classes}
    nonzero = [v for v in counts.values() if v > 0]
    imbalance = None
    minor = []
    if nonzero:
        biggest = max(nonzero)
        imbalance = round(biggest / min(nonzero), 2)
        minor = [c for c, v in counts.items() if 0 < v and v * IMBALANCE_RATIO <= biggest]
    for c, v in counts.items():
        if v == 0 and images:
            issues.append(("WARNING", f"kelas '{c}' tidak punya satu pun kotak di dataset ini"))
    for split in sorted(per_split_images):
        for c in classes:
            if split in ("val", "test") and per_split_objs[split].get(c, 0) == 0 and counts[c] > 0:
                issues.append(("WARNING", f"split '{split}' tidak memuat kelas '{c}' -> metrik kelas itu tidak bermakna"))

    errors = [m for s, m in issues if s == "ERROR"]
    warnings = [m for s, m in issues if s == "WARNING"]
    report = {
        "root": str(root), "classes": list(classes), "images": len(images),
        "images_per_split": dict(per_split_images), "empty_labels_background": empty_labels,
        "objects_per_class": counts,
        "objects_per_class_per_split": {s: dict(c) for s, c in per_split_objs.items()},
        "largest_over_smallest_class_ratio": imbalance, "minor_classes": minor,
        "duplicate_image_groups": len(dup_groups), "cross_split_duplicate_groups": cross_split,
        "num_errors": len(errors), "num_warnings": len(warnings),
        "errors": errors, "warnings": warnings,
        "label_fingerprint": dataset_fingerprint(root) if labels_root.exists() else None,
    }

    if visualize > 0:
        report["visual_dir"] = str(_visualize(root, parsed, sizes, classes, visualize, seed, name or root.name))
    return report


def _visualize(root: Path, parsed: Dict[Path, list], sizes, classes, n: int, seed: int, name: str) -> Path:
    out = REPORTS_DIR / f"visual_{name}"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = sorted(parsed, key=lambda p: str(p))
    for p in rng.sample(pool, min(n, len(pool))):
        img = imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        for cid, cx, cy, bw, bh in parsed[p]:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            col = VISUAL_COLORS[cid % len(VISUAL_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, classes[cid], (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        imwrite(out / (p.stem + ".jpg"), img)
    return out


def format_report(rep: Dict, max_lines: int = 25) -> str:
    L = ["=" * 70, f"LAPORAN DATASET: {rep['root']}", "=" * 70,
         f"Gambar                : {rep['images']}  per split: {rep['images_per_split']}",
         f"Gambar latar (label kosong): {rep['empty_labels_background']}",
         "Jumlah kotak per kelas:"]
    total = sum(rep["objects_per_class"].values()) or 1
    for c, v in rep["objects_per_class"].items():
        L.append(f"  {c:14s}: {v:6d}  ({100 * v / total:5.1f}% dari semua kotak)")
    if rep["objects_per_class_per_split"]:
        L.append("Kotak per kelas per split:")
        for s, d in sorted(rep["objects_per_class_per_split"].items()):
            L.append(f"  {s:6s}: {d}")
    r = rep["largest_over_smallest_class_ratio"]
    if r is not None:
        L.append(f"Rasio kelas terbanyak / tersedikit: {r}x")
        if rep["minor_classes"]:
            L.append(f"Kelas yang jauh lebih sedikit (<= 1/{IMBALANCE_RATIO:g} kelas terbanyak): "
                     f"{', '.join(rep['minor_classes'])}  -> perhatikan hasil per kelas, bukan hanya rata-rata")
        else:
            L.append(f"Tidak ada kelas dengan jumlah <= 1/{IMBALANCE_RATIO:g} kelas terbanyak.")
    L.append(f"Gambar identik: {rep['duplicate_image_groups']} grup (lintas split: {rep['cross_split_duplicate_groups']})")
    L.append(f"Sidik jari label: {rep['label_fingerprint']}")
    L.append(f"ERROR: {rep['num_errors']}   WARNING: {rep['num_warnings']}")
    for tag, items in (("ERROR", rep["errors"]), ("WARNING", rep["warnings"])):
        for m in items[:max_lines]:
            L.append(f"  [{tag}] {m}")
        if len(items) > max_lines:
            L.append(f"  ... {len(items) - max_lines} {tag} lainnya (lihat file laporan JSON)")
    if rep.get("visual_dir"):
        L.append(f"Visualisasi kotak: {rep['visual_dir']}")
    return "\n".join(L)


def save_report(rep: Dict, name: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / f"{name}_check.json"
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / f"{name}_check.txt").write_text(format_report(rep, 10 ** 6) + "\n", encoding="utf-8")
    return p


# ========================================================================================
# coco-subset
# ========================================================================================
def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url}\n           -> {dest}  (bisa beberapa menit)")

    def hook(blocks, bsize, total):
        if total > 0 and blocks % 400 == 0:
            print(f"           {min(100, blocks * bsize * 100 // total)}%", end="\r", flush=True)

    urllib.request.urlretrieve(url, dest, hook)
    print()


def ensure_coco_files(coco_dir: Path, allow_download: bool) -> Tuple[Path, Path]:
    images = coco_dir / "val2017"
    ann = coco_dir / "annotations" / "instances_val2017.json"
    if images.exists() and ann.exists():
        return images, ann
    if not allow_download:
        raise YoloSetupError(
            f"File COCO val2017 belum ada di {coco_dir}.\n"
            f"  Pilihan A (otomatis, hanya bila Anda mengizinkan): jalankan ulang dengan --download\n"
            f"  Pilihan B (manual): unduh dan ekstrak sendiri sehingga tersedia\n"
            f"      {images}{'/'}  (gambar *.jpg)\n"
            f"      {ann}\n"
            f"    Sumber (dari program dosen, belum diverifikasi dari sini):\n"
            f"      {COCO_IMAGES_URL}\n      {COCO_ANNOTATIONS_URL}\n"
            f"  Periksa ketentuan lisensi di https://cocodataset.org/#termsofuse sebelum memakai data.")
    coco_dir.mkdir(parents=True, exist_ok=True)
    for url, check in ((COCO_IMAGES_URL, images), (COCO_ANNOTATIONS_URL, ann)):
        if check.exists():
            continue
        zpath = coco_dir / url.split("/")[-1]
        if not zpath.exists():
            _download(url, zpath)
        print(f"[EXTRACT] {zpath.name}")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(coco_dir)
    if not (images.exists() and ann.exists()):
        raise YoloSetupError("Ekstraksi selesai tetapi struktur val2017/ dan annotations/ tidak ditemukan.")
    return images, ann


def _prepare_out(out: Path, overwrite: bool, keep=("images", "labels", "classes.txt")):
    existing = [out / k for k in keep if (out / k).exists() and any((out / k).iterdir() if (out / k).is_dir() else [1])]
    if existing and not overwrite:
        raise YoloSetupError(f"{out} sudah berisi data ({', '.join(p.name for p in existing)}). "
                             f"Tambahkan --overwrite untuk menimpa, atau pilih --out lain.")
    for k in keep:
        t = out / k
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            t.unlink()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)


def coco_subset(classes: Sequence[str], count: int, background: float, seed: int, out: Path,
                coco_dir: Path, allow_download: bool, overwrite: bool) -> Dict:
    images_dir, ann_path = ensure_coco_files(coco_dir, allow_download)
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    bad = [k for k in classes if k not in name_to_id]
    if bad:
        raise YoloSetupError(f"Kelas tidak ada di COCO: {bad}. Contoh nama yang benar: "
                             f"{', '.join(sorted(name_to_id)[:15])}, ...")
    if len(set(classes)) != len(classes):
        raise YoloSetupError("Daftar --classes berisi kelas ganda.")
    new_id = {name_to_id[k]: i for i, k in enumerate(classes)}
    info = {g["id"]: g for g in coco["images"]}

    per_image: Dict[int, list] = {}
    skipped_crowd = skipped_invalid = 0
    for a in coco["annotations"]:
        if a["category_id"] not in new_id:
            continue
        if a.get("iscrowd", 0):
            skipped_crowd += 1
            continue
        g = info.get(a["image_id"])
        x, y, w, h = a["bbox"]
        if g is None or w <= 0 or h <= 0:
            skipped_invalid += 1
            continue
        per_image.setdefault(a["image_id"], []).append(a)

    rng = random.Random(seed)
    with_obj = rng.sample(sorted(per_image), min(count, len(per_image)))
    bg_pool = sorted(set(info) - set(per_image))
    background_ids = rng.sample(bg_pool, min(int(len(with_obj) * background), len(bg_pool)))

    _prepare_out(out, overwrite, keep=("images", "labels", "classes.txt", "source_info.json"))
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    counts = Counter()
    missing = 0
    for img_id in with_obj + background_ids:
        g = info[img_id]
        src = images_dir / g["file_name"]
        if not src.exists():
            missing += 1
            continue
        W, H = g["width"], g["height"]
        lines = []
        for a in per_image.get(img_id, []):
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)   # potong ke batas gambar
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                skipped_invalid += 1
                continue
            cid = new_id[a["category_id"]]
            lines.append(f"{cid} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            counts[classes[cid]] += 1
        shutil.copy2(src, out / "images" / g["file_name"])
        (out / "labels" / (Path(g["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    summary = {"source": "COCO val2017", "images_url": COCO_IMAGES_URL, "annotations_url": COCO_ANNOTATIONS_URL,
               "classes": list(classes), "seed": seed, "images_with_objects": len(with_obj),
               "background_images": len(background_ids), "boxes_per_class": {c: counts.get(c, 0) for c in classes},
               "skipped_crowd": skipped_crowd, "skipped_invalid": skipped_invalid, "missing_image_files": missing,
               "note": "Objek COCO selain kelas terpilih tidak dianotasi pada subset ini.",
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "source_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ========================================================================================
# COCO <-> YOLO
# ========================================================================================
def coco_to_yolo(coco_file: Path, out: Path, images_dir: Optional[Path] = None, classes: Optional[Sequence[str]] = None,
                 keep_crowd: bool = False, overwrite: bool = False) -> Dict:
    coco = json.loads(Path(coco_file).read_text(encoding="utf-8"))
    for k in ("images", "annotations", "categories"):
        if k not in coco:
            raise YoloSetupError(f"File COCO tidak memiliki kunci '{k}'.")
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    if classes:
        by_name = {c["name"]: c for c in cats}
        bad = [n for n in classes if n not in by_name]
        if bad:
            raise YoloSetupError(f"Kategori tidak ada di file COCO: {bad}. Yang ada: {[c['name'] for c in cats]}")
        cats = [by_name[n] for n in classes]                 # urutan --classes = class_id
    id_map = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    stems = Counter(Path(g["file_name"]).stem for g in coco["images"])
    dup = [s for s, n in stems.items() if n > 1]
    if dup:
        raise YoloSetupError(f"Nama file gambar ganda (stem sama) di COCO JSON, label akan bertabrakan: {dup[:5]}")

    _prepare_out(out, overwrite, keep=("labels", "classes.txt", "conversion_report.json") +
                 (("images",) if images_dir else ()))
    img_info = {g["id"]: g for g in coco["images"]}
    per_image: Dict[int, list] = defaultdict(list)
    rep = Counter()
    problems: List[str] = []
    for a in coco["annotations"]:
        rep["annotations_total"] += 1
        if a["category_id"] not in id_map:
            rep["skipped_other_category"] += 1
            continue
        if a.get("iscrowd", 0) and not keep_crowd:
            rep["skipped_crowd"] += 1
            continue
        g = img_info.get(a["image_id"])
        if g is None:
            rep["skipped_unknown_image"] += 1
            problems.append(f"anotasi {a.get('id')}: image_id {a['image_id']} tidak ada")
            continue
        try:
            x, y, w, h = (float(v) for v in a["bbox"])
        except (ValueError, TypeError, KeyError):
            rep["skipped_invalid_bbox"] += 1
            problems.append(f"anotasi {a.get('id')}: bbox tidak valid")
            continue
        W, H = g["width"], g["height"]
        if not all(np.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            rep["skipped_invalid_bbox"] += 1
            problems.append(f"anotasi {a.get('id')} ({g['file_name']}): lebar/tinggi <= 0")
            continue
        x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
        if (x1, y1, x2, y2) != (x, y, x + w, y + h):
            rep["clipped_to_image"] += 1
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            rep["skipped_outside_image"] += 1
            problems.append(f"anotasi {a.get('id')} ({g['file_name']}): kotak sepenuhnya di luar gambar")
            continue
        per_image[a["image_id"]].append((id_map[a["category_id"]], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                         (x2 - x1) / W, (y2 - y1) / H))
        rep["converted"] += 1
    if images_dir:
        (out / "images").mkdir(exist_ok=True)
    for g in coco["images"]:
        stem = Path(g["file_name"]).stem
        rows = per_image.get(g["id"], [])
        (out / "labels" / f"{stem}.txt").write_text(
            "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows) + ("\n" if rows else ""))
        rep["label_files"] += 1
        if not rows:
            rep["images_without_boxes"] += 1
        if images_dir:
            src = Path(images_dir) / g["file_name"]
            if src.exists():
                shutil.copy2(src, out / "images" / Path(g["file_name"]).name)
            else:
                rep["missing_image_files"] += 1
                problems.append(f"gambar tidak ditemukan: {src}")
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    report = dict(rep)
    report.update({"category_mapping": {c["name"]: {"coco_id": c["id"], "yolo_class_id": id_map[c["id"]]} for c in cats},
                   "problems": problems[:200], "num_problems": len(problems)})
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def yolo_to_coco(root: Path, out_file: Path, classes: Sequence[str]) -> Dict:
    root = Path(root)
    images = list_images(root / "images")
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)]}
    ann_id, skipped = 1, Counter()
    for img_id, p in enumerate(images, 1):
        img = imread(p)
        if img is None:
            skipped["unreadable_image"] += 1
            continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lbl = label_path_for(p)
        if not lbl.exists():
            skipped["missing_label"] += 1
            continue
        rows, issues = parse_label(lbl, len(classes))
        skipped["invalid_rows"] += len(issues)
        for cid, cx, cy, bw, bh in rows:
            bw_px, bh_px = bw * w, bh * h
            coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cid + 1,
                                        "bbox": [round(cx * w - bw_px / 2, 2), round(cy * h - bh_px / 2, 2),
                                                 round(bw_px, 2), round(bh_px, 2)],
                                        "area": round(bw_px * bh_px, 2), "iscrowd": 0})
            ann_id += 1
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return {"images": len(coco["images"]), "annotations": len(coco["annotations"]), "skipped": dict(skipped)}


# ========================================================================================
# split (reproducible, anti-leakage)
# ========================================================================================
class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def write_data_yaml(path: Path, dataset_dir: Path, classes: Sequence[str]) -> None:
    try:
        rel = Path(dataset_dir).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = Path(dataset_dir).resolve().as_posix()
    lines = ["# data.yaml - diperbarui oleh: python prepare_dataset.py split",
             "# path relatif dihitung dari folder project; urutan names = class_id (mulai 0).",
             f"path: {rel}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(classes)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_dataset(src: Path, dst: Path, ratios: Sequence[float], seed: int, group_regex: Optional[str],
                  overwrite: bool, update_data_yaml: bool = True, force: bool = False) -> Dict:
    src, dst = Path(src), Path(dst)
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise YoloSetupError("--ratios harus tiga angka positif berjumlah 1, mis. 0.7 0.2 0.1")
    if any((src / "images" / s).is_dir() for s in SPLITS):
        raise YoloSetupError(f"{src} sudah memiliki pembagian train/val/test. Dataset dengan pembagian resmi tidak "
                             f"diacak ulang. Periksa dengan: python prepare_dataset.py check --root {src}")
    classes = classes_for_root(src)
    rep = check_dataset(src, classes)
    if rep["num_errors"] and not force:
        raise YoloSetupError(f"Dataset sumber punya {rep['num_errors']} ERROR; perbaiki dulu (lihat: prepare_dataset.py "
                             f"check --root {src}) atau paksa dengan --force.\n  contoh: {rep['errors'][:3]}")
    images = [p for p in list_images(src / "images") if label_path_for(p).exists()]
    if len(images) < 10:
        raise YoloSetupError(f"Terlalu sedikit gambar berlabel ({len(images)}) untuk dibagi tiga.")

    dsu = _DSU(len(images))
    by_hash: Dict[str, int] = {}
    by_group: Dict[str, int] = {}
    group_name = [p.stem for p in images]
    rx = re.compile(group_regex) if group_regex else None
    for i, p in enumerate(images):
        h = file_md5(p)
        if h in by_hash:
            dsu.union(i, by_hash[h])
        else:
            by_hash[h] = i
        if rx:
            m = rx.search(p.stem)
            if m:
                key = m.group(1) if m.groups() else m.group(0)
                group_name[i] = key
                if key in by_group:
                    dsu.union(i, by_group[key])
                else:
                    by_group[key] = i
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(images)):
        groups[dsu.find(i)].append(i)
    ordered = sorted(groups.values(), key=lambda g: images[g[0]].name)
    random.Random(seed).shuffle(ordered)

    n = len(images)
    target = {s: r * n for s, r in zip(SPLITS, ratios)}
    assigned: Dict[str, List[int]] = {s: [] for s in SPLITS}
    for g in ordered:
        s = max(SPLITS, key=lambda k: (target[k] - len(assigned[k])) / target[k])
        assigned[s].extend(g)
    if any(not v for v in assigned.values()):
        raise YoloSetupError(f"Ada split yang kosong ({ {k: len(v) for k, v in assigned.items()} }); "
                             f"data terlalu sedikit atau grup terlalu besar.")

    existing = [d for d in ("images", "labels") if (dst / d).exists() and any(
        f for f in (dst / d).rglob("*") if f.is_file() and f.name != ".gitkeep")]
    if existing and not overwrite:
        raise YoloSetupError(f"{dst} sudah berisi data. Tambahkan --overwrite untuk menimpa (data lama dihapus).")
    for d in ("images", "labels"):
        if (dst / d).exists():
            shutil.rmtree(dst / d)
    manifest = []
    for s in SPLITS:
        (dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / s).mkdir(parents=True, exist_ok=True)
        for i in sorted(assigned[s], key=lambda k: images[k].name):
            p = images[i]
            shutil.copy2(p, dst / "images" / s / p.name)
            shutil.copy2(label_path_for(p), dst / "labels" / s / (p.stem + ".txt"))
            manifest.append((p.name, group_name[i], s))
    (dst / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    with open(dst / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "group", "split"])
        w.writerows(manifest)
    out_rep = check_dataset(dst, classes)
    summary = {"source": str(src), "seed": seed, "ratios": dict(zip(SPLITS, ratios)), "group_regex": group_regex,
               "images": {s: len(assigned[s]) for s in SPLITS}, "groups": len(ordered),
               "objects_per_class_per_split": out_rep["objects_per_class_per_split"],
               "label_fingerprint": dataset_fingerprint(dst), "warnings": out_rep["warnings"],
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (dst / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if update_data_yaml:
        write_data_yaml(DATA_YAML, dst, classes)
    return summary


# ========================================================================================
# CLI
# ========================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pipeline dataset YOLO (RET503). Lihat README bagian dataset.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("coco-subset", help="ambil subset kelas dari COCO val2017")
    s.add_argument("--classes", nargs="+", default=["person", "car", "cup"],
                   help="nama kelas COCO; urutan = class_id (default: person car cup)")
    s.add_argument("--count", type=int, default=200, help="jumlah gambar berisi objek (default 200)")
    s.add_argument("--background", type=float, default=0.1, help="proporsi gambar latar tanpa objek (default 0.1)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(RAW_DATASET_DIR))
    s.add_argument("--coco-dir", default=str(COCO_DIR), help="lokasi val2017/ dan annotations/ COCO")
    s.add_argument("--download", action="store_true", help="IZINKAN mengunduh COCO val2017 (+/- 1 GB) bila belum ada")
    s.add_argument("--overwrite", action="store_true")

    c = sub.add_parser("check", help="periksa dataset + laporan + visualisasi")
    c.add_argument("--root", default=str(RAW_DATASET_DIR), help="folder berisi images/ dan labels/")
    c.add_argument("--classes-file", default=None, help="classes.txt (default: <root>/classes.txt atau configs/data.yaml)")
    c.add_argument("--visualize", type=int, default=0, metavar="N", help="simpan N gambar contoh dengan kotak")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--min-side", type=int, default=MIN_SIDE_PX)

    k = sub.add_parser("coco-to-yolo", help="COCO JSON -> label YOLO")
    k.add_argument("--coco", required=True, help="file COCO JSON")
    k.add_argument("--out", default=str(RAW_DATASET_DIR))
    k.add_argument("--images", default=None, help="folder gambar sumber (opsional; disalin ke <out>/images)")
    k.add_argument("--classes", nargs="+", default=None, help="subset/urutan kategori (default: semua, urut id COCO)")
    k.add_argument("--keep-crowd", action="store_true", help="ikutkan anotasi iscrowd=1 (default: dilewati)")
    k.add_argument("--overwrite", action="store_true")

    y = sub.add_parser("yolo-to-coco", help="dataset YOLO -> satu COCO JSON")
    y.add_argument("--root", default=str(RAW_DATASET_DIR))
    y.add_argument("--classes-file", default=None)
    y.add_argument("--out", default=str(RAW_DATASET_DIR / "annotations_coco.json"))

    t = sub.add_parser("split", help="bagi train/val/test (reproducible, anti-kebocoran)")
    t.add_argument("--src", default=str(RAW_DATASET_DIR))
    t.add_argument("--dst", default=str(DATASET_DIR))
    t.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--group-regex", default=None,
                   help="regex; grup pertama = kunci adegan/video. Gambar satu grup selalu satu split. "
                        "Contoh: '^(.*)_frame\\d+$'")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--no-update-data-yaml", action="store_true")
    t.add_argument("--force", action="store_true", help="tetap bagi walau dataset sumber punya ERROR")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "coco-subset":
            s = coco_subset(args.classes, args.count, args.background, args.seed, resolve_path(args.out),
                            resolve_path(args.coco_dir), args.download, args.overwrite)
            print(f"Kelas        : {', '.join(f'{i}={c}' for i, c in enumerate(s['classes']))}")
            print(f"Gambar       : {s['images_with_objects']} berobjek + {s['background_images']} latar")
            print(f"Kotak/kelas  : {s['boxes_per_class']}")
            print(f"Dilewati     : crowd={s['skipped_crowd']} tidak valid={s['skipped_invalid']} "
                  f"file hilang={s['missing_image_files']}")
            print(f"Selanjutnya  : python prepare_dataset.py check --root {args.out} --visualize 12")
            return 0
        if args.cmd == "check":
            root = resolve_path(args.root)
            if not (root / "images").exists():
                raise YoloSetupError(f"Folder {root / 'images'} tidak ada. Siapkan dataset dulu (lihat README).")
            rep = check_dataset(root, classes_for_root(root, args.classes_file), args.visualize, args.seed,
                                args.min_side, root.name)
            print(format_report(rep))
            print(f"Laporan lengkap: {save_report(rep, root.name)}")
            return 1 if rep["num_errors"] else 0
        if args.cmd == "coco-to-yolo":
            rep = coco_to_yolo(resolve_path(args.coco), resolve_path(args.out),
                               resolve_path(args.images) if args.images else None, args.classes,
                               args.keep_crowd, args.overwrite)
            print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=2, ensure_ascii=False))
            for m in rep["problems"][:10]:
                print("  [!]", m)
            print(f"Laporan: {resolve_path(args.out) / 'conversion_report.json'}")
            return 0
        if args.cmd == "yolo-to-coco":
            root = resolve_path(args.root)
            res = yolo_to_coco(root, resolve_path(args.out), classes_for_root(root, args.classes_file))
            print(res)
            return 0
        if args.cmd == "split":
            s = split_dataset(resolve_path(args.src), resolve_path(args.dst), args.ratios, args.seed, args.group_regex,
                              args.overwrite, not args.no_update_data_yaml, args.force)
            print(f"Seed {s['seed']} | gambar per split: {s['images']} | grup: {s['groups']}")
            for sp, d in s["objects_per_class_per_split"].items():
                print(f"  {sp:6s}: {d}")
            for w in s["warnings"][:10]:
                print("  [WARNING]", w)
            print(f"Sidik jari label: {s['label_fingerprint']}")
            if not args.no_update_data_yaml:
                print(f"configs/data.yaml diperbarui. Selanjutnya: python train_yolo.py --exp e1")
            return 0
    except YoloSetupError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
