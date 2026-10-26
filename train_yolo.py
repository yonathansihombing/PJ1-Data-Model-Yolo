#!/usr/bin/env python3
"""
train_yolo.py
=============
Menjalankan eksperimen YOLO RET503. Tiga eksperimen terpisah, masing-masing dengan konfigurasi dan folder hasil sendiri:

  E1  configs/e1_pretrained.yaml   transfer learning (model .pt, bobot pretrained COCO)  -> baseline
  E2  configs/e2_scratch.yaml      dari awal (arsitektur .yaml, bobot acak, TANPA pretrained)
  E3  configs/e3_finetuning.yaml   E1 + tepat SATU perubahan hyperparameter

Pemakaian (dari folder project):
  python train_yolo.py --exp e1
  python train_yolo.py --exp e2
  python train_yolo.py --exp e3
  python train_yolo.py --exp all                 # E1 -> E2 -> E3 berurutan
  python train_yolo.py --exp e1 --dry-run        # periksa konfigurasi & dataset TANPA training
  python train_yolo.py --exp e1 --epochs 1 --name-suffix _smoke    # uji cepat (hasil di folder lain)

Hasil: runs/<nama_eksperimen>/ (weights/best.pt, weights/last.pt, results.csv, results.png,
confusion_matrix.png, args.yaml, ... dibuat oleh Ultralytics) + experiment_meta.json (waktu latih, konfigurasi
efektif, versi paket, sidik jari dataset) + experiment_config.yaml.

Aturan yang ditegakkan skrip ini:
  - E2 menolak model .pt; E1/E3 menolak model .yaml.
  - E3 harus berbeda dari E1 pada TEPAT satu hyperparameter (dicek dari file YAML).
  - Tidak ada unduhan diam-diam: bobot pretrained harus Anda taruh di weights/ (atau beri --allow-download).
  - Tidak menimpa hasil lama kecuali --overwrite.
  - Test set TIDAK disentuh di sini (hanya train dan val); test dipakai sekali oleh evaluate_yolo.py --final-test.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yolo_config as yc
from yolo_config import YoloSetupError

EXP_ORDER = ["e1", "e2", "e3"]
# Kunci 'train' yang dikelola skrip (tidak boleh datang dari YAML karena dapat merusak aturan eksperimen)
RESERVED_KEYS = {"data", "project", "name", "exist_ok", "pretrained", "model"}


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override dari CLI diterapkan ke bagian 'train' dan dicatat. Gunakan nilai yang sama untuk E1/E2/E3."""
    used = {}
    for key in ("epochs", "batch", "workers"):
        v = getattr(args, key, None)
        if v is not None:
            cfg["train"][key] = v
            used[key] = v
    if args.device is not None:
        cfg["train"]["device"] = args.device
        used["device"] = args.device
    cfg["_cli_overrides"] = used
    return cfg


def build_train_kwargs(cfg: Dict[str, Any], data_yaml: Path, run_name: str, overwrite: bool, model_ref: str) -> Dict[str, Any]:
    bad = RESERVED_KEYS & set(cfg["train"])
    if bad:
        raise YoloSetupError(f"Kunci {sorted(bad)} tidak boleh ada di bagian 'train' (dikelola skrip).")
    kw = {k: v for k, v in cfg["train"].items() if v is not None}      # device: null -> tidak diteruskan (otomatis)
    kw.update(data=str(data_yaml), project=str(yc.RUNS_DIR), name=run_name, exist_ok=bool(overwrite),
              pretrained=bool(cfg["pretrained"]))
    return kw


def resolve_model_reference(cfg: Dict[str, Any], allow_download: bool) -> str:
    """Kembalikan argumen untuk YOLO(...). TIDAK mengunduh kecuali --allow-download."""
    model = str(cfg["model"])
    if cfg["pretrained"]:
        found = yc.find_pretrained_weights(Path(model).name)
        if found is not None:
            return str(found)
        if allow_download:
            print(f"[INFO] {model} tidak ada lokal; --allow-download aktif -> Ultralytics akan mengunduhnya "
                  f"ke folder kerja ({yc.PROJECT_ROOT}).")
            return Path(model).name
        raise YoloSetupError(
            f"Bobot pretrained '{Path(model).name}' tidak ditemukan.\n"
            f"  Letakkan file tersebut di: {yc.WEIGHTS_DIR}{os.sep}{Path(model).name}\n"
            f"  atau jalankan dengan --allow-download agar Ultralytics mengunduhnya (butuh internet).\n"
            f"  Jika nama model tidak dikenali versi Ultralytics Anda, ganti 'model' di configs/e1_pretrained.yaml "
            f"dan configs/e2_scratch.yaml ke model yang tersedia (sesuai PPT: yolo26n.pt / yolo26n.yaml).")
    local = yc.CONFIGS_DIR / model
    return str(local) if local.exists() else model                       # .yaml bawaan Ultralytics atau file lokal


def prepare(key: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Semua pemeriksaan sebelum training (tidak butuh ultralytics)."""
    cfgs = {k: yc.load_experiment(k) for k in EXP_ORDER}
    errs = yc.check_experiment_design(cfgs["e1"], cfgs["e2"], cfgs["e3"])
    if errs:
        raise YoloSetupError("Desain eksperimen tidak memenuhi aturan tugas:\n  - " + "\n  - ".join(errs))
    cfg = apply_cli_overrides(cfgs[key], args)
    data_path = yc.resolve_path(cfg["data"])
    data_cfg = yc.load_data_config(data_path)
    counts = yc.ensure_dataset_ready(data_cfg)
    run_name = cfg["name"] + (args.name_suffix or "")
    run_dir = yc.RUNS_DIR / run_name
    if (run_dir / "weights" / "best.pt").exists() and not args.overwrite:
        raise YoloSetupError(f"Hasil {run_dir} sudah ada. Tambahkan --overwrite untuk menimpanya, atau pakai "
                             f"--name-suffix _v2 agar disimpan di folder lain.")
    model_ref = resolve_model_reference(cfg, args.allow_download)
    resolved = yc.write_resolved_data_yaml(data_path)
    return {"cfg": cfg, "data_cfg": data_cfg, "counts": counts, "run_name": run_name, "run_dir": run_dir,
            "model_ref": model_ref, "resolved_yaml": resolved, "design_diff_e3": yc.diff_experiments(cfgs["e1"], cfgs["e3"])}


def import_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as e:
        raise YoloSetupError(
            "Paket 'ultralytics' belum terpasang di Python ini.\n"
            "  1) Pasang PyTorch sesuai komputer Anda (lihat README bagian B) lalu\n"
            "  2) pip install ultralytics   (atau: pip install -r requirements.txt)") from e


def run_experiment(key: str, args: argparse.Namespace, yolo_cls=None) -> Dict[str, Any]:
    plan = prepare(key, args)
    cfg = plan["cfg"]
    print("=" * 72)
    print(f"EKSPERIMEN {key.upper()}  ->  {plan['run_name']}")
    print(f"  deskripsi : {cfg['description']}")
    print(f"  model     : {plan['model_ref']}  (pretrained={cfg['pretrained']})")
    print(f"  dataset   : {plan['data_cfg']['path']}  | kelas: {plan['data_cfg']['names_list']}")
    print(f"  jumlah    : {plan['counts']}")
    print(f"  train     : {cfg['train']}")
    if key == "e3":
        print(f"  perbedaan E3 terhadap E1 (harus tepat satu): {plan['design_diff_e3']}")
    if cfg["_cli_overrides"]:
        print(f"  [PERHATIAN] override CLI: {cfg['_cli_overrides']} -> pakai nilai yang SAMA untuk E1, E2, E3 agar adil.")
    print(f"  hasil di  : {plan['run_dir']}")
    print("=" * 72)
    if args.dry_run:
        print("[DRY-RUN] Konfigurasi dan dataset valid. Training tidak dijalankan.")
        return {"dry_run": True, "run_name": plan["run_name"]}

    YOLO = yolo_cls or import_yolo()
    model = YOLO(plan["model_ref"])
    ckpt = getattr(model, "ckpt", "__missing__")
    if ckpt != "__missing__":
        if cfg["pretrained"] is False and ckpt is not None:
            raise YoloSetupError("E2 seharusnya dari awal, tetapi model memuat bobot pretrained (.pt). Dibatalkan.")
        if cfg["pretrained"] is True and ckpt is None:
            raise YoloSetupError("E1/E3 seharusnya memakai bobot pretrained, tetapi tidak ada bobot yang termuat. Dibatalkan.")
    else:
        print("[WARNING] Versi Ultralytics ini tidak menyediakan atribut 'ckpt'; pemeriksaan pretrained/scratch dilewati.")

    kwargs = build_train_kwargs(cfg, plan["resolved_yaml"], plan["run_name"], args.overwrite, plan["model_ref"])
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    model.train(**kwargs)
    minutes = (time.time() - t0) / 60.0

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", plan["run_dir"]))
    if save_dir.resolve() != plan["run_dir"].resolve():
        print(f"[WARNING] Ultralytics menyimpan hasil di {save_dir} (bukan {plan['run_dir']}). "
              f"evaluate_yolo.py membaca runs/<nama>; salin/pindahkan folder bila perlu.")
    meta = {
        "experiment": key, "name": plan["run_name"], "description": cfg["description"],
        "model_reference": plan["model_ref"], "pretrained": cfg["pretrained"],
        "train_config": cfg["train"], "cli_overrides": cfg["_cli_overrides"],
        "e3_diff_vs_e1": plan["design_diff_e3"] if key == "e3" else None,
        "data_yaml": str(plan["resolved_yaml"]), "classes": plan["data_cfg"]["names_list"],
        "dataset_counts": plan["counts"], "dataset_fingerprint": yc.dataset_fingerprint(Path(plan["data_cfg"]["path"])),
        "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "train_minutes": round(minutes, 2),
        "environment": yc.environment_info(), "config_source": cfg["_source"], "command": " ".join(sys.argv),
    }
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        yc.dump_yaml({k: v for k, v in cfg.items() if not k.startswith("_")}, save_dir / "experiment_config.yaml")
    except OSError as e:
        print(f"[WARNING] Gagal menulis metadata eksperimen: {e}")
    best = save_dir / "weights" / "best.pt"
    print(f"\n[OK] {plan['run_name']} selesai dalam {minutes:.1f} menit.")
    print(f"     best.pt : {best} ({'ADA' if best.exists() else 'TIDAK DITEMUKAN - periksa log training'})")
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Training eksperimen YOLO RET503 (E1 pretrained, E2 scratch, E3 tuning).")
    p.add_argument("--exp", required=True, choices=EXP_ORDER + ["all"], help="eksperimen yang dijalankan")
    p.add_argument("--epochs", type=int, default=None, help="override epoch (pakai nilai sama untuk semua eksperimen)")
    p.add_argument("--batch", type=int, default=None, help="override batch size")
    p.add_argument("--workers", type=int, default=None, help="override jumlah dataloader worker (Windows: coba 0 bila error)")
    p.add_argument("--device", default=None, help="cpu | 0 | 0,1 (default: otomatis / dari YAML)")
    p.add_argument("--overwrite", action="store_true", help="timpa hasil eksperimen yang sudah ada")
    p.add_argument("--name-suffix", default="", help="akhiran nama folder hasil, mis. _smoke")
    p.add_argument("--allow-download", action="store_true",
                   help="izinkan Ultralytics mengunduh bobot pretrained yang belum ada di weights/")
    p.add_argument("--dry-run", action="store_true", help="hanya periksa konfigurasi + dataset, tanpa training")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(yc.PROJECT_ROOT)                      # path stabil, tidak bergantung folder terminal
    keys = EXP_ORDER if args.exp == "all" else [args.exp]
    done: List[str] = []
    try:
        for k in keys:
            run_experiment(k, args)
            done.append(k)
    except YoloSetupError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        if done:
            print(f"[INFO] Selesai sebelum error: {done}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan (Ctrl+C). Folder hasil parsial mungkin ada di runs/. "
              "last.pt dapat dipakai untuk melanjutkan dengan Ultralytics (resume).")
        return 130
    except Exception:
        print("\n[ERROR] Training gagal dengan error tak terduga:", file=sys.stderr)
        traceback.print_exc()
        print("[INFO] Lihat README bagian Troubleshooting.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
