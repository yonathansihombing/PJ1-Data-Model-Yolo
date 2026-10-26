#!/usr/bin/env python3
"""
evaluate_yolo.py
================
Evaluasi dan perbandingan E1, E2, E3 secara konsisten (dataset, split, definisi kelas, dan prosedur evaluasi sama).

Dua tahap (sesuai ketentuan tugas):
  1) VAL  - dipakai untuk memantau dan MEMILIH model (boleh dijalankan berulang):
        python evaluate_yolo.py                      # selalu E1, E2, E3 (yang belum dilatih ditandai BELUM DIJALANKAN)
     Hasil: runs/summary_val.csv, runs/comparison.md, runs/comparison_val.png (bila matplotlib ada),
            runs/_evaluation/<nama>_val/ (confusion matrix, kurva PR/F1) dan *_per_class.csv.
  2) TEST - dipakai SEKALI di akhir pada model yang terpilih (mAP50-95 val tertinggi):
        python evaluate_yolo.py --final-test
     Hasil: runs/final_test.json, runs/final_test.csv. Skrip MENOLAK menjalankan test kedua kali
     (kecuali --force-retest, yang dicatat sebagai pelanggaran prosedur di file hasil).

Eksperimen yang belum dilatih ditandai "BELUM DIJALANKAN"; tidak ada angka yang dikarang.
Metrik: Precision (P), Recall (R), mAP@50, mAP@50-95, hasil per kelas, kecepatan inferensi (ms/gambar).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yolo_config as yc
from yolo_config import YoloSetupError

EVAL_DIR = yc.RUNS_DIR / "_evaluation"
SUMMARY_VAL_CSV = yc.RUNS_DIR / "summary_val.csv"
COMPARISON_MD = yc.RUNS_DIR / "comparison.md"
COMPARISON_PNG = yc.RUNS_DIR / "comparison_val.png"
FINAL_TEST_JSON = yc.RUNS_DIR / "final_test.json"
FINAL_TEST_CSV = yc.RUNS_DIR / "final_test.csv"
EXP_ORDER = ["e1", "e2", "e3"]
NOT_RUN = "BELUM DIJALANKAN"
ARTIFACTS = ["weights/best.pt", "weights/last.pt", "results.csv", "results.png", "confusion_matrix.png", "args.yaml"]
FIELDS = ["experiment", "run_name", "status", "model", "pretrained", "changed_vs_e1", "imgsz", "epochs", "batch", "seed",
          "precision", "recall", "map50", "map50_95", "inference_ms", "train_minutes", "weights", "missing_artifacts"]


def _f(x, nd=4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def import_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as e:
        raise YoloSetupError("Paket 'ultralytics' belum terpasang. Lihat README bagian B (PyTorch + pip install ultralytics).") from e


def sha_short(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12].upper()


def read_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "experiment_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def extract_metrics(m: Any) -> Dict[str, Any]:
    """Ambil metrik dari objek hasil .val() Ultralytics (m.box.*). Per kelas bila tersedia."""
    box = m.box
    out = {"precision": _f(box.mp), "recall": _f(box.mr), "map50": _f(box.map50), "map50_95": _f(box.map),
           "inference_ms": _f(getattr(m, "speed", {}).get("inference"), 2), "per_class": []}
    names = getattr(m, "names", {}) or {}
    try:
        idx = list(getattr(box, "ap_class_index", []))
        for j, ci in enumerate(idx):
            out["per_class"].append({"class_id": int(ci), "class": names.get(int(ci), str(ci)),
                                     "precision": _f(box.p[j]), "recall": _f(box.r[j]),
                                     "ap50": _f(box.ap50[j]), "ap50_95": _f(box.ap[j])})
    except Exception as e:                                               # API berbeda antarversi -> jangan gagal total
        out["per_class_error"] = f"hasil per kelas tidak tersedia pada versi Ultralytics ini: {e}"
    return out


def run_val(YOLO, weights: Path, data_yaml: Path, split: str, imgsz: int, batch: int, device: Optional[str], tag: str):
    kw = dict(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, plots=True, verbose=False,
              project=str(EVAL_DIR), name=tag, exist_ok=True)
    if device is not None:
        kw["device"] = device
    return YOLO(str(weights)).val(**kw)


def write_per_class(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class", "precision", "recall", "ap50", "ap50_95"])
        w.writeheader()
        w.writerows(rows)


def describe_change(e1: Dict, cfg: Dict) -> str:
    d = yc.diff_experiments(e1, cfg)
    return "-" if not d else "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in d.items())


def evaluate_val(keys: List[str], batch: int, device: Optional[str], yolo_cls=None) -> List[Dict[str, Any]]:
    cfgs = {k: yc.load_experiment(k) for k in EXP_ORDER}
    errs = yc.check_experiment_design(cfgs["e1"], cfgs["e2"], cfgs["e3"])
    if errs:
        raise YoloSetupError("Desain eksperimen tidak memenuhi aturan tugas:\n  - " + "\n  - ".join(errs))
    data_cfg = yc.load_data_config(yc.resolve_path(cfgs["e1"]["data"]))
    yc.ensure_dataset_ready(data_cfg)
    resolved = yc.write_resolved_data_yaml(yc.resolve_path(cfgs["e1"]["data"]))
    YOLO = None
    rows: List[Dict[str, Any]] = []
    for k in keys:
        cfg = cfgs[k]
        run_dir = yc.RUNS_DIR / cfg["name"]
        best = run_dir / "weights" / "best.pt"
        meta = read_meta(run_dir)
        tc = meta.get("train_config") or cfg["train"]
        row: Dict[str, Any] = {
            "experiment": k.upper(), "run_name": cfg["name"], "status": NOT_RUN, "model": cfg["model"],
            "pretrained": cfg["pretrained"], "changed_vs_e1": describe_change(cfgs["e1"], cfg),
            "imgsz": tc.get("imgsz"), "epochs": tc.get("epochs"), "batch": tc.get("batch"), "seed": tc.get("seed"),
            "precision": None, "recall": None, "map50": None, "map50_95": None, "inference_ms": None,
            "train_minutes": meta.get("train_minutes"), "weights": str(best) if best.exists() else "",
            "missing_artifacts": ",".join(a for a in ARTIFACTS if not (run_dir / a).exists())}
        if not best.exists():
            print(f"[INFO] {k.upper()}: {best} belum ada -> {NOT_RUN}. Jalankan: python train_yolo.py --exp {k}")
            rows.append(row)
            continue
        YOLO = YOLO or (yolo_cls or import_yolo())
        print(f"[EVAL] {k.upper()} pada split VAL (imgsz={row['imgsz']}) ...")
        res = run_val(YOLO, best, resolved, "val", int(row["imgsz"] or cfg["train"]["imgsz"]), batch, device, f"{cfg['name']}_val")
        met = extract_metrics(res)
        row.update(status="OK", **{k2: met[k2] for k2 in ("precision", "recall", "map50", "map50_95", "inference_ms")})
        if met["per_class"]:
            write_per_class(met["per_class"], EVAL_DIR / f"{cfg['name']}_val_per_class.csv")
        elif "per_class_error" in met:
            print(f"[WARNING] {met['per_class_error']}")
        rows.append(row)
    write_summary(rows)
    return rows


def write_summary(rows: List[Dict[str, Any]]):
    yc.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_VAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS} for r in rows])
    lines = ["# Perbandingan eksperimen (split VAL)", "",
             "| Eksperimen | Status | Perubahan vs E1 | imgsz | Precision | Recall | mAP@50 | mAP@50-95 | ms/gambar | menit latih |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = lambda x: "-" if x is None else x
        lines.append(f"| {r['experiment']} | {r['status']} | {r['changed_vs_e1']} | {v(r['imgsz'])} | {v(r['precision'])} | "
                     f"{v(r['recall'])} | {v(r['map50'])} | {v(r['map50_95'])} | {v(r['inference_ms'])} | {v(r['train_minutes'])} |")
    ok = [r for r in rows if r["status"] == "OK"]
    if ok:
        best = max(ok, key=lambda r: r["map50_95"])
        lines += ["", f"Model terpilih (mAP@50-95 val tertinggi): **{best['experiment']}**",
                  "", "Angka berasal dari evaluasi nyata `best.pt` pada split val. Test set belum/tidak dipakai di tabel ini."]
    else:
        lines += ["", "Belum ada eksperimen yang selesai dilatih."]
    COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _chart(ok)


def _chart(ok: List[Dict[str, Any]]):
    if not ok:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib tidak ada -> grafik perbandingan dilewati (tabel CSV/MD tetap dibuat).")
        return
    names = [r["experiment"] for r in ok]
    x = range(len(ok))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], [r["map50"] for r in ok], 0.4, label="mAP@50")
    ax.bar([i + 0.2 for i in x], [r["map50_95"] for r in ok], 0.4, label="mAP@50-95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("nilai (val)")
    ax.set_title("Perbandingan E1 / E2 / E3 (split val)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=120)
    plt.close(fig)


def load_summary() -> List[Dict[str, Any]]:
    rows = []
    with open(SUMMARY_VAL_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("precision", "recall", "map50", "map50_95", "inference_ms", "train_minutes"):
                r[k] = _f(r[k]) if r.get(k) not in ("", None) else None
            rows.append(r)
    return rows


def final_test(args: argparse.Namespace, yolo_cls=None) -> Dict[str, Any]:
    prior = None
    if FINAL_TEST_JSON.exists():
        prior = json.loads(FINAL_TEST_JSON.read_text(encoding="utf-8"))
        if not args.force_retest:
            raise YoloSetupError(
                f"Test set SUDAH dipakai ({prior.get('timestamp')}, model {prior.get('selected_experiment')}). "
                f"Test hanya boleh dipakai sekali untuk laporan akhir.\n  Hasil tercatat: {FINAL_TEST_JSON}\n"
                f"  Bila benar-benar harus mengulang, gunakan --force-retest (akan dicatat sebagai pelanggaran prosedur).")
    cfgs = {k: yc.load_experiment(k) for k in EXP_ORDER}
    data_path = yc.resolve_path(cfgs["e1"]["data"])
    data_cfg = yc.load_data_config(data_path)
    yc.ensure_dataset_ready(data_cfg, need_test=True)
    if not SUMMARY_VAL_CSV.exists():
        print("[INFO] summary_val.csv belum ada -> evaluasi VAL dijalankan dulu untuk memilih model.")
        evaluate_val(EXP_ORDER, args.batch, args.device, yolo_cls)
    rows = [r for r in load_summary() if r["status"] == "OK"]
    if not rows:
        raise YoloSetupError("Belum ada model yang dilatih dan dievaluasi di val. Jalankan training lalu: python evaluate_yolo.py")
    if args.select:
        chosen = next((r for r in rows if r["experiment"].lower() == args.select.lower()), None)
        if chosen is None:
            raise YoloSetupError(f"Eksperimen '{args.select}' tidak punya hasil val yang valid. Tersedia: {[r['experiment'] for r in rows]}")
        how = "manual (--select); disarankan memakai pemilihan otomatis berdasarkan mAP@50-95 val"
    else:
        chosen = max(rows, key=lambda r: r["map50_95"])
        how = "otomatis: mAP@50-95 val tertinggi"
    weights = Path(chosen["weights"])
    if not weights.exists():
        raise YoloSetupError(f"Bobot terpilih tidak ditemukan: {weights}")
    resolved = yc.write_resolved_data_yaml(data_path)
    YOLO = yolo_cls or import_yolo()
    imgsz = int(chosen["imgsz"] or cfgs["e1"]["train"]["imgsz"])
    print(f"[FINAL TEST] model {chosen['experiment']} ({how}) pada split TEST - ini dilakukan SEKALI.")
    res = run_val(YOLO, weights, resolved, "test", imgsz, args.batch, args.device, f"{chosen['run_name']}_test")
    met = extract_metrics(res)
    result = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "selected_experiment": chosen["experiment"],
              "selection": how, "selected_val_map50_95": chosen["map50_95"], "weights": str(weights),
              "weights_sha256_12": sha_short(weights), "imgsz": imgsz,
              "dataset_fingerprint": yc.dataset_fingerprint(Path(data_cfg["path"])),
              "test_counts": yc.dataset_counts(data_cfg).get("test"),
              "test_metrics": {k: met[k] for k in ("precision", "recall", "map50", "map50_95", "inference_ms")},
              "per_class": met["per_class"], "retest_count": 0, "previous_results": []}
    if prior:
        result["retest_count"] = int(prior.get("retest_count", 0)) + 1
        result["previous_results"] = prior.get("previous_results", []) + [
            {k: prior.get(k) for k in ("timestamp", "selected_experiment", "test_metrics")}]
        result["procedure_note"] = "PELANGGARAN PROSEDUR: test set dipakai lebih dari sekali (--force-retest)."
    FINAL_TEST_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(FINAL_TEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["selected", "precision", "recall", "map50", "map50_95", "inference_ms", "timestamp"])
        t = result["test_metrics"]
        w.writerow([chosen["experiment"], t["precision"], t["recall"], t["map50"], t["map50_95"], t["inference_ms"], result["timestamp"]])
    if met["per_class"]:
        write_per_class(met["per_class"], EVAL_DIR / f"{chosen['run_name']}_test_per_class.csv")
    print(f"[OK] Hasil TEST: mAP50={t['map50']} mAP50-95={t['map50_95']} P={t['precision']} R={t['recall']}")
    print(f"     Disimpan: {FINAL_TEST_JSON}\n     Jangan memakai test set lagi untuk memilih/menyetel model.")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluasi E1/E2/E3 (val) dan uji akhir sekali di test.")
    p.add_argument("--final-test", action="store_true", help="uji model terpilih di split TEST (hanya sekali)")
    p.add_argument("--select", default=None, help="paksa pilih eksperimen untuk test (mis. E3); default otomatis")
    p.add_argument("--force-retest", action="store_true", help="izinkan test kedua kali (dicatat sebagai pelanggaran prosedur)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="cpu | 0 (default otomatis)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(yc.PROJECT_ROOT)
    try:
        if args.final_test:
            final_test(args)
        else:
            rows = evaluate_val(EXP_ORDER, args.batch, args.device)
            print(f"\nRingkasan: {SUMMARY_VAL_CSV}\nTabel    : {COMPARISON_MD}")
            for r in rows:
                print(f"  {r['experiment']}: {r['status']}  mAP50={r['map50']}  mAP50-95={r['map50_95']}")
        return 0
    except YoloSetupError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan (Ctrl+C).")
        return 130
    except Exception:
        print("[ERROR] Evaluasi gagal dengan error tak terduga:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
