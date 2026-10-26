
import argparse
from pathlib import Path

import yolo_config as yc


def main():
    parser = argparse.ArgumentParser(
        description="Perbandingan learning rate YOLO26n dengan pretrained yang sama."
    )
    parser.add_argument(
        "--exp",
        required=True,
        choices=["e1", "e3"],
        help="e1 = lr0 0.01, e3 = lr0 0.005",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_file = {
        "e1": "e1_fixed_lr.yaml",
        "e3": "e3_fixed_lr.yaml",
    }[args.exp]

    cfg = yc.load_experiment(yc.CONFIGS_DIR / config_file)

    expected_lr = {"e1": 0.01, "e3": 0.005}[args.exp]
    if cfg["train"].get("optimizer") != "AdamW":
        raise RuntimeError("Optimizer harus AdamW.")
    if float(cfg["train"].get("lr0", -1)) != expected_lr:
        raise RuntimeError(
            f"Learning rate {args.exp.upper()} tidak sesuai: "
            f"{cfg['train'].get('lr0')}"
        )
    if cfg["pretrained"] is not True or not str(cfg["model"]).endswith(".pt"):
        raise RuntimeError("Kedua eksperimen harus menggunakan model pretrained .pt.")

    run_name = cfg["name"]
    run_dir = yc.RUNS_DIR / run_name
    if (run_dir / "weights" / "best.pt").exists():
        raise RuntimeError(
            f"Folder hasil sudah memiliki best.pt: {run_dir}. "
            "Hasil lama tidak akan ditimpa."
        )

    data_path = yc.resolve_path(cfg["data"])
    data_cfg = yc.load_data_config(data_path)
    counts = yc.ensure_dataset_ready(data_cfg)

    print(f"Eksperimen       : {args.exp.upper()}")
    print(f"Konfigurasi      : {yc.CONFIGS_DIR / config_file}")
    print(f"Model            : {cfg['model']}")
    print(f"Pretrained       : {cfg['pretrained']}")
    print(f"Optimizer        : {cfg['train']['optimizer']}")
    print(f"Learning rate    : {cfg['train']['lr0']}")
    print(f"Epoch             : {cfg['train']['epochs']}")
    print(f"Image size        : {cfg['train']['imgsz']}")
    print(f"Seed              : {cfg['train']['seed']}")
    print(f"Dataset           : {data_cfg['path']}")
    print(f"Jumlah dataset    : {counts}")
    print(f"Folder output     : {run_dir}")

    if args.dry_run:
        print("[OK] Pemeriksaan konfigurasi selesai. Training belum dijalankan.")
        return

    from ultralytics import YOLO

    model_name = Path(str(cfg["model"])).name
    weights = yc.find_pretrained_weights(model_name)
    if weights is None:
        raise RuntimeError(
            f"Bobot pretrained {model_name} tidak ditemukan secara lokal. "
            "Tidak ada unduhan otomatis yang dilakukan."
        )

    model = YOLO(str(weights))

    train_args = {
        k: v for k, v in cfg["train"].items()
        if v is not None
    }
    train_args.update(
        data=str(yc.write_resolved_data_yaml(data_path)),
        project=str(yc.RUNS_DIR),
        name=run_name,
        exist_ok=False,
        pretrained=True,
    )

    print("\n[INFO] Memulai training dengan konfigurasi di atas.")
    model.train(**train_args)
    print(f"\n[OK] Training selesai. Periksa hasil di: {run_dir}")


if __name__ == "__main__":
    main()
