"""Train YOLO segmentation from the unified polygon dataset.
Usage: python train_segmentation.py --data datasets/segmentation/data.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='datasets/segmentation/data.yaml')
    ap.add_argument('--model', default='yolo11n-seg.pt', help='Checkpoint segmentation, default yolo11n-seg.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default=None, help='Mis. 0 untuk GPU, cpu untuk CPU')
    ap.add_argument('--name', default='camera_polygon_seg')
    args=ap.parse_args()
    data=Path(args.data).expanduser().resolve()
    if not data.is_file():
        print(f'[ERROR] Konfigurasi dataset tidak ditemukan: {data}')
        print('Jalankan menu [9] > [7] untuk validasi dan split dataset segmentation dahulu.')
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print('[ERROR] Ultralytics belum terpasang. Aktifkan .venv lalu jalankan: pip install ultralytics')
        return 2
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        print('[ERROR] epochs, imgsz, dan batch harus bernilai positif.'); return 2
    print(f'Model: {args.model} | data: {data} | epochs: {args.epochs} | imgsz: {args.imgsz}')
    try:
        model=YOLO(args.model)
        kw=dict(data=str(data),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,name=args.name,task='segment')
        if args.device: kw['device']=args.device
        result=model.train(**kw)
        print('[OK] Training selesai. Cari weights/best.pt pada folder runs/segment/ yang dilaporkan Ultralytics.')
        return 0
    except Exception as exc:
        print(f'[ERROR] Training gagal: {exc}')
        return 1

if __name__=='__main__': raise SystemExit(main())
