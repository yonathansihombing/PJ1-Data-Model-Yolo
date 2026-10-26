#!/usr/bin/env python3
"""Convert camera project's YOLO segmentation polygons to YOLO detection boxes and split data.

Input expected:
  datasets/raw_yolo/images/**/*.{jpg,png,...}
  datasets/raw_yolo/labels_seg/<relative path joined with __>.txt
  datasets/raw_yolo/classes.txt
Output (for RET503 detection experiments):
  datasets/object_detection/images/{train,val,test}/...
  datasets/object_detection/labels/{train,val,test}/...
  configs/data.yaml

This does NOT change or overwrite the original polygon labels. Each polygon becomes the enclosing
axis-aligned bounding box, formatted as `class_id cx cy w h`, normalized to [0,1].
"""
from __future__ import annotations
import argparse, json, random, shutil, sys
from pathlib import Path
from collections import Counter
import cv2

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'datasets' / 'raw_yolo'
OUT = ROOT / 'datasets' / 'object_detection'
CONFIG = ROOT / 'configs' / 'data.yaml'
EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

def read_classes():
    p=RAW/'classes.txt'
    if not p.exists(): raise RuntimeError(f'classes.txt tidak ditemukan: {p}')
    names=[x.strip() for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if not names: raise RuntimeError('classes.txt kosong')
    return names

def image_label_path(img:Path, src:Path):
    # The camera annotation tool names labels as <class>__<image_stem>.txt,
    # while images retain their original filename. Support both this convention
    # and the ordinary mirrored-path convention.
    rel=img.relative_to(src/'images').with_suffix('')
    key='__'.join(rel.parts)
    labels_dir=src/'labels_seg'
    exact=labels_dir/f'{key}.txt'
    if exact.exists():
        return exact
    suffix=f'__{img.stem}.txt'
    matches=sorted(p for p in labels_dir.rglob('*.txt') if p.name.endswith(suffix))
    if len(matches)==1:
        return matches[0]
    if len(matches)>1:
        raise RuntimeError(f'Lebih dari satu anotasi cocok untuk {img.name}: ' + ', '.join(str(p) for p in matches))
    return exact

def parse_polygon_label(label:Path,w:int,h:int,nc:int):
    rows=[]; problems=[]
    if not label.exists(): return rows, ['label hilang']
    for lineno,line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip(): continue
        t=line.split()
        try:
            cls=int(t[0]); vals=list(map(float,t[1:]))
        except Exception:
            problems.append(f'baris {lineno}: format angka salah'); continue
        if cls<0 or cls>=nc: problems.append(f'baris {lineno}: class_id {cls} di luar rentang'); continue
        if len(vals)<6 or len(vals)%2: problems.append(f'baris {lineno}: polygon perlu >=3 titik (x,y)'); continue
        if any(v<0 or v>1 for v in vals): problems.append(f'baris {lineno}: koordinat di luar 0..1'); continue
        xs=vals[0::2]; ys=vals[1::2]
        xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
        if xmax<=xmin or ymax<=ymin: problems.append(f'baris {lineno}: kotak hasil polygon nol luas'); continue
        cx=(xmin+xmax)/2; cy=(ymin+ymax)/2; bw=xmax-xmin; bh=ymax-ymin
        rows.append(f'{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return rows,problems

def main():
    ap=argparse.ArgumentParser(description='Validasi polygon, konversi ke YOLO detection AABB, lalu split train/val/test.')
    ap.add_argument('--src',type=Path,default=RAW); ap.add_argument('--dst',type=Path,default=OUT)
    ap.add_argument('--ratios',nargs=3,type=float,default=[.7,.2,.1],metavar=('TRAIN','VAL','TEST'))
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--allow-unlabeled',action='store_true',help='izinkan gambar tanpa polygon; label detection kosong')
    args=ap.parse_args()
    src=args.src.resolve(); dst=args.dst.resolve()
    if abs(sum(args.ratios)-1)>1e-6 or min(args.ratios)<=0: raise RuntimeError('Rasio split harus positif dan totalnya 1.0')
    classes_path=src/'classes.txt'
    if not classes_path.exists(): raise RuntimeError(f'classes.txt tidak ditemukan: {classes_path}')
    names=[s.strip() for s in classes_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    imgs=sorted(p for p in (src/'images').rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    if not imgs: raise RuntimeError(f'Tidak ada gambar di {src}/images')
    if dst.exists() and any(dst.iterdir()) and not args.overwrite: raise RuntimeError(f'{dst} sudah berisi data. Gunakan --overwrite jika ingin membuat ulang.')
    records=[]; errors=[]; perclass=Counter()
    for img in imgs:
        im=cv2.imread(str(img))
        if im is None: errors.append(f'Gambar rusak/tidak terbaca: {img}'); continue
        h,w=im.shape[:2]
        try:
            lp=image_label_path(img,src)
        except RuntimeError as exc:
            errors.append(str(exc)); continue
        rows,problems=parse_polygon_label(lp,w,h,len(names))
        if problems and not (problems==['label hilang'] and args.allow_unlabeled):
            errors.extend(f'{lp}: {x}' for x in problems)
        if not lp.exists() and not args.allow_unlabeled: errors.append(f'Label tidak ditemukan: {lp}')
        for row in rows: perclass[int(row.split()[0])]+=1
        records.append((img,rows))
    if errors:
        print('VALIDASI GAGAL; tidak ada output dibuat:')
        for e in errors[:60]: print(' -',e)
        if len(errors)>60: print(f' ... dan {len(errors)-60} masalah lainnya')
        return 2
    rng=random.Random(args.seed); rng.shuffle(records)
    n=len(records); ntr=round(n*args.ratios[0]); nval=round(n*args.ratios[1]);
    if n>=3:
        ntr=max(1,min(n-2,ntr)); nval=max(1,min(n-ntr-1,nval))
    splits={'train':records[:ntr],'val':records[ntr:ntr+nval],'test':records[ntr+nval:]}
    if dst.exists() and args.overwrite: shutil.rmtree(dst)
    for split,items in splits.items():
        for img,rows in items:
            rel=img.relative_to(src/'images')
            outimg=dst/'images'/split/rel; outlab=dst/'labels'/split/rel.with_suffix('.txt')
            outimg.parent.mkdir(parents=True,exist_ok=True); outlab.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(img,outimg); outlab.write_text('\n'.join(rows)+('\n' if rows else ''),encoding='utf-8')
    config=ROOT/'configs'/'data.yaml'; config.parent.mkdir(parents=True,exist_ok=True)
    # use absolute POSIX path to avoid Windows relative path ambiguity
    import yaml
    data={'path':dst.as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':{i:n for i,n in enumerate(names)}}
    config.write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding='utf-8')
    summary={'source':str(src),'output':str(dst),'seed':args.seed,'ratios':args.ratios,'images':{k:len(v) for k,v in splits.items()},'classes':names,'objects_per_class_total':{names[k]:v for k,v in sorted(perclass.items())},'format':'YOLO detection class_id cx cy w h converted from polygon enclosing boxes','note':'split acak per gambar; jika beberapa gambar adalah frame dari adegan/video sama, gunakan group split untuk menghindari leakage.'}
    (dst/'split_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False)); print(f'\n[OK] data.yaml dibuat: {config}')
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e: print('[ERROR]',e,file=sys.stderr); raise SystemExit(2)
