"""Kelola dataset dengan satu format anotasi: YOLO segmentation polygon bertitik bebas.
Semua label final disimpan di datasets/raw_yolo/labels_seg; usulan dipisahkan sampai direview.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import yolo_config as yc

ROOT = yc.RAW_DATASET_DIR
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
LABELS_OBB = ROOT / "labels_obb"
LABELS_SEG = ROOT / "labels_seg"
PENDING_SEG = ROOT / "labels_pending_seg"
PENDING = ROOT / "labels_pending"
PENDING_OBB = ROOT / "labels_pending_obb"
CLASSES = ROOT / "classes.txt"


def image_files(folder: Path = IMAGES) -> List[Path]:
    """Baca gambar langsung di folder maupun semua subfolder di bawahnya."""
    if not folder.exists():
        return []
    return sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in yc.IMAGE_EXTS), key=lambda p: str(p).casefold())


def image_key(path: Path) -> str:
    """Nama label unik berdasarkan path relatif; gambar di subfolder tidak bentrok."""
    rel = path.relative_to(IMAGES).with_suffix("")
    return "__".join(rel.parts)


def choose_image_scope(prompt="Pilih sumber foto:") -> tuple[Path, Optional[int]]:
    """Pilih folder foto; nama subfolder menjadi kelas default untuk anotasi."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    folders = sorted((p for p in IMAGES.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
    print("\\n" + prompt)
    print("[0] Semua foto (folder utama + semua subfolder)")
    for i, folder in enumerate(folders, 1):
        print(f"[{i}] {folder.name} ({len(image_files(folder))} foto)")
    if not folders:
        print("[INFO] Belum ada subfolder di images. Foto di folder utama tetap bisa diproses.")
        return IMAGES, None
    raw = input("Pilih nomor folder [0]: ").strip() or "0"
    if not raw.isdigit() or not 0 <= int(raw) <= len(folders):
        print("[WARNING] Pilihan tidak valid; memakai semua foto.")
        return IMAGES, None
    if int(raw) == 0:
        return IMAGES, None
    selected = folders[int(raw)-1]
    names = read_classes()
    class_id = next((i for i, name in enumerate(names) if name.casefold() == selected.name.casefold()), None)
    if class_id is None:
        answer = input(f"Nama folder '{selected.name}' belum menjadi kelas. Tambahkan sebagai kelas? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            class_id = add_class(selected.name)
            print(f"[OK] Kelas '{selected.name}' ditambahkan dengan ID {class_id}.")
        else:
            print("[INFO] Kelas default tidak ditetapkan; pilih kelas secara manual di anotator.")
    elif class_id is not None:
        print(f"[INFO] Kelas default folder: {names[class_id]} (ID {class_id}).")
    print(f"[INFO] Folder terpilih: {selected.name}")
    return selected, class_id


def scoped_images(folder: Path) -> List[Path]:
    return image_files(folder)


def read_classes() -> List[str]:
    if not CLASSES.exists(): return []
    return [line.strip() for line in CLASSES.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sync_data_yaml(names: List[str]) -> None:
    """Buat configs/data.yaml bila belum ada, lalu sinkronkan registry kelas."""
    cfg = {}
    if yc.DATA_YAML.exists():
        cfg = yc.load_yaml(yc.DATA_YAML)
    cfg.setdefault("path", str(ROOT.relative_to(yc.PROJECT_ROOT)).replace("\\", "/"))
    cfg.setdefault("train", "images")
    cfg.setdefault("val", "images")
    cfg["nc"] = len(names)
    cfg["names"] = {i: name for i, name in enumerate(names)}
    yc.dump_yaml(cfg, yc.DATA_YAML)


def add_class(name: str) -> int:
    name = " ".join(name.strip().split())
    if not name or any(c in name for c in "\r\n\t"):
        raise yc.YoloSetupError("Nama kelas tidak boleh kosong atau mengandung baris baru.")
    names = read_classes()
    if name.casefold() in {n.casefold() for n in names}:
        raise yc.YoloSetupError(f"Kelas '{name}' sudah ada. ID tidak diubah.")
    names.append(name); ROOT.mkdir(parents=True, exist_ok=True)
    CLASSES.write_text("\n".join(names) + "\n", encoding="utf-8")
    sync_data_yaml(names)
    return len(names) - 1


def load_image(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def _clamp_point(x, y, width, height):
    return max(0.0, min(width - 1.0, float(x))), max(0.0, min(height - 1.0, float(y)))


def save_aabb_label(path: Path, boxes, width: int, height: int):
    """Simpan kotak AABB dalam format YOLO detect."""
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, y1 = _clamp_point(min(x1,x2), min(y1,y2), width, height)
        x2, y2 = _clamp_point(max(x1,x2), max(y1,y2), width, height)
        if x2 <= x1 or y2 <= y1: continue
        cx, cy = (x1+x2)/2/width, (y1+y2)/2/height
        bw, bh = (x2-x1)/width, (y2-y1)/height
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_aabb_label(path: Path, width: int, height: int):
    boxes=[]
    if not path.exists(): return boxes
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        vals=line.split()
        if len(vals)!=5: continue
        try: cls,cx,cy,bw,bh=int(vals[0]),*map(float,vals[1:])
        except ValueError: continue
        boxes.append((cls,(cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height))
    return boxes


def rotated_corners(box):
    """Return exact user-clicked corners when present, otherwise rectangle corners."""
    if len(box) > 6 and box[6] is not None:
        return [(float(x), float(y)) for x, y in box[6]]
    _,cx,cy,bw,bh,angle=box
    a=math.radians(angle); ca,sa=math.cos(a),math.sin(a)
    pts=[]
    for dx,dy in [(-bw/2,-bh/2),(bw/2,-bh/2),(bw/2,bh/2),(-bw/2,bh/2)]:
        pts.append((cx+dx*ca-dy*sa, cy+dx*sa+dy*ca))
    return pts


def save_obb_label(path: Path, boxes, width: int, height: int):
    """Simpan YOLO OBB: class dan empat titik sudut ternormalisasi."""
    lines=[]
    for box in boxes:
        cls=box[0]
        pts=[]
        for x,y in rotated_corners(box):
            x,y=_clamp_point(x,y,width,height); pts.extend((x/width,y/height))
        if len(pts)==8: lines.append(str(int(cls))+" "+" ".join(f"{v:.6f}" for v in pts))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")


def read_obb_label(path: Path, width: int, height: int):
    boxes=[]
    if not path.exists(): return boxes
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        vals=line.split()
        if len(vals)!=9: continue
        try: cls=int(vals[0]); coords=list(map(float,vals[1:]))
        except ValueError: continue
        pts=[(coords[i]*width,coords[i+1]*height) for i in range(0,8,2)]
        # Keep the saved four corners exactly; do not refit them with minAreaRect.
        arr=np.asarray(pts,dtype=np.float32)
        rect=cv2.minAreaRect(arr)
        (cx,cy),(bw,bh),angle=rect
        boxes.append((cls,float(cx),float(cy),float(bw),float(bh),float(angle),tuple(pts)))
    return boxes


def _choose_mode(prompt="Pilih mode anotasi: [1] AABB (kotak biasa), [2] OBB (kotak miring) [1]: "):
    answer=input(prompt).strip().lower()
    if answer in ("2","obb"): return "obb"
    return "aabb"


def annotate_one(image_path: Path, mode="aabb", initial_boxes=None, initial_path: Optional[Path]=None, default_class_id: Optional[int]=None) -> str:
    """Drag kiri membuat kotak; OBB dapat diputar dengan tombol , dan .; S simpan."""
    names=read_classes()
    if not names: print("[ERROR] Tambahkan kelas terlebih dahulu lewat menu workflow."); return "quit"
    image=load_image(image_path)
    if image is None: print(f"[WARNING] Gambar rusak/tidak terbaca: {image_path}"); return "skip"
    h,w=image.shape[:2]
    if initial_boxes is not None: boxes=list(initial_boxes)
    elif mode=="obb": boxes=read_obb_label(initial_path or LABELS_OBB/(image_key(image_path)+".txt"),w,h)
    else: boxes=read_aabb_label(initial_path or LABELS/(image_key(image_path)+".txt"),w,h)
    state={"start":None,"end":None,"class_id":(default_class_id if default_class_id is not None and 0 <= default_class_id < len(names) else 0),"selected":len(boxes)-1,"action":None,"last":None,"handle":None,"point_mode": mode=="obb","points":[]}
    scale=min(1.0,1200/max(w,h))
    handle_radius=14  # radius in original-image pixels, converted from screen coordinates below

    def obb_local(b, px, py):
        dx,dy=px-b[1],py-b[2]; a=math.radians(b[5]); ca,sa=math.cos(a),math.sin(a)
        return dx*ca+dy*sa, -dx*sa+dy*ca

    def inside_box(b, px, py):
        if mode=="obb":
            corners=rotated_corners(b)
            poly=np.asarray(corners,dtype=np.float32).reshape((-1,1,2))
            return cv2.pointPolygonTest(poly,(float(px),float(py)),False) >= 0
        _,x1,y1,x2,y2=b
        return min(x1,x2)<=px<=max(x1,x2) and min(y1,y2)<=py<=max(y1,y2)

    def resize_handle(b, px, py):
        # Return nearest corner handle, if clicked close enough.
        if mode=="obb": corners=rotated_corners(b)
        else:
            _,x1,y1,x2,y2=b; corners=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        distances=[(px-cx)**2+(py-cy)**2 for cx,cy in corners]
        j=int(np.argmin(distances))
        return j if distances[j] <= (handle_radius/scale)**2 else None
    win=f"YOLO Annotator {mode.upper()} | drag kiri=tambah | S=simpan | N=lewati | Q=keluar"

    def mouse(event,x,y,flags,param):
        x,y=x/scale,y/scale
        if event==cv2.EVENT_LBUTTONDOWN:
            # OBB 4-point mode: click the four corners in order, then fit a rotated rectangle.
            if mode=="obb" and state["point_mode"]:
                state["points"].append((float(x),float(y)))
                if len(state["points"]) == 4:
                    quad=[_clamp_point(px,py,w,h) for px,py in state["points"]]
                    # Store the four clicks verbatim. Rectangle parameters are only
                    # retained as fallback metadata; drawing/saving uses exact points.
                    arr=np.asarray(quad,dtype=np.float32)
                    (cx,cy),(bw,bh),angle=cv2.minAreaRect(arr)
                    if bw >= 3 and bh >= 3:
                        boxes.append((state["class_id"],float(cx),float(cy),float(bw),float(bh),float(angle),tuple(quad)))
                        state["selected"]=len(boxes)-1
                    state["points"]=[]
                return
            # First prefer resizing a selected box from one of its corner handles.
            sel=state["selected"]
            if 0 <= sel < len(boxes):
                hnd=resize_handle(boxes[sel],x,y)
                if hnd is not None:
                    state.update(action="resize",handle=hnd,last=(x,y)); return
            # Otherwise select/move any existing box clicked inside it.
            hit=next((i for i in range(len(boxes)-1,-1,-1) if inside_box(boxes[i],x,y)),None)
            if hit is not None:
                state["selected"]=hit
                hnd=resize_handle(boxes[hit],x,y)
                if hnd is not None:
                    state.update(action="resize",handle=hnd,last=(x,y))
                else:
                    state.update(action="move",last=(x,y))
                return
            state.update(start=(x,y),end=(x,y),action="draw",last=(x,y))
        elif event==cv2.EVENT_MOUSEMOVE:
            if state["action"]=="draw" and state["start"] is not None:
                state["end"]=(x,y)
            elif state["action"] in ("move","resize") and state["last"] is not None and 0<=state["selected"]<len(boxes):
                dx,dy=x-state["last"][0],y-state["last"][1]; i=state["selected"]; b=boxes[i]
                if state["action"]=="move":
                    if mode=="obb":
                        if len(b)>6 and b[6] is not None:
                            moved=tuple((px+dx,py+dy) for px,py in b[6])
                            boxes[i]=(b[0],b[1]+dx,b[2]+dy,b[3],b[4],b[5],moved)
                        else: boxes[i]=(b[0],b[1]+dx,b[2]+dy,b[3],b[4],b[5])
                    else: boxes[i]=(b[0],b[1]+dx,b[2]+dy,b[3]+dx,b[4]+dy)
                elif mode=="obb":
                    if len(b)>6 and b[6] is not None:
                        # In exact 4-point mode resize the dragged corner only;
                        # the other three clicked corners remain unchanged.
                        pts=list(b[6]); pts[state["handle"]]=_clamp_point(x,y,w,h)
                        arr=np.asarray(pts,dtype=np.float32)
                        (cx,cy),(bw,bh),angle=cv2.minAreaRect(arr)
                        boxes[i]=(b[0],float(cx),float(cy),float(bw),float(bh),float(angle),tuple(pts))
                    else:
                        # Standard rotated rectangle resize.
                        lx,ly=obb_local(b,x,y)
                        bw=max(4.0,2*abs(lx)); bh=max(4.0,2*abs(ly))
                        boxes[i]=(b[0],b[1],b[2],bw,bh,b[5])
                else:
                    cls,x1,y1,x2,y2=b; corners=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                    # Corner indices: TL, TR, BR, BL.
                    cx,cy=corners[state["handle"]]
                    nx,ny=cx+dx,cy+dy
                    if state["handle"]==0: x1,y1=nx,ny
                    elif state["handle"]==1: x2,y1=nx,ny
                    elif state["handle"]==2: x2,y2=nx,ny
                    else: x1,y2=nx,ny
                    boxes[i]=(cls,min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2))
                state["last"]=(x,y)
        elif event==cv2.EVENT_LBUTTONUP:
            if state["action"]=="draw" and state["start"] is not None:
                ax,ay=state["start"]
                if abs(ax-x)>3 and abs(ay-y)>3:
                    if mode=="obb": boxes.append((state["class_id"],(ax+x)/2,(ay+y)/2,abs(x-ax),abs(y-ay),0.0))
                    else: boxes.append((state["class_id"],min(ax,x),min(ay,y),max(ax,x),max(ay,y)))
                    state["selected"]=len(boxes)-1
            state.update(start=None,end=None,action=None,last=None,handle=None)
        elif event==cv2.EVENT_RBUTTONDOWN and boxes:
            # Hapus kotak yang diklik, atau kotak terakhir jika klik di area kosong.
            hit=next((i for i in range(len(boxes)-1,-1,-1) if inside_box(boxes[i],x,y)),len(boxes)-1)
            boxes.pop(hit); state["selected"]=min(len(boxes)-1,state["selected"]); state.update(action=None,last=None,handle=None)
    cv2.namedWindow(win,cv2.WINDOW_NORMAL); cv2.setMouseCallback(win,mouse)
    try:
        while True:
            view=image.copy()
            for i,b in enumerate(boxes):
                cls=b[0]; color=((37*(cls+1))%255,(97*(cls+2))%255,(173*(cls+3))%255)
                if mode=="obb":
                    pts=np.array(rotated_corners(b),dtype=np.int32).reshape((-1,1,2)); cv2.polylines(view,[pts],True,color,3 if i==state["selected"] else 2)
                    x1,y1=pts.reshape(-1,2)[0]
                else:
                    _,x1,y1,x2,y2=b; cv2.rectangle(view,(int(x1),int(y1)),(int(x2),int(y2)),color,3 if i==state["selected"] else 2)
                label=names[cls] if 0<=cls<len(names) else f"ID {cls}"
                cv2.putText(view,f"{i}: {label}",(int(x1),max(18,int(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
            if state["start"] and state["end"]:
                cv2.rectangle(view,tuple(map(int,state["start"])),tuple(map(int,state["end"])),(255,255,255),1)
            if mode=="obb":
                hint=("OBB 4 titik: klik 4 sudut berurutan | E=edit/geser/resize | P=mode 4 titik | "
                      "[/] atau ,/.=putar 5 derajat | A/D=1 derajat")
                for pi,(px,py) in enumerate(state["points"]):
                    cv2.circle(view,(int(px),int(py)),max(5,int(7/scale)),(0,255,255),-1)
                    cv2.putText(view,str(pi+1),(int(px)+8,int(py)-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
                if len(state["points"]) >= 2:
                    preview=np.asarray(state["points"],dtype=np.int32).reshape((-1,1,2))
                    cv2.polylines(view,[preview],False,(0,255,255),2)
            else: hint="AABB: drag badan=geser | drag sudut=ubah ukuran"
            # Draw corner handles on the selected box so resizing is discoverable.
            if 0 <= state["selected"] < len(boxes):
                sb=boxes[state["selected"]]
                hc=rotated_corners(sb) if mode=="obb" else [(sb[1],sb[2]),(sb[3],sb[2]),(sb[3],sb[4]),(sb[1],sb[4])]
                for hx,hy in hc:
                    cv2.circle(view,(int(hx),int(hy)),max(4,int(6/scale)),(255,255,255),-1)
                    cv2.circle(view,(int(hx),int(hy)),max(4,int(6/scale)),(0,0,0),1)
            cv2.putText(view,f"{image_path.name} | {mode.upper()} | kelas={state['class_id']}:{names[state['class_id']]} | kotak={len(boxes)}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            help_line=("Pilih kelas: tekan angka 0-9 | OBB: klik 4 sudut | E=edit | P=4 titik | klik kanan=hapus" if mode=="obb" and state["point_mode"] else "Pilih kelas: tekan angka 0-9 | klik kiri-drag area kosong=tambah | drag badan=geser | drag sudut=ubah ukuran | klik kanan=hapus")
            cv2.putText(view,help_line,(10,max(45,h-32)),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
            cv2.putText(view,hint+" | S simpan | N lewati | Q keluar",(10,max(62,h-12)),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
            cv2.imshow(win,cv2.resize(view,(max(1,int(w*scale)),max(1,int(h*scale)))))
            key=cv2.waitKey(20)&0xFF
            if mode=="obb" and key in (ord('p'),ord('P')):
                state["point_mode"]=True; state["points"]=[]; state["action"]=None
            elif mode=="obb" and key in (ord('e'),ord('E')):
                state["point_mode"]=False; state["points"]=[]; state["action"]=None
            elif ord('0')<=key<=ord('9') and key-ord('0')<len(names): state['class_id']=key-ord('0')
            elif key in (ord('['),ord('-')): state['class_id']=(state['class_id']-1)%len(names)
            elif key in (ord(']'),ord('+'),ord('=')): state['class_id']=(state['class_id']+1)%len(names)
            elif mode=="obb" and key in (ord('.'),ord('d'), ord('D')) and boxes:
                i=state['selected'] if state['selected']>=0 else len(boxes)-1
                b=list(boxes[i]); delta=1 if key in (ord('d'),ord('D')) else 5
                if len(b)>6 and b[6] is not None:
                    pts=rotated_corners(b); cx=sum(p[0] for p in pts)/4; cy=sum(p[1] for p in pts)/4; a=math.radians(delta); ca,sa=math.cos(a),math.sin(a)
                    b[6]=tuple((cx+(x-cx)*ca-(y-cy)*sa, cy+(x-cx)*sa+(y-cy)*ca) for x,y in pts); b[1]=cx; b[2]=cy; b[5]=(b[5]+delta)%180
                else: b[5]=(b[5]+delta)%180
                boxes[i]=tuple(b)
            elif mode=="obb" and key in (ord(','),ord('a'),ord('A')) and boxes:
                i=state['selected'] if state['selected']>=0 else len(boxes)-1
                b=list(boxes[i]); delta=-1 if key in (ord('a'),ord('A')) else -5
                if len(b)>6 and b[6] is not None:
                    pts=rotated_corners(b); cx=sum(p[0] for p in pts)/4; cy=sum(p[1] for p in pts)/4; a=math.radians(delta); ca,sa=math.cos(a),math.sin(a)
                    b[6]=tuple((cx+(x-cx)*ca-(y-cy)*sa, cy+(x-cx)*sa+(y-cy)*ca) for x,y in pts); b[1]=cx; b[2]=cy; b[5]=(b[5]+delta)%180
                else: b[5]=(b[5]+delta)%180
                boxes[i]=tuple(b)
            elif mode=="obb" and key in (ord('['), ord(']')) and boxes:
                i=state['selected'] if state['selected']>=0 else len(boxes)-1
                b=list(boxes[i]); delta=-5 if key==ord('[') else 5
                if len(b)>6 and b[6] is not None:
                    pts=rotated_corners(b); cx=sum(p[0] for p in pts)/4; cy=sum(p[1] for p in pts)/4; a=math.radians(delta); ca,sa=math.cos(a),math.sin(a)
                    b[6]=tuple((cx+(x-cx)*ca-(y-cy)*sa, cy+(x-cx)*sa+(y-cy)*ca) for x,y in pts); b[1]=cx; b[2]=cy; b[5]=(b[5]+delta)%180
                else: b[5]=(b[5]+delta)%180
                boxes[i]=tuple(b)
            elif key in (ord('s'),ord('S')):
                out=(LABELS_OBB if mode=="obb" else LABELS)/(image_key(image_path)+".txt")
                (save_obb_label if mode=="obb" else save_aabb_label)(out,boxes,w,h)
                print(f"[OK] Label {mode.upper()} disimpan: {out}"); return "saved"
            elif key in (ord('n'),ord('N')): return "skip"
            elif key in (ord('q'),27): return "quit"
    finally: cv2.destroyWindow(win)



def migrate_legacy_labels_to_polygon() -> int:
    """Satukan label AABB/OBB lama ke format polygon; tidak menimpa polygon yang sudah ada."""
    LABELS_SEG.mkdir(parents=True, exist_ok=True)
    migrated = 0
    imgs = image_files()
    for ip in imgs:
        key = image_key(ip)
        dest = LABELS_SEG / (key + ".txt")
        if dest.exists():
            continue
        im = load_image(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        obb_path = LABELS_OBB / (key + ".txt")
        aabb_path = LABELS / (key + ".txt")
        if obb_path.exists():
            # YOLO OBB coordinates are already class + four normalized points.
            raw = obb_path.read_text(encoding="utf-8-sig").strip()
            valid = []
            for line in raw.splitlines():
                vals = line.split()
                if len(vals) == 9:
                    try:
                        int(vals[0]); [float(v) for v in vals[1:]]
                        valid.append(" ".join(vals))
                    except ValueError:
                        pass
            if valid:
                dest.write_text("\n".join(valid) + "\n", encoding="utf-8")
                migrated += 1
                continue
        if aabb_path.exists():
            lines=[]
            for line in aabb_path.read_text(encoding="utf-8-sig").splitlines():
                vals=line.split()
                if len(vals) != 5: continue
                try:
                    cls=int(vals[0]); cx,cy,bw,bh=map(float,vals[1:])
                except ValueError: continue
                coords=[cx-bw/2,cy-bh/2,cx+bw/2,cy-bh/2,cx+bw/2,cy+bh/2,cx-bw/2,cy+bh/2]
                coords=[max(0.0,min(1.0,v)) for v in coords]
                lines.append(str(cls)+" "+" ".join(f"{v:.6f}" for v in coords))
            if lines:
                dest.write_text("\n".join(lines)+"\n", encoding="utf-8")
                migrated += 1
    return migrated


def read_seg_label(path: Path, width: int, height: int):
    """Baca YOLO segmentation polygon: class x1 y1 x2 y2 ... (min. 3 titik)."""
    polygons=[]
    if not path.exists(): return polygons
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        vals=line.split()
        if len(vals)<7 or len(vals)%2==0: continue
        try: cls=int(vals[0]); coords=list(map(float,vals[1:]))
        except ValueError: continue
        pts=tuple((coords[i]*width,coords[i+1]*height) for i in range(0,len(coords),2))
        if len(pts)>=3: polygons.append((cls,pts))
    return polygons


def save_seg_label(path: Path, polygons, width: int, height: int):
    """Simpan polygon YOLO segmentation dengan jumlah titik sesuai klik pengguna."""
    lines=[]
    for cls,pts in polygons:
        if len(pts)<3: continue
        vals=[]
        for x,y in pts:
            x,y=_clamp_point(x,y,width,height); vals.extend((x/width,y/height))
        lines.append(str(int(cls))+" "+" ".join(f"{v:.6f}" for v in vals))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")


def _choose_manual_mode():
    """Manual labeling always uses a closed free-point polygon, like template labeling."""
    print("Mode anotasi manual: Polygon (jumlah titik bebas; garis otomatis menutup ke titik pertama).")
    return "seg"


def annotate_polygon_one(image_path: Path, initial_path: Optional[Path]=None, default_class_id: Optional[int]=None) -> str:
    """Annotasi segmentation polygon manual; klik titik bebas, Enter menutup polygon."""
    names=read_classes()
    if not names: print("[ERROR] Tambahkan kelas terlebih dahulu."); return "quit"
    image=load_image(image_path)
    if image is None: print(f"[WARNING] Gambar tidak terbaca: {image_path}"); return "skip"
    h,w=image.shape[:2]; scale=min(1.0,1200/max(w,h))
    label_path=initial_path or LABELS_SEG/(image_key(image_path)+".txt")
    polygons=read_seg_label(label_path,w,h)
    state={"points":[],"class_id":default_class_id if default_class_id is not None and 0<=default_class_id<len(names) else 0}
    win="Manual Label | klik titik bebas | Enter=selesai | S=simpan"
    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            state["points"].append(_clamp_point(x/scale,y/scale,w,h))
        elif event==cv2.EVENT_RBUTTONDOWN:
            if state["points"]: state["points"].pop()
            elif polygons: polygons.pop()
    cv2.namedWindow(win,cv2.WINDOW_NORMAL); cv2.setMouseCallback(win,mouse)
    try:
        while True:
            view=image.copy()
            for i,(cls,pts) in enumerate(polygons):
                arr=np.asarray(pts,dtype=np.int32).reshape(-1,1,2); color=((37*(cls+1))%255,(97*(cls+2))%255,(173*(cls+3))%255)
                cv2.polylines(view,[arr],True,color,3)
                cv2.putText(view,names[cls] if 0<=cls<len(names) else str(cls),tuple(arr[0,0]),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                for j,(px,py) in enumerate(pts): cv2.circle(view,(int(px),int(py)),4,color,-1)
            pts=state["points"]
            if len(pts)>1: cv2.polylines(view,[np.asarray(pts,dtype=np.int32).reshape(-1,1,2)],len(pts)>=3,(0,255,255),2)
            for j,(px,py) in enumerate(pts):
                cv2.circle(view,(int(px),int(py)),6,(0,255,255),-1); cv2.putText(view,str(j+1),(int(px)+7,int(py)-7),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.putText(view,f"Kelas {state['class_id']}:{names[state['class_id']]} | polygon tersimpan={len(polygons)} | titik aktif={len(pts)}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            cv2.putText(view,"Klik titik batas objek; garis otomatis tertutup setelah 3 titik | Enter=selesai (min 3) | Backspace/klik kanan=undo",(10,max(45,h-32)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1)
            cv2.putText(view,"0-9 pilih kelas | S=simpan | N=lewati | Q=keluar",(10,max(62,h-12)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            cv2.imshow(win,cv2.resize(view,(max(1,int(w*scale)),max(1,int(h*scale)))))
            key=cv2.waitKey(20)&0xff
            if key in (10,13):
                if len(pts)>=3: polygons.append((state["class_id"],tuple(pts))); state["points"]=[]
                else: print("[INFO] Polygon butuh minimal 3 titik.")
            elif key in (8,127):
                if pts: pts.pop()
            elif key in (ord('u'),ord('U')):
                if polygons: polygons.pop()
            elif ord('0')<=key<=ord('9') and key-ord('0')<len(names): state['class_id']=key-ord('0')
            elif key in (ord('['),ord('-')): state['class_id']=(state['class_id']-1)%len(names)
            elif key in (ord(']'),ord('+'),ord('=')): state['class_id']=(state['class_id']+1)%len(names)
            elif key in (ord('s'),ord('S')):
                if len(pts)>=3: polygons.append((state['class_id'],tuple(pts))); state['points']=[]
                elif pts: print("[INFO] Selesaikan polygon aktif dengan minimal 3 titik terlebih dahulu."); continue
                save_seg_label(LABELS_SEG/(image_key(image_path)+".txt"),polygons,w,h)
                print(f"[OK] Label segmentation tersimpan: {LABELS_SEG/(image_key(image_path)+'.txt')}"); return "saved"
            elif key in (ord('n'),ord('N')): return "skip"
            elif key in (ord('q'),27): return "quit"
    finally: cv2.destroyWindow(win)


def annotate_polygon_pending():
    names=read_classes()
    if not names: print("Tambahkan kelas dulu."); return
    scope,default_class_id=choose_image_scope("Pilih folder untuk anotasi polygon manual:")
    label_dir=LABELS_SEG
    todo=[p for p in scoped_images(scope) if not (label_dir/(image_key(p)+".txt")).exists()]
    print(f"[INFO] Polygon/segmentation | gambar belum berlabel: {len(todo)}. Jumlah titik bebas (minimal 3).")
    for idx,p in enumerate(todo,1):
        print(f"[{idx}/{len(todo)}] {p.relative_to(IMAGES)}")
        if annotate_polygon_one(p,default_class_id=default_class_id)=="quit": break


def annotate_pending(mode=None) -> None:
    """Satu alur anotasi manual: polygon segmentation dengan titik bebas."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    names = read_classes()
    if not names:
        print("Tambahkan kelas dulu (menu [9] > [1]).")
        return
    migrated = migrate_legacy_labels_to_polygon()
    if migrated:
        print(f"[INFO] {migrated} label lama AABB/OBB dikonversi ke polygon agar semua anotasi satu format.")
    scope, default_class_id = choose_image_scope("Pilih folder + anotasi polygon manual:")
    todo = [ip for ip in scoped_images(scope) if not (LABELS_SEG / (image_key(ip)+".txt")).exists()]
    print(f"[INFO] Folder: {scope.name if scope != IMAGES else 'SEMUA'} | Belum berlabel: {len(todo)}.")
    print("Klik titik di batas objek; mulai 3 titik garis langsung tertutup. Enter=selesai, Backspace/klik kanan=undo, U=hapus polygon terakhir, S=simpan, N=lewati, Q=keluar.")
    for idx, ip in enumerate(todo, 1):
        print(f"[{idx}/{len(todo)}] {ip.relative_to(IMAGES)}")
        outcome = annotate_polygon_one(ip, default_class_id=default_class_id)
        if outcome == "quit":
            print("[INFO] Dihentikan; jalankan lagi untuk melanjutkan.")
            break


def auto_label(model_path: str, conf: float=0.25, mode=None) -> None:
    """Jalankan checkpoint deteksi umum; hanya kelas yang namanya cocok yang diusulkan."""
    try: from ultralytics import YOLO
    except ImportError as e: raise yc.YoloSetupError("Ultralytics belum terpasang. Aktifkan .venv lalu pasang requirements.txt.") from e
    names=read_classes()
    if not names: raise yc.YoloSetupError("Tambahkan kelas dulu sebelum auto-labeling.")
    if not 0.0<=conf<=1.0: raise yc.YoloSetupError("Confidence harus antara 0.0 dan 1.0.")
    mode=mode or _choose_mode("Mode label usulan: [1] AABB, [2] OBB [1]: ")
    model_file=yc.resolve_path(model_path)
    if not model_file.is_file(): raise yc.YoloSetupError(f"Checkpoint tidak ditemukan: {model_file}")
    model=YOLO(str(model_file))
    model_names=model.names if isinstance(model.names,dict) else {i:n for i,n in enumerate(model.names)}
    model_names={int(k):str(v) for k,v in model_names.items()}
    scope, default_class_id = choose_image_scope("Pilih folder foto untuk auto-labeling:")
    pending=PENDING_OBB if mode=="obb" else PENDING
    label_dir=LABELS_OBB if mode=="obb" else LABELS
    pending.mkdir(parents=True,exist_ok=True)
    exact_matches={n.casefold() for n in model_names.values()} & {n.casefold() for n in names}
    if not exact_matches:
        print("[INFO] Tidak ada nama kelas dataset yang cocok persis dengan kelas checkpoint.")
        print("Kelas checkpoint yang tersedia (contoh): " + ", ".join(f"{k}:{v}" for k,v in list(model_names.items())[:30]))
        print("Agar tidak memberi label kelas yang keliru, deteksi tidak akan dipetakan otomatis ke kelas custom.")
        print("Jika objek custom (mis. box_hitam/box_putih/field) tidak dikenali checkpoint, label beberapa foto manual dahulu dan latih model khusus.")
    todo=[p for p in scoped_images(scope) if not (label_dir/(image_key(p)+".txt")).exists()]
    if not todo: print("Tidak ada gambar tanpa label untuk auto-labeling."); return
    print(f"[INFO] Auto-label mode {mode.upper()} | Folder: {scope.name if scope != IMAGES else 'SEMUA'} | {len(todo)} gambar | confidence={conf:.2f}")
    if default_class_id is not None and names[default_class_id].casefold() not in {n.casefold() for n in model_names.values()}:
        print(f"[WARNING] Checkpoint pretrained tidak memiliki kelas '{names[default_class_id]}'. Tidak dibuat kotak palsu; hasil kelas ini mungkin 0 kandidat.")
    for idx,p in enumerate(todo,1):
        image=load_image(p)
        if image is None: print(f"[SKIP] Gambar rusak: {p}"); continue
        result=model.predict(source=image,conf=conf,verbose=False)[0]
        h,w=image.shape[:2]; rows=[]
        for b in result.boxes:
            cls=int(b.cls.item())
            model_name=str(result.names.get(cls,cls)) if isinstance(result.names,dict) else str(result.names[cls])
            target_id=next((i for i,n in enumerate(names) if n.casefold()==model_name.casefold()),None)
            if target_id is None: continue
            x1,y1,x2,y2=[float(v) for v in b.xyxy[0].tolist()]
            if mode=="obb": rows.append((target_id,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,0.0))
            else: rows.append((target_id,x1,y1,x2,y2))
        out=pending/(image_key(p)+".txt")
        (save_obb_label if mode=="obb" else save_aabb_label)(out,rows,w,h)
        print(f"[{idx}/{len(todo)}] {p.relative_to(IMAGES)}: {len(rows)} kandidat -> {out}")
    print("[INFO] Kandidat disimpan sebagai label usulan, belum dipakai training.")
    print("CATATAN: checkpoint pretrained hanya mengusulkan kelas yang namanya cocok dengan kelas proyek. Jika kelas custom tidak cocok, jumlah kandidat bisa 0; periksa dengan menu review.")



def template_auto_label(mode="obb") -> None:
    """Masked template auto-labeling from a user-drawn polygon.

    The user can choose 3-20 points. The polygon mask excludes background outside
    the example object. Results are YOLO segmentation polygons in a separate
    pending directory; never silently treat them as detect/OBB labels.
    """
    names = read_classes()
    if not names:
        raise yc.YoloSetupError("Tambahkan kelas terlebih dahulu.")
    scope, default_class_id = choose_image_scope("Pilih folder gambar untuk template auto-labeling:")
    imgs = scoped_images(scope)
    if len(imgs) < 2:
        print("Butuh minimal 2 gambar dalam folder terpilih."); return
    print("Pilih gambar contoh:")
    for i, ip in enumerate(imgs, 1): print(f"[{i}] {ip.relative_to(IMAGES)}")
    raw = input("Nomor gambar contoh: ").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= len(imgs): print("Pilihan tidak valid."); return
    sample_path = imgs[int(raw)-1]
    sample = load_image(sample_path)
    if sample is None: print("Gambar contoh tidak bisa dibaca."); return
    class_id = default_class_id
    if class_id is None:
        for i, n in enumerate(names): print(f"[{i}] {n}")
        raw = input("ID kelas objek yang dicari: ").strip()
        if not raw.isdigit() or not 0 <= int(raw) < len(names): print("ID kelas tidak valid."); return
        class_id = int(raw)
    else: print(f"Kelas dari folder: {names[class_id]} (ID {class_id})")

    raw_n = input("Jumlah titik polygon objek (3-20) [4]: ").strip() or "4"
    if not raw_n.isdigit() or not 3 <= int(raw_n) <= 20:
        print("Jumlah titik harus 3 sampai 20."); return
    n_points = int(raw_n)
    points=[]
    view_scale=min(1.0,1200/max(sample.shape[:2]))
    win=f"Contoh objek | {n_points} titik | Enter=OK R=reset Q=batal"
    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(points)<n_points:
            points.append((int(x/view_scale),int(y/view_scale)))
    cv2.namedWindow(win,cv2.WINDOW_NORMAL); cv2.setMouseCallback(win,mouse)
    try:
        while True:
            v=sample.copy()
            if len(points)>1:
                cv2.polylines(v,[np.asarray(points,dtype=np.int32).reshape(-1,1,2)],False,(0,255,255),2)
            if len(points)>=3:
                cv2.polylines(v,[np.asarray(points,dtype=np.int32).reshape(-1,1,2)],True,(0,220,0),2)
            for i,(x,y) in enumerate(points):
                cv2.circle(v,(x,y),6,(0,255,255),-1)
                cv2.putText(v,str(i+1),(x+7,y-7),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
            cv2.putText(v,f"Klik {n_points} titik mengelilingi OBJEK, bukan latar ({len(points)}/{n_points})",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            cv2.putText(v,"Enter=lanjut jika titik lengkap | R=ulang | Q=batal",(10,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            cv2.imshow(win,cv2.resize(v,(int(v.shape[1]*view_scale),int(v.shape[0]*view_scale))))
            key=cv2.waitKey(20)&0xff
            if key in (ord('r'),ord('R')): points.clear()
            elif key in (ord('q'),27): return
            elif key in (10,13) and len(points)==n_points: break
    finally: cv2.destroyWindow(win)

    h0,w0=sample.shape[:2]
    pts=np.asarray([_clamp_point(x,y,w0,h0) for x,y in points],dtype=np.int32)
    x0=max(0,int(pts[:,0].min())); y0=max(0,int(pts[:,1].min()))
    x1=min(w0,int(pts[:,0].max())+1); y1=min(h0,int(pts[:,1].max())+1)
    if x1-x0<12 or y1-y0<12: print("Polygon terlalu kecil."); return
    local=(pts-np.array([x0,y0])).astype(np.int32)
    template=sample[y0:y1,x0:x1]
    mask=np.zeros((y1-y0,x1-x0),dtype=np.uint8)
    cv2.fillPoly(mask,[local.reshape(-1,1,2)],255)
    # Shrink the search template to the annotated object bounds; the mask prevents
    # the matcher from rewarding pixels outside the polygon.
    gray_t=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    pending=ROOT/"labels_pending_seg"
    pending.mkdir(parents=True,exist_ok=True)
    threshold_text=input("Ambang kemiripan objek [0.78] (0-1): ").strip()
    try: threshold=float(threshold_text or "0.78")
    except ValueError: print("Ambang tidak valid."); return
    if not 0<=threshold<=1: print("Ambang harus 0 sampai 1."); return
    scales=np.linspace(.70,1.35,18)
    todo=[ip for ip in imgs if ip != sample_path]
    if not todo: print("Tidak ada gambar target."); return
    print(f"Template objek {x1-x0}x{y1-y0}, polygon {n_points} titik; latar di luar polygon diabaikan saat pencocokan.")
    for idx,ip in enumerate(todo,1):
        im=load_image(ip)
        if im is None: print(f"[{idx}/{len(todo)}] {ip.name}: gambar rusak"); continue
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); best=(float('inf'),None,None,None)
        for scale in scales:
            tw,th=max(8,int(gray_t.shape[1]*scale)),max(8,int(gray_t.shape[0]*scale))
            if tw>=gray.shape[1] or th>=gray.shape[0]: continue
            interp=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC
            resized=cv2.resize(gray_t,(tw,th),interpolation=interp)
            resized_mask=cv2.resize(mask,(tw,th),interpolation=cv2.INTER_NEAREST)
            try:
                result=cv2.matchTemplate(gray,resized,cv2.TM_CCORR_NORMED,mask=resized_mask)
                minval,maxval,minloc,maxloc=cv2.minMaxLoc(result)
            except cv2.error: continue
            if maxval>0 and (1.0-maxval)<best[0]: best=(1.0-float(maxval),maxloc,tw,th)
        out=pending/(image_key(ip)+".txt"); hh,ww=im.shape[:2]
        # CCORR_NORMED score maps to similarity in [0,1]; low-confidence results are blank.
        similarity=1.0-best[0] if best[1] is not None else 0.0
        if best[1] is not None and similarity>=threshold:
            xx,yy=best[1]; tw,th=best[2],best[3]
            scaled=[((float(x)-x0)*tw/(x1-x0)+xx, (float(y)-y0)*th/(y1-y0)+yy) for x,y in pts]
            coords=[]
            for px,py in scaled: coords.extend((max(0,min(1,px/ww)),max(0,min(1,py/hh))))
            out.write_text(str(class_id)+" "+" ".join(f"{v:.6f}" for v in coords)+"\n",encoding="utf-8")
            status=f"USULAN POLYGON ({n_points} titik)"
        else:
            out.write_text("",encoding="utf-8"); status="TIDAK YAKIN - kosong"
        print(f"[{idx}/{len(todo)}] {ip.relative_to(IMAGES)}: similarity={similarity:.3f} | {status} -> {out.name}")
    print("Selesai. Label polygon disimpan di datasets/raw_yolo/labels_pending_seg.")
    print("PENTING: ini format YOLO segmentation (kelas + semua titik), bukan format AABB/OBB 4 titik. Jangan dipakai untuk training detect/OBB tanpa konversi.")
    print("Template mask mengabaikan latar di luar polygon, tetapi perubahan sudut pandang/skala/pencahayaan tetap dapat menyebabkan salah deteksi. Periksa setiap usulan.")

def review_pending(mode=None) -> None:
    """Review usulan polygon; hasil yang disimpan masuk ke satu folder label final."""
    names = read_classes()
    if not names:
        print("Tidak ada kelas. Tambahkan kelas terlebih dahulu.")
        return
    scope, _ = choose_image_scope("Pilih folder untuk review usulan polygon:")
    PENDING_SEG.mkdir(parents=True, exist_ok=True)
    by_key = {image_key(p): p for p in scoped_images(scope)}
    files = sorted((PENDING_SEG/(key+".txt") for key in by_key if (PENDING_SEG/(key+".txt")).exists()), key=lambda p:p.name.casefold())
    if not files:
        print("Belum ada usulan polygon di folder ini. Jalankan auto-label polygon terlebih dahulu.")
        return
    for i, lf in enumerate(files, 1):
        ip = by_key.get(lf.stem)
        if ip is None:
            print(f"[SKIP] Gambar pasangan tidak ditemukan untuk {lf.name}")
            continue
        print(f"Review {i}/{len(files)}: {ip.relative_to(IMAGES)} | Enter=selesaikan polygon, S=simpan, N=lewati, Q=keluar")
        im = load_image(ip)
        if im is None: continue
        boxes = read_seg_label(lf, im.shape[1], im.shape[0])
        outcome = annotate_polygon_one(ip, initial_path=lf, default_class_id=(boxes[0][0] if boxes else None))
        if outcome == "saved":
            lf.unlink(missing_ok=True)
            print("[OK] Polygon disimpan ke label final.")
        elif outcome == "quit":
            print("[INFO] Usulan yang belum direview dipertahankan.")
            break
        else:
            print("[INFO] Usulan dilewati dan tetap disimpan.")


def status() -> None:
    migrated = migrate_legacy_labels_to_polygon()
    names = read_classes(); imgs = image_files()
    seg = list(LABELS_SEG.glob("*.txt")) if LABELS_SEG.exists() else []
    pending_seg = list(PENDING_SEG.glob("*.txt")) if PENDING_SEG.exists() else []
    labeled_keys = {p.stem for p in seg}
    unlabeled = [ip for ip in imgs if image_key(ip) not in labeled_keys]
    print(f"Root dataset : {ROOT.resolve()}\nKelas        : {len(names)}")
    for i, name in enumerate(names): print(f"  {i}: {name}")
    print(f"Gambar       : {len(imgs)} (termasuk subfolder)")
    print(f"Label final (polygon): {len(seg)} | Usulan polygon: {len(pending_seg)}")
    print(f"Belum label  : {len(unlabeled)}")
    if migrated: print(f"[INFO] {migrated} label lama dikonversi otomatis ke format polygon.")
    if unlabeled[:10]: print("Contoh belum label: "+", ".join(str(ip.relative_to(IMAGES)) for ip in unlabeled[:10]))



def validate_segmentation() -> None:
    """Validate the unified polygon dataset and report all issues before splitting."""
    names = read_classes()
    imgs = image_files()
    if not names:
        print("[ERROR] classes.txt kosong; tambahkan kelas terlebih dahulu."); return
    if not imgs:
        print("[ERROR] Tidak ada gambar di datasets/raw_yolo/images."); return
    issues=[]; labeled=0; objects=0
    for ip in imgs:
        lp=LABELS_SEG/(image_key(ip)+".txt")
        if not lp.exists():
            issues.append(f"BELUM LABEL: {ip.relative_to(IMAGES)}"); continue
        im=load_image(ip)
        if im is None:
            issues.append(f"GAMBAR RUSAK: {ip.relative_to(IMAGES)}"); continue
        h,w=im.shape[:2]
        lines=lp.read_text(encoding="utf-8-sig").splitlines()
        if not lines:
            issues.append(f"LABEL KOSONG: {ip.relative_to(IMAGES)}"); continue
        valid_here=0
        for li,line in enumerate(lines,1):
            vals=line.split()
            if len(vals)<7 or len(vals)%2==0:
                issues.append(f"FORMAT POLYGON: {lp.name} baris {li} (minimal 3 titik)"); continue
            try:
                cls=int(vals[0]); coords=[float(v) for v in vals[1:]]
            except ValueError:
                issues.append(f"ANGKA TIDAK VALID: {lp.name} baris {li}"); continue
            if not 0<=cls<len(names):
                issues.append(f"ID KELAS {cls} DI LUAR RENTANG: {lp.name} baris {li}"); continue
            if any(not np.isfinite(v) or v<0 or v>1 for v in coords):
                issues.append(f"KOORDINAT DI LUAR 0..1: {lp.name} baris {li}"); continue
            pts=np.asarray(coords,dtype=float).reshape(-1,2)
            if len(np.unique(np.round(pts,6),axis=0))<3:
                issues.append(f"TITIK UNIK KURANG DARI 3: {lp.name} baris {li}"); continue
            area=abs(cv2.contourArea(pts.astype(np.float32)))
            if area < 1e-7:
                issues.append(f"POLYGON LUAS NOL/TERLALU KECIL: {lp.name} baris {li}"); continue
            valid_here+=1; objects+=1
        if valid_here: labeled+=1
    print("\n========== VALIDASI POLYGON SEGMENTATION ==========")
    print(f"Gambar total: {len(imgs)} | gambar dengan label valid: {labeled} | objek valid: {objects}")
    print(f"Masalah ditemukan: {len(issues)}")
    for issue in issues[:100]: print("[CEK] "+issue)
    if len(issues)>100: print(f"... dan {len(issues)-100} masalah lainnya")
    if not issues: print("[OK] Semua pasangan gambar-label lolos pemeriksaan format. Periksa visual mask juga untuk memastikan objek yang dilabel benar.")


def split_segmentation() -> None:
    """Create YOLO segmentation train/val/test dataset with 70/20/10 split."""
    import random, shutil
    names=read_classes(); imgs=image_files()
    if not names: print("[ERROR] classes.txt kosong."); return
    if not imgs: print("[ERROR] Tidak ada gambar."); return
    missing=[ip for ip in imgs if not (LABELS_SEG/(image_key(ip)+".txt")).is_file() or not (LABELS_SEG/(image_key(ip)+".txt")).read_text(encoding="utf-8-sig").strip()]
    if missing:
        print(f"[BATAL] Ada {len(missing)} gambar belum punya label polygon. Jalankan validasi [6] dan lengkapi label dahulu.")
        for ip in missing[:10]: print(" - "+str(ip.relative_to(IMAGES)))
        return
    # Validate all labels before copying; do not silently build a broken dataset.
    bad=[]
    for ip in imgs:
        lp=LABELS_SEG/(image_key(ip)+".txt")
        im=load_image(ip)
        if im is None: bad.append(str(ip.relative_to(IMAGES))+" (gambar rusak)"); continue
        h,w=im.shape[:2]
        for li,line in enumerate(lp.read_text(encoding="utf-8-sig").splitlines(),1):
            v=line.split()
            try: cls=int(v[0]); c=list(map(float,v[1:]))
            except (ValueError,IndexError): bad.append(f"{lp.name}:{li} (format)"); continue
            if len(c)<6 or len(c)%2 or not 0<=cls<len(names) or any(not np.isfinite(x) or x<0 or x>1 for x in c): bad.append(f"{lp.name}:{li} (koordinat/kelas)" )
    if bad:
        print(f"[BATAL] {len(bad)} label bermasalah. Jalankan validasi [6]. Contoh: {', '.join(bad[:5])}"); return
    raw=input("Seed acak pembagian [42]: ").strip() or "42"
    try: seed=int(raw)
    except ValueError: print("Seed harus angka."); return
    random.Random(seed).shuffle(imgs)
    n=len(imgs); n_train=max(1,int(round(n*.70))); n_val=max(1,int(round(n*.20)))
    if n_train+n_val>=n: n_train=max(1,n-2); n_val=1
    groups={"train":imgs[:n_train],"val":imgs[n_train:n_train+n_val],"test":imgs[n_train+n_val:]}
    out=yc.PROJECT_ROOT/"datasets"/"segmentation"
    if out.exists():
        answer=input(f"Folder {out} sudah ada. Timpa folder split segmentation ini? [y/N]: ").strip().lower()
        if answer not in ("y","yes"): print("Dibatalkan; folder lama tidak diubah."); return
        shutil.rmtree(out)
    for split, group in groups.items():
        (out/"images"/split).mkdir(parents=True,exist_ok=True); (out/"labels"/split).mkdir(parents=True,exist_ok=True)
        for ip in group:
            stem=image_key(ip); dest_img=out/"images"/split/(stem+ip.suffix.lower()); dest_lab=out/"labels"/split/(stem+".txt")
            shutil.copy2(ip,dest_img); shutil.copy2(LABELS_SEG/(stem+".txt"),dest_lab)
    yaml_path=out/"data.yaml"
    try:
        import yaml
        data={"path":str(out.resolve()).replace("\\","/"),"train":"images/train","val":"images/val","test":"images/test","names":{i:nm for i,nm in enumerate(names)},"nc":len(names)}
        yaml_path.write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding="utf-8")
    except ImportError:
        # YAML can be written safely for these plain class names without PyYAML.
        lines=[f'path: "{str(out.resolve()).replace(chr(92), "/")}"','train: images/train','val: images/val','test: images/test',f'nc: {len(names)}','names:']
        lines += [f'  {i}: "{nm.replace(chr(34), chr(39))}"' for i,nm in enumerate(names)]
        yaml_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n[OK] Dataset YOLO segmentation selesai dibuat.")
    for split, group in groups.items(): print(f"  {split}: {len(group)} gambar")
    print(f"Data YAML: {yaml_path}")
    print("Berikutnya: python train_segmentation.py --data datasets/segmentation/data.yaml")

def menu() -> None:
    while True:
        print("\n========== DATASET YOLO WORKFLOW ==========")
        print("[1] Tambah kelas objek")
        print("[2] Label manual (polygon; jumlah titik bebas)")
        print("[4] Review/edit usulan auto-label polygon")
        print("[5] Status dataset (semua label disatukan)")
        print("[6] Validasi dataset raw")
        print("[7] Bagi dataset train/val/test (70/20/10) — Polygon segmentation")
        print("[9] Training YOLO Segmentation")
        print("[8] Auto-label polygon dari contoh gambar")
        print("[0] Kembali")
        choice = input("Choose: ").strip()
        try:
            if choice == "0": return
            elif choice == "8": template_auto_label("polygon")
            elif choice == "1":
                name = input("Nama kelas baru: ").strip(); cid = add_class(name)
                print(f"[OK] Kelas '{name}' ditambahkan dengan ID {cid} (ID lama tidak berubah).")
            elif choice == "2": annotate_pending()
            elif choice == "4": review_pending("polygon")
            elif choice == "5": status()
            elif choice == "6": validate_segmentation()
            elif choice == "7": split_segmentation()
            elif choice == "9":
                import subprocess
                subprocess.run([sys.executable, str(yc.PROJECT_ROOT / "train_segmentation.py"), "--data", str(yc.PROJECT_ROOT / "datasets" / "segmentation" / "data.yaml")], check=False)
            else: print("Pilihan tidak valid.")
        except (ValueError, yc.YoloSetupError, OSError) as exc:
            print(f"[ERROR] {exc}")


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("status"); add=sub.add_parser("add-class"); add.add_argument("name")
    sub.add_parser("annotate"); sub.add_parser("template-auto-label"); auto=sub.add_parser("auto-label"); auto.add_argument("--model",required=True); auto.add_argument("--conf",type=float,default=.25); auto.add_argument("--mode",choices=["aabb","obb"],default=None)
    rev=sub.add_parser("review"); rev.add_argument("--mode",choices=["aabb","obb"],default=None)
    args=parser.parse_args(argv)
    try:
        if args.cmd=="status": status()
        elif args.cmd=="add-class": print(f"ID kelas baru: {add_class(args.name)}")
        elif args.cmd=="annotate": annotate_pending()
        elif args.cmd=="template-auto-label": template_auto_label()
        elif args.cmd=="auto-label": auto_label(args.model,args.conf,args.mode)
        elif args.cmd=="review": review_pending(args.mode)
        return 0
    except (ValueError,yc.YoloSetupError,OSError) as exc: print(f"[ERROR] {exc}",file=sys.stderr); return 2

if __name__=="__main__": raise SystemExit(main())
