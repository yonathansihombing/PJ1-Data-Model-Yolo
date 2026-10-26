#!/usr/bin/env python3
"""
main.py  -  MAIN MENU / CONTROLLER  (Vision Pick & Place, Logitech C270)
=========================================================================
File ini HANYA menu. Semua logika ada di modul masing-masing:

    [1] Camera Settings        camera_settings.py     -> camera_settings.json
    [2] Camera Calibration     camera_calibration.py  -> camera_calibration.npz / .json
    [3] Select / Manage Field  field_manager.py       -> fields.json
    [4] Case 2 - Object Test   vision_lapangan.py     (pakai fields.json; calibration OPSIONAL)
    [5] Camera Test            camera_calibration.py  (run_camera_test)
    [6] YOLO Object Detection  yolo_detector.py       (modul BARU, RET503; butuh paket ultralytics)

Aturan:
  - Menu bebas dipilih; tidak ada urutan yang dipaksakan. Calibration boleh di-skip.
  - Tiap mode membuka kamera sendiri dan MELEPASNYA saat keluar (hanya satu pemilik kamera
    pada satu waktu), sehingga mode lain bisa membukanya lagi tanpa "camera busy".
  - Semua mode membuka kamera lewat camera_settings.open_camera() -> camera_settings.json
    otomatis dimuat (hardware di-verifikasi, lalu software brightness/contrast/gamma).
  - Tanpa calibration / homography: mode PIXEL (tidak ada koordinat mm palsu).
  - TIDAK ADA PLC / Modbus / UR3 di tahap ini.

Jalankan:   python main.py            (opsi: python main.py --camera-index 1 --stable-frames 5 ...)
"""

import argparse
import os
import traceback
from typing import List

import cv2

import camera_calibration
import vision_lapangan as vl
from camera_settings import SETTINGS_FILE, run_camera_settings
from field_manager import FieldStore, load_calibration_if_available, run_field_manager

CALIBRATION_BASE = "camera_calibration"
DEFAULT_COLS, DEFAULT_ROWS, DEFAULT_SQUARE_MM = 7, 7, 34.0     # papan 8x8 kotak -> 7x7 inner corner, 34 mm

BANNER = """
================================================
VISION PICK & PLACE
================================================
"""
MENU = """[1] Camera Settings
[2] Camera Calibration
[3] Select / Manage Field
[4] Case 2 - Object Detection Test
[5] Camera Test
[6] YOLO Object Detection
[7] Capture Gambar Dataset (kamera)
[8] Training / Transfer Learning YOLO
[9] Kelola Dataset / Kelas / Anotasi / Auto-labeling
[10] Evaluasi VAL + bandingkan E1/E2/E3
[11] Final TEST (hanya jika belum pernah dilakukan)
[12] Buat laporan TXT/CSV/README dari hasil yang tersedia
[0] Exit
"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision Pick & Place - Main Menu")
    vl.add_arguments(p)
    p.set_defaults(source="webcam")                      # menu memakai webcam (C270) kecuali diminta lain
    return p.parse_args(argv)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "0"


def cleanup_windows():
    """Tutup semua window OpenCV dan beri waktu GUI memprosesnya (penting di Windows)."""
    cv2.destroyAllWindows()
    for _ in range(3):
        cv2.waitKey(1)


def run_mode(fn, *args):
    """Jalankan satu mode; error/exit di dalam mode TIDAK menutup program menu."""
    try:
        return fn(*args)
    except SystemExit as e:                              # mis. sys.exit("[ERROR] ...") di camera_calibration
        if e.code not in (None, 0):
            print(e.code if isinstance(e.code, str) else f"[ERROR] exit code {e.code}")
    except KeyboardInterrupt:
        print("\n[INFO] Dibatalkan (Ctrl+C). Kembali ke Main Menu.")
    except Exception:
        print("[ERROR] Terjadi kesalahan di mode ini (program menu tetap berjalan):")
        traceback.print_exc()
    finally:
        cleanup_windows()
    return None


def status_lines(cfg: vl.Config) -> str:
    store = FieldStore.load(cfg.field_file)
    n = len(store.fields)
    return (f"Camera  : index {cfg.index}, {cfg.width}x{cfg.height} @ {cfg.fps} FPS, source {cfg.source}\n"
            f"Settings: {SETTINGS_FILE} {'ADA' if os.path.exists(SETTINGS_FILE) else 'belum ada'}\n"
            f"Calib   : {CALIBRATION_BASE}.npz {'ADA' if os.path.exists(CALIBRATION_BASE + '.npz') else 'belum ada (PIXEL TEST MODE)'}\n"
            f"Fields  : {cfg.field_file} -> {n} field" +
            ("" if store.source != 'legacy field.json' else " (dari field.json lama)"))


# ----------------------------------------------------------------------------------------
# MENU 1 - Camera Settings
# ----------------------------------------------------------------------------------------
def mode_camera_settings(cfg: vl.Config):
    cam = vl.open_camera_from_config(cfg)
    try:
        run_camera_settings(cam)
    finally:
        cam.close()


# ----------------------------------------------------------------------------------------
# MENU 2 - Camera Calibration  (memanggil camera_calibration.main; tidak ditulis ulang)
# ----------------------------------------------------------------------------------------
def mode_calibration(cfg: vl.Config):
    if cfg.source == "realsense":
        print("[INFO] Camera Calibration saat ini untuk webcam (C270). Jalankan dengan --source webcam.")
        return
    cols, rows, size = DEFAULT_COLS, DEFAULT_ROWS, DEFAULT_SQUARE_MM
    print(f"\nCheckerboard: {cols + 1}x{rows + 1} kotak -> {cols}x{rows} INNER corner, kotak {size:g} mm")
    raw = ask("Tekan Enter untuk memakai nilai ini, atau ketik 'cols rows mm' (mis. 7 7 34): ")
    if raw:
        try:
            c, r, s = raw.replace(",", " ").split()
            cols, rows, size = int(c), int(r), float(s)
        except ValueError:
            print("[WARNING] Format tidak dikenali -> memakai nilai default.")
    if os.path.exists(CALIBRATION_BASE + ".npz"):
        if ask(f"{CALIBRATION_BASE}.npz sudah ada dan akan DITIMPA bila Anda menekan C. Lanjut? [y/N]: ").lower() != "y":
            return
    argv: List[str] = ["--camera-index", str(cfg.index), "--cols", str(cols), "--rows", str(rows),
                       "--square-size", f"{size:g}", "--width", str(cfg.width), "--height", str(cfg.height),
                       "--fps", str(cfg.fps), "--output", CALIBRATION_BASE]
    if cfg.no_threaded:
        argv.append("--no-threaded")
    camera_calibration.main(argv)


# ----------------------------------------------------------------------------------------
# MENU 3 - Select / Manage Field  (Case 1)
# ----------------------------------------------------------------------------------------
def mode_field(cfg: vl.Config):
    calib = load_calibration_if_available()
    store = FieldStore.load(cfg.field_file)
    cam = vl.open_camera_from_config(cfg)
    try:
        run_field_manager(cam, store, vl.ColorClassifier(), calib, cfg.display_scale, cfg.field_file)
    finally:
        cam.close()


# ----------------------------------------------------------------------------------------
# MENU 4 - Case 2 Object Detection Test
# ----------------------------------------------------------------------------------------
def choose_fields(store: FieldStore):
    """Menu terminal pemilihan field. Return list Field, atau None bila kembali / tidak ada."""
    fields = store.active_fields()
    if not fields:
        print("\n[WARNING] No field selected.")
        print("[1] Go to Select / Manage Field")
        print("[2] Back to Main Menu")
        return "go_field" if ask("Choose: ") == "1" else None
    print("\nSelect Active Field:\n")
    for f in fields:
        print(f"[{f.id}] FIELD {f.id}")
    print("[A] ALL FIELDS")
    print("[B] Back")
    ans = ask("Choose: ").lower()
    if ans in ("b", "", "0"):
        return None
    if ans == "a":
        return fields
    try:
        ids = {int(x) for x in ans.replace(",", " ").split()}
    except ValueError:
        print("[WARNING] Pilihan tidak dikenali.")
        return []
    chosen = [f for f in fields if f.id in ids]
    if not chosen:
        print("[WARNING] Field tidak ditemukan.")
    return chosen


def mode_object(cfg: vl.Config):
    while True:
        store = FieldStore.load(cfg.field_file)
        if store.load_error:
            print("[WARNING]", store.load_error)
        picked = choose_fields(store)
        if picked is None:
            return
        if picked == "go_field":
            run_mode(mode_field, cfg)
            continue
        if not picked:
            continue
        calib = load_calibration_if_available()                  # None -> PIXEL TEST MODE (tetap berjalan)
        print(f"[INFO] Coordinate mode: PIXEL  |  fields: {', '.join('FIELD ' + str(f.id) for f in picked)}")
        cam = vl.open_camera_from_config(cfg)
        try:
            res = vl.run_case2(cam, cfg, vl.Visualizer(cfg.display_scale), vl.ColorClassifier(),
                               picked, calib, store.image_space)
        finally:
            cam.close()
        cleanup_windows()
        if res == "fields":                                      # tombol F di Case 2
            run_mode(mode_field, cfg)
            continue
        return


# ----------------------------------------------------------------------------------------
# MENU 5 - Camera Test
# ----------------------------------------------------------------------------------------
def mode_camera_test(cfg: vl.Config):
    cam = vl.open_camera_from_config(cfg)
    try:
        camera_calibration.run_camera_test(cam)
    finally:
        cam.close()


# ----------------------------------------------------------------------------------------
# MENU 6 - YOLO Object Detection  (modul terpisah; tidak mengubah mode 1-5)
# ----------------------------------------------------------------------------------------
def mode_yolo(cfg: vl.Config):
    """Diimpor malas: menu lama tetap jalan walau ultralytics/torch belum terpasang.
    Kamera dibuka & dilepas oleh run_yolo_menu() lewat fungsi pembuka kamera project yang sama."""
    try:
        import yolo_detector
    except ImportError as e:
        print(f"[ERROR] Modul YOLO tidak bisa dimuat: {e}")
        return
    yolo_detector.run_yolo_menu(cfg, lambda: vl.open_camera_from_config(cfg))


# ----------------------------------------------------------------------------------------
# MENU 7 - Ambil gambar dataset dari kamera (tidak membutuhkan checkpoint YOLO)
# ----------------------------------------------------------------------------------------
def mode_capture_dataset(cfg: vl.Config):
    try:
        import camera_capture
        camera_capture.run_capture_menu(lambda: vl.open_camera_from_config(cfg))
    except ImportError as e:
        print(f"[ERROR] Modul pengambilan gambar tidak dapat dimuat: {e}")


# ----------------------------------------------------------------------------------------
# MENU 8 - Training / transfer learning YOLO
# ----------------------------------------------------------------------------------------
def mode_yolo_training(cfg: vl.Config):
    import train_yolo
    print("\nPilih eksperimen training YOLO:")
    print("[1] E1 - Transfer learning (pretrained .pt)")
    print("[2] E2 - Training from scratch (arsitektur .yaml)")
    print("[3] E3 - Transfer learning + satu perubahan hyperparameter")
    print("[4] Semua eksperimen berurutan (E1, E2, E3)")
    print("[0] Kembali")
    choice = ask("Choose: ")
    exp = {"1": "e1", "2": "e2", "3": "e3", "4": "all"}.get(choice)
    if exp is None:
        return
    print("\n[INFO] Training menggunakan dataset yang sudah dianotasi dan dipersiapkan.")
    if ask("Izinkan Ultralytics mengunduh pretrained weights jika belum ada? [y/N]: ").lower() == "y":
        args = ["--exp", exp, "--allow-download"]
    else:
        args = ["--exp", exp]
    result = train_yolo.main(args)
    if result == 0:
        print("[OK] Training selesai. Periksa runs/<nama_eksperimen>/weights/ untuk checkpoint yang dihasilkan.")
    else:
        print(f"[ERROR] Training belum selesai dengan sukses (exit code {result}). Periksa pesan error di atas.")


# ----------------------------------------------------------------------------------------
# MENU 9 - Workflow dataset dinamis (kelas, anotasi, auto-labeling + human review)
# ----------------------------------------------------------------------------------------
def mode_dataset_workflow(cfg: vl.Config):
    try:
        import dataset_workflow
        dataset_workflow.menu()
    except ImportError as e:
        print(f"[ERROR] Modul dataset workflow tidak dapat dimuat: {e}")


# ----------------------------------------------------------------------------------------
# MENU 10 - Evaluasi validation (boleh diulang; tidak menyentuh test set)
# ----------------------------------------------------------------------------------------
def mode_evaluate_val(cfg: vl.Config):
    import evaluate_yolo
    print("\n[INFO] Evaluasi VAL boleh diulang. Split TEST tidak disentuh oleh menu ini.")
    if ask("Jalankan evaluasi VAL E1/E2/E3 dan perbarui grafik perbandingan? [y/N]: ").lower() != "y":
        return
    result = evaluate_yolo.main([])
    print("[OK] Evaluasi VAL selesai." if result == 0 else f"[ERROR] Evaluasi VAL gagal (kode {result}).")


# ----------------------------------------------------------------------------------------
# MENU 11 - Final test dengan pengaman: tidak pernah force retest dari menu
# ----------------------------------------------------------------------------------------
def mode_final_test(cfg: vl.Config):
    from pathlib import Path
    import json
    import yolo_config as yc
    result_file = yc.RUNS_DIR / "final_test.json"
    if result_file.exists():
        print(f"[AMAN] Hasil final test sudah ada: {result_file}")
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            print(json.dumps(data, ensure_ascii=False, indent=2)[:6000])
        except Exception as e:
            print(f"[WARNING] File ada tetapi tidak bisa dibaca sebagai JSON: {e}")
        print("[DIBATALKAN] Test set tidak dijalankan ulang. Menu ini tidak menyediakan force retest.")
        return
    import evaluate_yolo
    print("[PERINGATAN] Final test akan memakai split TEST dan seharusnya hanya dilakukan sekali, setelah model dipilih berdasarkan VAL.")
    if ask("Konfirmasi bahwa model sudah dipilih dari hasil VAL dan test belum pernah dilakukan? ketik TEST: ") != "TEST":
        print("Dibatalkan.")
        return
    result = evaluate_yolo.main(["--final-test"])
    print("[OK] Final test selesai." if result == 0 else f"[ERROR] Final test gagal (kode {result}).")


# ----------------------------------------------------------------------------------------
# MENU 12 - Laporan ringkas berdasarkan artefak yang ada; tidak menguji model
# ----------------------------------------------------------------------------------------
def mode_reports(cfg: vl.Config):
    import csv, json
    from pathlib import Path
    import yolo_config as yc
    runs = yc.RUNS_DIR
    runs.mkdir(parents=True, exist_ok=True)
    out_txt = runs / "hasil_praktik_4222411028.txt"
    out_csv = runs / "hasil_praktik_4222411028.csv"
    lines = ["RET503 - LAPORAN HASIL PRAKTIK YOLO", "Nama: Yonatan Andreas", "NIM: 4222411028", "",
             "Catatan: laporan ini merangkum artefak yang ditemukan; tidak menjalankan training atau test.", ""]
    rows = []
    for key, folder in (("E1", "e1_pretrained"), ("E2", "e2_scratch"), ("E3", "e3_finetuning")):
        d = runs / folder
        meta_path = d / "experiment_meta.json"
        results = d / "results.csv"
        best = d / "weights" / "best.pt"
        status = "TERSEDIA" if best.exists() else "BELUM TERSEDIA"
        metric_note = "Metrik VAL tidak diekstrak otomatis dari results.csv; jalankan menu evaluasi VAL untuk perbandingan terkini."
        if results.exists(): metric_note = f"results.csv tersedia ({results.stat().st_size} bytes); lihat grafik/CSV asli."
        meta = {}
        if meta_path.exists():
            try: meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception: pass
        lines += [f"{key} ({folder}): {status}", f"  best.pt: {'ADA' if best.exists() else 'TIDAK ADA'}", f"  results.png: {'ADA' if (d/'results.png').exists() else 'TIDAK ADA'}", f"  results.csv: {'ADA' if results.exists() else 'TIDAK ADA'}", f"  Catatan: {metric_note}", ""]
        rows.append({"experiment": key, "run_folder": folder, "status": status, "best_pt": best.exists(), "results_png": (d/"results.png").exists(), "results_csv": results.exists(), "train_meta_present": bool(meta)})
    test_path = runs / "final_test.json"
    lines += [f"Final test tersimpan: {'YA' if test_path.exists() else 'TIDAK'}", "Test set tidak dijalankan oleh pembuat laporan."]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] Laporan dibuat:\n - {out_txt}\n - {out_csv}")
    print("[INFO] Ini laporan inventaris artefak, bukan pengganti hasil metrik evaluasi.")


ACTIONS = {"1": mode_camera_settings, "2": mode_calibration, "3": mode_field,
           "4": mode_object, "5": mode_camera_test, "6": mode_yolo,
           "7": mode_capture_dataset, "8": mode_yolo_training, "9": mode_dataset_workflow, "10": mode_evaluate_val,
           "11": mode_final_test, "12": mode_reports}


def main(argv=None):
    ns = parse_args(argv)
    cfg = vl.config_from_namespace(ns)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))         # semua file data di folder project
    print(f"[INFO] OpenCV {cv2.__version__} | folder kerja: {os.getcwd()}")
    while True:
        print(BANNER + MENU)
        print(status_lines(cfg) + "\n")
        choice = ask("Choose: ")
        if choice == "0":
            print("Sampai jumpa.")
            return
        action = ACTIONS.get(choice)
        if action is None:
            print("[WARNING] Pilihan tidak valid.")
            continue
        run_mode(action, cfg)
        print("\n[INFO] Kembali ke Main Menu.")


if __name__ == "__main__":
    main()
