# Vision Pick & Place + YOLO (Tugas RET503)

Project kamera/computer vision (Python + OpenCV) yang sudah ada, ditambah modul **YOLO Object Detection**
untuk tugas RET503 (persiapan dataset, transfer learning, training dari nol, tuning, evaluasi, dan uji webcam).

> **Status yang jujur.** Seluruh kode dan pipeline sudah dibuat dan diuji dengan data/kamera **tiruan** (40 tes otomatis, lihat
> bagian J). **Training YOLO nyata belum dijalankan** di paket ini (butuh PyTorch + Ultralytics + dataset di komputer Anda),
> sehingga **belum ada angka hasil eksperimen** di sini. Semua angka hasil akan muncul dari komputer Anda sendiri.
> Webcam fisik (Logitech C270) **belum diuji** oleh pembuat paket ini.

---

## Daftar isi
A. Gambaran project · B. Persiapan komputer · C. Menjalankan project lama · D. Menyiapkan dataset ·
E. Menjalankan E1, E2, E3 · F. Evaluasi dan laporan · G. YOLO pada kamera · H. Troubleshooting ·
I. Checklist akhir · J. Pengujian, perubahan file, dan keterbatasan

---

## A. Gambaran project

### A.1 Tujuan
1. Mempertahankan sistem kamera/vision yang sudah ada (menu 1–5).
2. Menambahkan menu [6] YOLO Object Detection, [7] Camera Dataset Capture, dan [8] Training/Transfer Learning.
3. Menyediakan pipeline: capture gambar → kelas dinamis → anotasi/review → validasi/split dataset → E1/E2/E3 → evaluasi → inferensi webcam.

### A.2 Fitur lama yang tetap tersedia (tidak diubah)
| Menu | Fungsi |
|---|---|
| [1] Camera Settings | Pengaturan kamera (brightness, contrast, gamma, exposure, dll.), disimpan di `camera_settings.json` |
| [2] Camera Calibration | Kalibrasi kamera (checkerboard) → `camera_calibration.npz/.json` |
| [3] Select / Manage Field | Memilih dan mengelola lapangan → `fields.json` |
| [4] Case 2 – Object Detection Test | Deteksi objek **lama** (warna HSV + kontur + bentuk + field) |
| [5] Camera Test | Tes kamera |

### A.3 Fitur dataset dan YOLO yang ditambahkan
- Menu **[7] Capture Gambar Dataset (kamera)** (`camera_capture.py`): membuka kamera project, tombol `S` menyimpan JPG ke `datasets/raw_yolo/images/`, tombol `Q`/ESC keluar. Nama gambar unik dan gambar lama tidak ditimpa. Fitur ini tidak memerlukan checkpoint YOLO.
- Menu **[8] Training / Transfer Learning YOLO**: menjalankan E1, E2, E3 atau semuanya, serta meminta konfirmasi sebelum mengizinkan unduhan pretrained weights.
- Menu **[9] Kelola Dataset / Kelas / Anotasi / Auto-labeling** (`dataset_workflow.py`): menambah kelas tanpa menetapkan semua kelas di awal, anotasi manual dengan mouse, auto-label dari checkpoint yang cocok, review/edit label usulan, validasi, dan split dataset.
- **Tambahan terbaru di menu dataset [8]**: template-based auto-labeling tanpa checkpoint custom. Pilih folder, gambar contoh, kelas, lalu klik empat sudut objek; program mencari kecocokan multi-skala di gambar lain dan menulis usulan OBB/AABB ke folder `labels_pending_obb` / `labels_pending`. Semua hasil wajib direview; template matching tidak dijamin benar bila sudut pandang, skala, pencahayaan, atau latar berubah.
- Menu **[6] YOLO Object Detection** (`yolo_detector.py`): webcam / file video / file gambar, bounding box + nama kelas +
  confidence + FPS, ubah confidence saat berjalan, pilih checkpoint.
- `prepare_dataset.py`: ambil subset COCO, konversi COCO↔YOLO, pemeriksaan dataset, visualisasi, pembagian train/val/test.
- `train_yolo.py`: eksperimen **E1** (pretrained), **E2** (dari nol), **E3** (tuning satu hyperparameter).
- `evaluate_yolo.py`: evaluasi val, perbandingan E1/E2/E3, dan uji akhir di test (hanya sekali).
- `configs/`: `data.yaml`, `e1_pretrained.yaml`, `e2_scratch.yaml`, `e3_finetuning.yaml`, `inference.yaml`.
- `tests/`: tes otomatis dataset, eksperimen, runtime YOLO, capture gambar, dan regresi fitur lama.

### A.4 Arsitektur dan alur kerja
```
main.py  ── menu [1..5] ──► modul kamera/vision lama
   ├──── menu [6] ────────► yolo_detector.py ──► checkpoint .pt + webcam (ultralytics, diimpor malas)
   ├──── menu [7] ────────► camera_capture.py ──► datasets/raw_yolo/images/ (tanpa checkpoint)
   ├──── menu [8] ────────► train_yolo.py ──► runs/<eksperimen>/weights/
   └──── menu [9] ────────► dataset_workflow.py ──► kelas/anotasi/review
                                  │  kamera dibuka lewat fungsi kamera project yang sama
                                  └─ yolo_config.py (path, YAML, aturan eksperimen)

prepare_dataset.py ─► datasets/raw_yolo ─► datasets/object_detection ─► train_yolo.py ─► runs/e1|e2|e3_*
                                                                     └─► evaluate_yolo.py ─► runs/summary_val.csv, final_test.json
```
`ultralytics` tidak diimpor saat aplikasi mulai. Kamera biasa dan capture gambar tetap bisa digunakan tanpa checkpoint; menu inferensi YOLO membutuhkan model `.pt` yang valid.

### A.5 Perbedaan deteksi lama vs YOLO
| | Case 2 (lama) | YOLO (baru) |
|---|---|---|
| Cara kerja | Aturan buatan manusia: warna HSV → kontur → bentuk persegi/persegi panjang → stabil beberapa frame | Jaringan saraf yang **belajar dari gambar berlabel** |
| Mengenali | Warna dan bentuk dalam field yang Anda pilih | **Kelas objek** (mis. person, car, cup) di seluruh frame |
| Butuh dataset/training | Tidak | Ya (atau model pretrained) |
| Memakai field & kalibrasi | Ya (field, mode pixel/undistort) | **Tidak** – bekerja di seluruh frame |
| Keluaran | Object ID, warna, bentuk, pixel X,Y | Kotak (x1,y1,x2,y2), kelas, confidence |
| Koordinat | Pixel | **Pixel** (bukan mm / koordinat dunia) |

> Hasil YOLO adalah koordinat **pixel** pada gambar. Mengubahnya menjadi koordinat lapangan/dunia memerlukan homography
> dan kalibrasi yang valid; itu **belum diimplementasikan** dan tidak diklaim di project ini.

---

## B. Persiapan komputer

### B.1 Versi Python dan sistem operasi
- Bagian kamera lama dan seluruh tes otomatis diuji pada **Python 3.12.3 (Linux)**.
- Versi Python yang didukung YOLO ditentukan oleh **PyTorch** dan **Ultralytics**; periksa dokumentasi resmi keduanya untuk versi
  Python di komputer Anda. Saya tidak dapat memverifikasi kompatibilitas versi dari lingkungan pembuatan paket ini.
- Kode kamera memakai DirectShow di **Windows** (`cv2.CAP_DSHOW`) dan `cv2.VideoCapture(index)` biasa di sistem lain
  (Linux Ubuntu). Project asli dikembangkan di Windows dengan Logitech C270; jalur Linux **belum diuji dengan kamera fisik**.
- Training sebaiknya di komputer dengan GPU NVIDIA; CPU tetap bisa tetapi lambat (lihat B.5).

### B.2 Membuat virtual environment
Buka terminal di folder project (folder yang berisi `main.py`).

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Jika muncul error *running scripts is disabled*: `Set-ExecutionPolicy -Scope Process Bypass` lalu ulangi aktivasi.

**Windows CMD**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux Ubuntu**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Bila `venv` belum ada di Ubuntu: `sudo apt install python3-venv`.

### B.3 Memasang PyTorch dan dependensi YOLO
1. **Pasang PyTorch lebih dulu** dengan perintah yang dihasilkan selektor di <https://pytorch.org> (pilih OS, Pip, dan
   CPU atau versi CUDA yang sesuai komputer). Jangan menyalin perintah dari sumber lain; versinya berbeda-beda.
2. Pasang sisanya:
```bash
pip install -r requirements.txt
```
`requirements.txt` sengaja **tanpa penguncian versi** karena kombinasi versi belum diuji bersama; bila ada konflik, lihat bagian H.

Pemeriksaan:
```bash
python -c "import cv2, numpy, yaml, ultralytics, torch; print('OK', torch.__version__)"
```

### B.4 Memeriksa CUDA (NVIDIA GPU)
```bash
nvidia-smi
python -c "import torch; print(torch.__version__, 'CUDA tersedia:', torch.cuda.is_available())"
```
Jika `False`: PyTorch yang terpasang adalah versi CPU atau driver tidak cocok → pasang ulang PyTorch dengan opsi CUDA dari
pytorch.org. Anda juga bisa melihat status dari menu `[6]` → `[6] Status dependensi & petunjuk`.

### B.5 Training tanpa GPU (CPU)
Tambahkan `--device cpu`. Training berjalan tetapi **jauh lebih lambat**. Untuk mengukur kecepatan komputer Anda, jalankan uji
1 epoch (`--epochs 1 --name-suffix _smoke`, lihat E.1), lalu perkirakan waktu 30 epoch. Bila terlalu lama: kurangi
`epochs` di `configs/e1_pretrained.yaml` (E2 dan E3 mengikuti otomatis), atau kurangi jumlah gambar (`--count`).

---

## C. Menjalankan project lama

```bash
python main.py
```
Opsi kamera (dari kode lama): `--index` (default 1 = Logitech C270), `--width`, `--height`, `--fps`, `--source webcam|realsense`.
Contoh webcam laptop: `python main.py --index 0`.

Menu yang tampil:
```
[1] Camera Settings
[2] Camera Calibration
[3] Select / Manage Field
[4] Case 2 - Object Detection Test
[5] Camera Test
[6] YOLO Object Detection
[7] Capture Gambar Dataset (kamera)
[8] Training / Transfer Learning YOLO
[0] Exit
```
Menu [1]–[5] mempertahankan fungsi lama; [4] tetap memakai deteksi lama. Menu [7] dapat menyimpan gambar tanpa model YOLO. Di tiap mode, tombol keluar (mis. Q) kembali ke menu
dan kamera dilepas. Kegagalan satu mode tidak menutup menu utama. File `camera_calibration.*` dan `fields.json` Anda tidak
diubah oleh fitur YOLO.

---

## D.1 Workflow kelas dinamis, anotasi, dan auto-labeling

1. Jalankan `python main.py`, lalu pilih **[9]**.
2. Pilih **[1] Tambah kelas objek** dan masukkan satu nama kelas, misalnya `botol_merah`. ID kelas baru ditambahkan di akhir; ID kelas yang sudah ada tidak digeser. File registry berada di `datasets/raw_yolo/classes.txt`, dan `configs/data.yaml` diperbarui.
3. Ambil gambar melalui menu **[7]**. Semua foto baru masuk ke `datasets/raw_yolo/images/`; foto tidak langsung dianggap data berlabel.
4. Kembali ke menu **[9] → [2]** untuk anotasi manual. Drag mouse kiri membuat bounding box, tombol angka ``0`–`9` atau `[`/`]` memilih ID kelas, `S` menyimpan label, `N` melewati gambar, dan `Q` keluar. Klik kanan menghapus kotak terakhir. Label yang disimpan ada di `datasets/raw_yolo/labels/`.
5. Auto-labeling melalui **[9] → [3]** hanya menggunakan checkpoint `.pt` yang Anda pilih. Prediksi hanya diterima jika nama kelas model cocok persis (tidak membedakan huruf besar/kecil) dengan nama pada registry. Hasil sementara disimpan ke `datasets/raw_yolo/labels_pending/` dan **tidak dipakai untuk training** sampai diperiksa.
6. Pilih **[9] → [4]** untuk memeriksa/mengedit kotak dan kelas. Tekan `S` untuk menyetujui dan menyimpan label final. Jika model belum mengenali kelas baru, anotasi manual diperlukan; model pretrained COCO tidak otomatis mengetahui kelas kustom.
7. Pilih **[9] → [6]** untuk validasi. Setiap gambar tanpa objek juga harus memiliki file label `.txt` kosong; jangan biarkan gambar tanpa file label. Pilih **[9] → [7]** setelah setidaknya 10 gambar valid berlabel untuk membuat split 70/20/10.

**Penting:** jangan menambah kelas baru setelah split lalu langsung melanjutkan training dari dataset lama. Tambahkan kelas, lengkapi anotasi, lalu buat ulang dataset split; menu akan meminta konfirmasi sebelum mengganti split lama. Auto-labeling adalah usulan, bukan label ground-truth.

## D. Menyiapkan dataset

Semua perintah dijalankan dari folder project. Tidak ada yang mengunduh sesuatu tanpa Anda minta.

### D.0 Mengambil gambar langsung dari kamera
1. Jalankan `python main.py`.
2. Pilih `[7] Capture Gambar Dataset (kamera)`, lalu `[1] Buka kamera dan ambil gambar`.
3. Tekan `S` pada jendela preview untuk menyimpan satu gambar; tekan `Q` atau `ESC` untuk selesai.
4. Gambar tersimpan di `datasets/raw_yolo/images/` dengan nama unik `camera_000001.jpg`, dan tidak tertimpa saat sesi berikutnya.
5. **Anotasi wajib:** gambar capture tidak memiliki bounding box otomatis. Gunakan alat anotasi yang mendukung format YOLO, lalu simpan file label `.txt` pasangan di `datasets/raw_yolo/labels/` dengan nama dasar yang sama. Untuk gambar latar tanpa objek, file label kosong hanya boleh dibuat jika memang sudah diperiksa bahwa tidak ada objek target.
6. Setelah semua gambar diberi label dan `classes.txt`/`configs/data.yaml` mencerminkan kelas yang benar, jalankan pemeriksaan dataset dan split sesuai bagian D.

Menu `[8] Training / Transfer Learning YOLO` menjalankan eksperimen hanya jika dataset berlabel, konfigurasi valid, dan bobot pretrained yang dibutuhkan tersedia (atau Anda mengizinkan unduhan). Jika prasyarat belum tersedia, training akan berhenti dengan pesan error; gambar yang belum dianotasi tidak otomatis dianggap dataset siap.

### D.1 Struktur folder
```
datasets/
├── coco_download/                 (opsional; unduhan COCO, tidak masuk ZIP)
├── raw_yolo/                      hasil pengambilan/konversi, SEBELUM dibagi
│   ├── images/   labels/   classes.txt
├── object_detection/              hasil pembagian (dipakai training)
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── classes.txt  split_manifest.csv  split_summary.json
└── reports/                       laporan pemeriksaan + visualisasi kotak
configs/data.yaml                  path dataset + daftar kelas
```

### D.2 Format label YOLO
Satu file `.txt` per gambar (nama sama dengan gambar, di folder `labels/` yang sejajar `images/`). Tiap objek satu baris:
```
class_id x_center y_center width height
```
Nilai 0–1 (dinormalisasi terhadap lebar/tinggi gambar), `class_id` mulai **0**. Gambar tanpa objek memakai file `.txt`
**kosong** (gambar latar). Urutan kelas harus sama dengan `names` di `configs/data.yaml`.

### D.3 Kelas yang dipakai
`configs/data.yaml` berisi kelas awal **person, car, cup** (contoh dari materi). **Jika dosen menentukan kombinasi kelas lain
untuk kelompok Anda, gunakan itu**: ganti nama kelas pada perintah `--classes` di bawah. Nama kelas COCO yang valid
mis. `person`, `car`, `cup`, `bottle`, `chair`, ... (perintah akan menampilkan contoh bila nama salah).

### D.4 Sumber dataset (pilih salah satu)

**Opsi 1 – Subset COCO val2017 (mengikuti program dosen).** Perintah ini mengunduh COCO val2017 (gambar + anotasi; ukuran
total kira-kira 1 GB, perkiraan – cek sendiri) **hanya bila Anda menambahkan `--download`**:
```bash
python prepare_dataset.py coco-subset --classes person car cup --count 200 --background 0.1 --seed 0 --download
```
Tanpa internet/ingin manual: unduh dan ekstrak sendiri sehingga ada `datasets/coco_download/val2017/*.jpg` dan
`datasets/coco_download/annotations/instances_val2017.json`, lalu jalankan perintah yang sama **tanpa** `--download`.
URL sumber (berasal dari program dosen `00_ambil_subset_coco.py`, tidak dapat saya verifikasi dari sini) akan ditampilkan
oleh program jika file belum ada. Lisensi: periksa <https://cocodataset.org/#termsofuse> (setahu saya anotasi berlisensi
CC BY 4.0 sedangkan gambar berasal dari Flickr dengan lisensi bervariasi – pastikan sendiri sebelum memakai/mengumpulkan).
Catatan: objek COCO selain kelas pilihan **tidak** dianotasi pada subset ini.

**Opsi 2 – Anotasi COCO JSON milik Anda.**
```bash
python prepare_dataset.py coco-to-yolo --coco PATH/anotasi.json --images PATH/folder_gambar --classes person car cup
```
Ganti `PATH/...` dengan lokasi file Anda. Kategori dipetakan ke `class_id` sesuai urutan `--classes` (tanpa `--classes`: semua
kategori, urut ID COCO). Anotasi `iscrowd` dilewati (kecuali `--keep-crowd`), kotak tak valid dilaporkan, kotak keluar gambar
dipotong; ringkasan di `datasets/raw_yolo/conversion_report.json`. Konversi sebaliknya:
`python prepare_dataset.py yolo-to-coco --root datasets/raw_yolo --out datasets/raw_yolo/annotations_coco.json`.

**Opsi 3 – Dataset YOLO sendiri.** Letakkan gambar di `datasets/raw_yolo/images/`, label di `datasets/raw_yolo/labels/`,
dan daftar kelas (satu nama per baris) di `datasets/raw_yolo/classes.txt`.

**Dataset sudah punya pembagian resmi train/val/test?** Letakkan langsung di `datasets/object_detection/images|labels/{train,val,test}`,
**jangan** jalankan `split` (program menolak mengacak ulang), lalu isi `configs/data.yaml` secara manual (path dan `names`).

### D.5 Memeriksa dataset dan visualisasi
```bash
python prepare_dataset.py check --root datasets/raw_yolo --visualize 12
```
Yang diperiksa: gambar rusak, label hilang, label tanpa gambar, jumlah kolom salah, `class_id` di luar daftar, nilai di luar 0–1,
lebar/tinggi ≤ 0, kotak keluar batas gambar, kotak sangat kecil, kotak duplikat, **gambar identik** (duplikat; lintas split =
kebocoran data = ERROR). Keluaran: jumlah gambar, jumlah kotak per kelas (per split), rasio kelas terbanyak/tersedikit, dan
laporan di `datasets/reports/raw_yolo_check.json|.txt`; gambar contoh bergambar kotak di `datasets/reports/visual_raw_yolo/`.
Kode keluar 1 bila ada ERROR.

**Class imbalance:** lihat baris *Rasio kelas terbanyak / tersedikit* dan *Kelas yang jauh lebih sedikit*. Kelas dianggap jauh
lebih sedikit bila jumlah kotaknya ≤ 1/3 kelas terbanyak (ambang `IMBALANCE_RATIO` di `prepare_dataset.py`). Untuk laporan,
sebut kelas "jarang" **hanya** berdasarkan angka aktual dari keluaran ini.

### D.6 Membagi train / validation / test
```bash
python prepare_dataset.py split
```
Bawaan: rasio 70/20/10, `--seed 42` (hasil identik bila data dan seed sama), hasil di `datasets/object_detection/`,
`configs/data.yaml` diperbarui otomatis dari `classes.txt`. Dataset sumber diperiksa dulu; bila ada ERROR, pembagian ditolak
(kecuali `--force`).

**Mencegah kebocoran data:**
- Gambar yang identik byte-per-byte selalu dimasukkan ke split yang sama.
- Bila gambar berasal dari video/adegan yang sama (frame hampir kembar), beri nama berpola dan gunakan `--group-regex`,
  contoh nama `lab1_frame001.jpg` → `python prepare_dataset.py split --group-regex "^(.*)_frame\d+$"`. Semua frame satu
  adegan masuk satu split. (Program hanya mendeteksi duplikat identik; gambar *mirip tapi tidak identik* hanya bisa dicegah lewat pengelompokan ini.)
- `split_manifest.csv` mencatat setiap gambar dan split-nya.

Pemeriksaan akhir (harus 0 ERROR; perhatikan WARNING tentang kelas yang tidak muncul di val/test):
```bash
python prepare_dataset.py check --root datasets/object_detection --visualize 12
```
Dataset kecil dapat membuat suatu kelas tidak muncul di val/test; tambah data (`--count`) bila itu terjadi.

### D.7 Dataset belum lengkap / error umum
`train_yolo.py` dan `evaluate_yolo.py` memeriksa dataset lebih dulu dan menampilkan langkah perbaikan bila folder `train`/`val`
kosong atau ada gambar tanpa label. `python train_yolo.py --exp e1 --dry-run` memeriksa semuanya **tanpa** training.

### D.8 Augmentasi
Didefinisikan eksplisit di `configs/e1_pretrained.yaml` (bagian `train`) dan diwarisi E2/E3: `fliplr 0.5`, `flipud 0.0`,
`degrees 0.0`, `scale 0.5`, `hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4`, `mosaic 1.0`. Augmentasi geometrik dihitung ulang
bersama kotak oleh Ultralytics (label tidak rusak) dan hanya berlaku pada data **train**. Nilai ini umum dipakai
Ultralytics tetapi bisa berbeda antarversi; yang berlaku sebenarnya tercatat di `runs/<eksperimen>/args.yaml`.

---

## E. Menjalankan E1, E2, dan E3

### E.0 Prasyarat
1. Dataset siap (bagian D).
2. **Bobot pretrained** untuk E1 dan E3: letakkan file `.pt` di folder `weights/` dengan nama yang sama dengan `model` di
   `configs/e1_pretrained.yaml` (bawaan: `weights/yolo26n.pt`). Unduh dari sumber resmi Ultralytics (lihat
   <https://docs.ultralytics.com>), **atau** tambahkan `--allow-download` agar Ultralytics mengunduhnya (butuh internet;
   file tersimpan di folder project). Tanpa itu, program berhenti dengan pesan jelas — **tidak ada unduhan diam-diam**.
3. **Nama model.** Contoh dosen memakai `yolo26n`. Jika versi Ultralytics Anda tidak mengenalinya, ubah `model:` di
   `configs/e1_pretrained.yaml` (mis. `yolo26n.pt`) **dan** `configs/e2_scratch.yaml` (mis. `yolo26n.yaml`). E3 mengikuti E1.
   Nama model mana yang tersedia pada versi Anda tidak dapat saya verifikasi dari sini.
4. Periksa tanpa training: `python train_yolo.py --exp e1 --dry-run` → harus menampilkan
   `[DRY-RUN] Konfigurasi dan dataset valid. Training tidak dijalankan.` (dry-run E1/E3 juga memeriksa bobot pretrained ada;
   dry-run E2 tidak butuh bobot).
5. Seluruh nilai bersama (epoch 30, imgsz 416, batch 16, seed 0, augmentasi) ada di `configs/e1_pretrained.yaml`;
   E2 dan E3 mewarisinya (`extends`), sehingga **dataset, split, seed, epoch, batch, dan augmentasi sama** untuk ketiganya.
   Nilai awal ini adalah titik mulai yang realistis, bukan nilai optimal; sesuaikan dengan komputer Anda.

### E.1 Uji cepat (disarankan sebelum training penuh)
```bash
python train_yolo.py --exp e1 --epochs 1 --name-suffix _smoke
```
Hasil di `runs/e1_pretrained_smoke/` (tidak dibaca oleh evaluasi). Hapus foldernya setelah selesai.

### E.2 E1 – Transfer learning (baseline)
- **Tujuan:** memulai dari bobot pretrained COCO lalu dilatih pada dataset tugas.
- **Perintah:** `python train_yolo.py --exp e1`
- **Konfigurasi:** `configs/e1_pretrained.yaml` (`model: yolo26n.pt`, `pretrained: true`).
- **Hasil:** `runs/e1_pretrained/`.

### E.3 E2 – Dari awal (scratch)
- **Tujuan:** arsitektur sama, **bobot acak**, tanpa pretrained, sebagai pembanding.
- **Perintah:** `python train_yolo.py --exp e2`
- **Konfigurasi:** `configs/e2_scratch.yaml` (`model: yolo26n.yaml`, `pretrained: false`; selain itu sama dengan E1).
- **Pengaman:** program menolak model `.pt` untuk E2, dan membatalkan training bila ternyata ada bobot pretrained yang termuat.
- **Keterbatasan:** dari nol butuh data dan epoch jauh lebih banyak; dengan epoch yang sama dengan E1 hasilnya biasanya jauh lebih
  rendah, dan itu hasil yang wajar untuk dibahas. Waktu per epoch kira-kira sama dengan E1 (arsitektur sama).
  Ultralytics dapat mengunduh file kecil pendukung sendiri (mis. font untuk grafik, atau bobot uji AMP pada GPU); itu
  perilaku Ultralytics, bukan bobot inisialisasi model. Matikan uji AMP dengan menambahkan `amp: false` pada bagian `train` bila perlu.

### E.4 E3 – Fine-tuning / tuning satu hyperparameter
- **Tujuan:** mengukur pengaruh **satu** perubahan terhadap E1.
- **Perintah:** `python train_yolo.py --exp e3`
- **Parameter yang diubah:** `imgsz` — **416 (E1) → 640 (E3)**. Alasan: objek kecil/jauh lebih mudah terlihat pada input
  lebih besar. Semua nilai lain **sama** dengan E1 (model pretrained yang sama, seed, epoch, batch, augmentasi).
- **Penegakan aturan:** program membandingkan YAML E3 dengan E1 dan menolak bila perubahan **bukan tepat satu**
  hyperparameter. Untuk mengganti parameter yang diuji (mis. `batch`), buka `configs/e3_finetuning.yaml`, hapus `imgsz`, isi
  parameter baru, dan perbarui catatan di komentar file itu.
- Catatan: E3 memakai input lebih besar sehingga lebih lambat; ini bagian dari analisis.

### E.5 Semua sekaligus dan opsi
```bash
python train_yolo.py --exp all            # E1 -> E2 -> E3
python train_yolo.py --exp e1 --device cpu            # paksa CPU
python train_yolo.py --exp e1 --device 0              # GPU pertama
python train_yolo.py --exp e1 --batch 8               # bila memori kurang
python train_yolo.py --exp e1 --workers 0             # bila error multiprocessing di Windows
python train_yolo.py --exp e1 --overwrite             # timpa hasil lama
```
> Opsi `--epochs/--batch/--workers/--device` mengubah **hanya proses itu**. Bila dipakai, gunakan nilai **sama** untuk E1, E2,
> E3 agar perbandingan adil (program memperingatkan dan mencatatnya di `experiment_meta.json`). Cara paling aman: ubah di
> `configs/e1_pretrained.yaml`.
> Program tidak menimpa hasil yang sudah ada kecuali `--overwrite`.

### E.6 Lokasi hasil dan cara memastikan berhasil
Setiap eksperimen menyimpan hasil di `runs/<nama>/` (`e1_pretrained`, `e2_scratch`, `e3_finetuning`):

| File | Isi | Dibuat oleh |
|---|---|---|
| `weights/best.pt`, `weights/last.pt` | checkpoint terbaik / terakhir | Ultralytics |
| `results.csv`, `results.png` | kurva loss & metrik per epoch | Ultralytics |
| `confusion_matrix.png` dll. | confusion matrix, kurva PR/F1 | Ultralytics |
| `args.yaml` | argumen training yang benar-benar dipakai | Ultralytics |
| `experiment_meta.json` | waktu latih (menit), konfigurasi efektif, versi paket, sidik jari dataset | skrip ini |
| `experiment_config.yaml` | konfigurasi eksperimen setelah penggabungan | skrip ini |

Nama file Ultralytics mengikuti versi yang Anda pakai; periksa isi foldernya. **Berhasil bila:** terminal menampilkan
`[OK] <nama> selesai dalam X menit` dan `best.pt : ... (ADA)`, jumlah baris `results.csv` = epoch, dan loss menurun pada `results.png`.

### E.7 Masalah umum saat training
Lihat bagian H (kecuali sudah dijelaskan di atas: bobot tidak ada, dataset belum siap, memori, worker Windows).

---

## F. Evaluasi dan laporan

### F.1 Evaluasi pada validation set (boleh berulang)
```bash
python evaluate_yolo.py
```
Mengevaluasi `best.pt` E1, E2, E3 dengan prosedur yang sama pada split **val** (memakai `imgsz` training masing-masing).
Eksperimen yang belum dilatih ditandai **BELUM DIJALANKAN** (tidak ada angka karangan).

Keluaran:
- `runs/summary_val.csv` dan `runs/comparison.md` — tabel perbandingan: perubahan terhadap E1, imgsz, Precision, Recall,
  mAP@50, mAP@50-95, ms/gambar, menit training.
- `runs/comparison_val.png` — grafik batang (bila matplotlib tersedia).
- `runs/_evaluation/<nama>_val/` — confusion matrix, kurva PR/F1 (dari Ultralytics) dan `<nama>_val_per_class.csv`
  (hasil per kelas, bila versi Ultralytics menyediakannya).

### F.2 Pengertian metrik
- **Precision:** dari semua kotak yang diprediksi, berapa yang benar.
- **Recall:** dari semua objek nyata, berapa yang berhasil ditemukan.
- **mAP@50:** rata-rata AP semua kelas bila kotak dianggap benar jika IoU ≥ 0.5.
- **mAP@50–95:** rata-rata mAP pada IoU 0.50, 0.55, …, 0.95 (lebih ketat; metrik utama pemilihan model di sini).
- **Confusion matrix:** kelas sebenarnya vs prediksi (termasuk "background" untuk kotak yang terlewat/salah).

### F.3 Memilih model dan uji akhir di test set (SEKALI)
Setelah membandingkan di val dan memilih model:
```bash
python evaluate_yolo.py --final-test
```
- Model dipilih **otomatis** = mAP@50-95 val tertinggi (atau `--select E3` secara manual; dicatat sebagai manual).
- Hasil: `runs/final_test.json` dan `runs/final_test.csv`.
- Skrip **menolak** test kedua kali. `--force-retest` tersedia tetapi dicatat sebagai *pelanggaran prosedur* di file hasil.
- **Jangan** memakai angka test untuk memilih/menyetel model. Training (`train_yolo.py`) tidak menyentuh test set sama sekali.

### F.4 Menyusun ringkasan tugas
1. Tabel perbandingan: salin dari `runs/comparison.md` / `runs/summary_val.csv` (angka val).
2. Hasil akhir: `runs/final_test.json` (angka test untuk model terpilih saja).
3. Lampirkan: `results.png` tiap eksperimen, `confusion_matrix.png`, `*_per_class.csv`, `comparison_val.png`.
4. Bahas: (a) E1 vs E2 – efek transfer learning dan keterbatasan E2; (b) E1 vs E3 – efek **satu** hyperparameter (imgsz)
   pada akurasi dan kecepatan; (c) kelas yang lemah (lihat per kelas dan confusion matrix) dan kaitannya dengan class
   imbalance dari laporan dataset; (d) keterbatasan: ukuran dataset, satu seed, dll.
5. Cantumkan sumber dataset, jumlah gambar/objek per kelas (dari laporan `check`), seed, dan versi paket
   (`experiment_meta.json`).

---

## G. Menjalankan YOLO pada kamera

### G.1 Menu
```bash
python main.py            # lalu pilih [6] YOLO Object Detection
```
Sub-menu:
```
[1] Deteksi dari Webcam      (memakai kamera & camera_settings.json project)
[2] Deteksi dari file video
[3] Deteksi dari file gambar
[4] Pilih checkpoint (.pt)
[5] Ubah confidence threshold
[6] Status dependensi & petunjuk
[0] Kembali ke Main Menu
```

### G.2 Memilih checkpoint
- Bawaan di `configs/inference.yaml`: `runs/e1_pretrained/weights/best.pt`. **Tidak ada pergantian diam-diam**: bila file itu tidak
  ada, menu menampilkan instruksi cara membuatnya dan daftar checkpoint yang **ada**, lalu Anda memilih sendiri lewat `[4]`.
- `[4]` menampilkan semua `runs/*/weights/best.pt|last.pt` dan `weights/*.pt` (bobot pretrained COCO juga boleh dipilih sebagai
  detektor umum 80 kelas), atau ketik path `.pt` sendiri. Pilihan berlaku untuk sesi itu saja.
- Memilih checkpoint dari folder run yang punya `experiment_meta.json` otomatis menyetel `imgsz` sesuai training (mis. 640 untuk E3);
  pesan ditampilkan. Ukuran input sebaiknya sama dengan saat training.
- Ubah nilai bawaan (confidence, imgsz, device) di `configs/inference.yaml`.

### G.3 Mengoperasikan jendela deteksi
| Tombol | Fungsi |
|---|---|
| `Q` / `ESC` | keluar, kembali ke menu YOLO |
| `+` / `=` | confidence threshold naik 0.05 |
| `-` | confidence threshold turun 0.05 |
| `SPASI` | jeda / lanjut |
| `S` | simpan snapshot ke `outputs/yolo_snapshots/` |

Jendela menampilkan kotak, **nama kelas + confidence**, FPS, waktu inferensi (ms), threshold, dan ringkasan jumlah deteksi.
Kotak berwarna per kelas. Koordinat kotak adalah **pixel** pada frame kamera.

### G.4 Memahami bounding box dan confidence
Bounding box = persegi panjang yang melingkupi objek terdeteksi. Confidence (0–1) = keyakinan model terhadap kelas itu.
Threshold tinggi → lebih sedikit kotak tetapi lebih yakin (Recall turun); rendah → lebih banyak kotak termasuk salah
(Precision turun).

### G.5 Keluar dengan aman
`Q`/`ESC` pada jendela, atau `Ctrl+C` di terminal: jendela OpenCV ditutup, kamera **selalu dilepas**, dan program kembali ke menu
(tes otomatis memeriksa pelepasan kamera pada keluar normal, `Ctrl+C`, dan error). Hanya satu mode yang memegang kamera pada satu waktu.

### G.6 Catatan penting
- YOLO **tidak** memakai field maupun kalibrasi, dan **tidak** meng-undistort frame. Model dilatih pada gambar apa adanya, sehingga
  frame kamera juga dibiarkan apa adanya (prinsip: preprocessing saat deteksi harus sama dengan saat training).
- Pengaturan kamera di `camera_settings.json` (brightness/contrast/gamma, dll.) **ikut berlaku** pada frame yang dilihat YOLO.
  Pengaturan ekstrem dapat menurunkan akurasi; kembalikan ke normal bila hasil aneh.
- Di CPU, FPS rendah itu wajar; di GPU jauh lebih tinggi.

### G.7 Kamera / model tidak ditemukan
Lihat bagian H (kamera tidak terbuka, checkpoint tidak ditemukan).

---

## H. Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` / `'yaml'` | Venv belum aktif atau paket belum terpasang: aktifkan venv, `pip install -r requirements.txt` (PyTorch dulu, bagian B.3). Menu 1–5 tetap jalan. |
| `No module named 'torch'` | Pasang PyTorch dari pytorch.org (B.3). |
| Versi PyTorch / Ultralytics tidak cocok, error saat import/training | Pasang ulang PyTorch lalu `pip install -U ultralytics`; atau buat venv baru khusus YOLO. Jangan memperbarui OpenCV tanpa perlu. Cek kompatibilitas di dokumentasi resmi keduanya. |
| Nama model tidak dikenali (mis. `yolo26n`) | Versi Ultralytics Anda belum mengenalnya: ganti `model:` di `configs/e1_pretrained.yaml` dan `e2_scratch.yaml` ke model yang tersedia (mis. `yolo26n.pt` / `yolov8n.yaml`), atau perbarui Ultralytics. |
| `Bobot pretrained '...pt' tidak ditemukan` | Taruh file di `weights/` atau jalankan dengan `--allow-download`. |
| Webcam tidak terbuka / dipakai program lain | Tutup aplikasi lain (Zoom, Camera, menu lain yang masih membuka kamera), coba `--index 0`/`1`/`2`, cabut-pasang kamera. Di menu YOLO kamera hanya dibuka setelah model berhasil dimuat. |
| `Kamera tidak mengirim frame` | Kamera tersambung tetapi tidak mengeluarkan gambar: ganti index, port USB, atau tutup program lain. |
| `Checkpoint tidak ditemukan` | Latih dulu (E.2) atau pilih file lewat menu `[4]`; instruksi lengkap ditampilkan menu. |
| `data.yaml` salah / dataset belum siap | Jalankan `python train_yolo.py --exp e1 --dry-run`; ikuti pesannya. Pastikan `names` di `configs/data.yaml` sama dengan `classes.txt` dan `path` benar. |
| Label tidak valid | `python prepare_dataset.py check --root ...` menunjukkan file dan baris yang salah. |
| `CUDA tidak tersedia` | Lihat B.4; sementara pakai `--device cpu`. |
| `CUDA out of memory` / RAM tidak cukup | `--batch 8` (atau 4), kurangi `imgsz` di E1 (E3 harus tetap berbeda tepat satu parameter dari E1), `--workers 0`. |
| Training terlalu lambat | Pakai GPU; kurangi `epochs` di `configs/e1_pretrained.yaml`; uji 1 epoch dulu (E.1). |
| Error multiprocessing / `workers` di Windows | `--workers 0`. |
| Hasil tidak ada di `runs/<nama>/` | Program memperingatkan bila Ultralytics menyimpan di tempat lain; salin foldernya ke `runs/<nama>` (mis. `runs/detect/...`). |
| `Hasil ... sudah ada` | Tambahkan `--overwrite` atau `--name-suffix _v2`. |
| `Test set SUDAH dipakai` | Memang disengaja (F.3). Gunakan hasil `runs/final_test.json`. |
| `SystemExit`/warning versi Python | Gunakan versi Python yang didukung PyTorch/Ultralytics (B.1). |

---

## I. Checklist akhir

**Fitur lama**
- [ ] `python main.py` menampilkan menu [1]–[5] dan [6]
- [ ] [1] Camera Settings terbuka dan menyimpan/memuat pengaturan
- [ ] [2] Camera Calibration berjalan; `camera_calibration.npz/.json` tidak hilang
- [ ] [3] Select / Manage Field membaca dan menyimpan `fields.json`
- [ ] [4] Case 2 mendeteksi dengan logika lama
- [ ] [5] Camera Test terbuka; kamera dapat dibuka lagi setelah keluar dari mode apa pun

**Dataset**
- [ ] `coco-subset` / data sendiri → `datasets/raw_yolo` (kelas sesuai ketentuan kelompok)
- [ ] `check` 0 ERROR; visualisasi kotak sudah dilihat
- [ ] Distribusi kelas dicatat (kelas jarang disebut berdasarkan angka aktual)
- [ ] `split` selesai (seed dicatat); `check` pada `object_detection` 0 ERROR; tidak ada duplikat lintas split

**Eksperimen**
- [ ] Bobot pretrained ada di `weights/`; `--dry-run` valid
- [ ] E1 selesai (pretrained) · E2 selesai (scratch, tanpa pretrained) · E3 selesai (satu perubahan: imgsz)
- [ ] Tiap run punya `best.pt`, `results.csv`, `results.png`, confusion matrix, `args.yaml`, `experiment_meta.json`

**Evaluasi**
- [ ] `python evaluate_yolo.py` → `runs/summary_val.csv`, `comparison.md`
- [ ] Model dipilih dari val; `python evaluate_yolo.py --final-test` dijalankan **sekali**
- [ ] Ringkasan laporan (F.4) disusun dari angka nyata

**Webcam**
- [ ] Menu [6] → [1] menampilkan kotak, kelas, confidence, FPS
- [ ] Mengubah confidence (+/-) berfungsi; `Q` kembali ke menu; kamera dapat dibuka mode lain
- [ ] Checkpoint tidak ada → muncul instruksi (bukan crash)

**Tes otomatis**
- [ ] `python -m unittest discover -s tests -v` → semua OK

---

## J. Pengujian, perubahan file, dan keterbatasan

### J.1 Menjalankan tes otomatis
```bash
python -m unittest discover -s tests -v
```
Tes memakai data sintetis, `ultralytics` tiruan, dan kamera tiruan (tanpa GPU, internet, atau webcam). Mencakup:
konversi/pemeriksaan/pembagian dataset (termasuk reprodusibilitas dan anti-kebocoran), aturan E1/E2/E3 (E3 tepat satu perubahan,
E2 tanpa pretrained, tanpa unduhan diam-diam, tidak menimpa hasil), evaluasi (BELUM DIJALANKAN tidak diberi angka, test hanya sekali),
menu YOLO (kamera dilepas pada keluar normal/Ctrl+C/error, tidak dibuka bila model gagal dimuat, tidak ada dua kamera sekaligus),
dan regresi (file lama identik dengan ZIP asli, modul lama bisa diimpor, menu 1–5 utuh, error satu mode tidak menutup menu).
`tests/baseline_original_sha256.json` berisi hash file asli; tes "file lama tidak berubah" akan gagal jika Anda **sendiri**
mengedit file lama (abaikan/perbarui hash bila itu disengaja).

### J.2 File baru
`yolo_config.py`, `yolo_detector.py`, `prepare_dataset.py`, `train_yolo.py`, `evaluate_yolo.py`,
`configs/{data,e1_pretrained,e2_scratch,e3_finetuning,inference}.yaml`, `tests/` (5 file + baseline), `requirements.txt`,
`.gitignore`, `README.md`, serta folder kosong `datasets/object_detection/...`, `runs/`, `weights/`, `outputs/` (berisi `.gitkeep`).

### J.3 File lama yang diubah
Hanya **`main.py`**, dan hanya penambahan: satu baris pada docstring, satu baris menu `[6] YOLO Object Detection`, fungsi `mode_yolo`
(impor malas `yolo_detector`), dan satu entri `"6": mode_yolo` pada `ACTIONS`. Semua file lain (termasuk `camera_calibration.*`
dan `fields.json`) identik dengan ZIP asli.

### J.4 Keterbatasan yang masih ada
- Training/evaluasi **nyata** belum dijalankan di paket ini; tidak ada angka metrik. Perilaku API Ultralytics yang sebenarnya
  (nama atribut hasil, nama file keluaran, daftar model yang tersedia) bergantung versi yang Anda pasang dan belum dapat saya verifikasi.
- Webcam fisik belum diuji; jalur Linux untuk kamera belum diuji dengan perangkat.
- Pemisahan anti-kebocoran hanya mengenali gambar **identik** dan kelompok yang Anda definisikan lewat `--group-regex`.
- E3 hanya menguji satu parameter (`imgsz`) dengan satu seed; hasil adalah satu run, bukan rata-rata beberapa seed.
- Subset COCO: objek di luar kelas pilihan tidak dianotasi (dapat menjadi "false positive" semu saat evaluasi).
- YOLO tidak terhubung ke field/kalibrasi/homography; tidak ada koordinat mm. Integrasi PLC/Modbus/UR3 tidak dibuat.

## Update: anotasi AABB/OBB dan auto-label review

Workflow dataset di `dataset_workflow.py` kini mendukung:

- Membaca gambar secara rekursif, termasuk foto yang masih berada di subfolder `datasets/raw_yolo/images/field`, `box_hitam`, `box_putih`, dan folder lain.
- AABB (kotak horizontal/vertikal) dan OBB (kotak berputar) dipilih saat menjalankan anotasi, auto-labeling, atau review.
- Label AABB disimpan di `datasets/raw_yolo/labels/` dalam format YOLO detect.
- Label OBB disimpan terpisah di `datasets/raw_yolo/labels_obb/` dalam format YOLO OBB (kelas + empat titik sudut ternormalisasi).
- Usulan auto-label AABB disimpan di `labels_pending/`, sedangkan OBB di `labels_pending_obb/`. Usulan tidak menjadi label final sebelum direview dan disimpan dengan tombol `S`.
- Nama label untuk foto di subfolder memakai path relatif yang digabung dengan `__`, sehingga foto bernama sama dari subfolder berbeda tidak menimpa label satu sama lain.

### Tombol anotator

- Drag tombol kiri: buat kotak baru. Pada OBB, kotak dimulai sejajar sumbu.
- `0`–`9`: pilih kelas berdasarkan ID.
- `[` / `]`: kelas sebelumnya/berikutnya.
- OBB: `,` / `.` memutar kotak terpilih 5 derajat; `A` / `D` memutar 1 derajat.
- Klik kanan: hapus kotak terakhir.
- `S`: simpan; `N`: lewati; `Q` atau `Esc`: keluar.

### Catatan checkpoint pretrained

Auto-labeling menggunakan kelas yang tersedia pada checkpoint dan hanya menerima deteksi bila nama kelas checkpoint cocok dengan nama kelas dataset (tidak peka huruf besar/kecil). Checkpoint COCO umum biasanya tidak memiliki kelas custom seperti `box_hitam` atau `field`, sehingga bisa menghasilkan nol usulan untuk kelas tersebut. Periksa hasil auto-labeling sebelum menyetujuinya. OBB yang dibuat dari checkpoint deteksi AABB pada awalnya bersudut 0 derajat; putar dan perbaiki saat review jika perlu.

### Catatan training

Menu split/training lama masih merupakan workflow AABB/YOLO detect. Label OBB disimpan terpisah supaya tidak tercampur ke training detect biasa. Untuk melatih OBB, gunakan model Ultralytics khusus OBB (misalnya keluarga model `*-obb.pt`) dan siapkan konfigurasi split/data OBB tersendiri; jangan langsung memakai menu split AABB untuk label OBB.


## Tambahan revisi: memilih folder foto saat anotasi

Foto boleh dimasukkan ke `datasets/raw_yolo/images/` secara langsung atau dikelompokkan ke subfolder, misalnya:

```
datasets/raw_yolo/images/
├── field/
│   ├── foto_001.jpg
│   └── foto_002.jpg
├── box_hitam/
│   └── foto_003.jpg
└── box_putih/
    └── foto_004.jpg
```

Jalankan `python main.py` → menu `[9]` → `[2]`. Pilih AABB/OBB, lalu pilih folder foto yang akan dikerjakan. Nama subfolder akan dipakai sebagai kelas default; jika kelas belum ada, program menawarkan untuk menambahkannya. Kelas tetap dapat diganti di jendela anotasi memakai tombol angka atau `[`/`]`. Pilihan `[0]` memproses seluruh foto.

Menu `[3]` dan `[4]` juga memungkinkan memilih folder untuk auto-labeling dan review. Auto-label checkpoint pretrained hanya akan menerima kelas yang namanya cocok dengan kelas checkpoint. Jika checkpoint tidak memiliki kelas `field`, `box_hitam`, atau `box_putih`, ia tidak bisa otomatis mengenali kelas khusus tersebut; lakukan anotasi manual untuk beberapa gambar lalu latih model khusus.

Label AABB dan OBB disimpan di direktori terpisah (`labels/` dan `labels_obb/`). Menu split/training detect biasa yang sudah ada tetap menggunakan AABB. Label OBB tidak boleh dicampur dengan training YOLO detect biasa; training OBB perlu model/config OBB yang sesuai.

---

## RET503 — konfigurasi eksperimen dan integrasi Polygon ke Detection

Lihat `RET503_PIPELINE.md` untuk konfigurasi E1/E2/E3, split train/val/test, evaluasi, konversi YOLO↔COCO, dan bukti yang harus dikumpulkan. Untuk tugas object detection, gunakan `prepare_polygon_detection.py` agar label polygon dari anotasi kamera dikonversi menjadi label bounding box YOLO `class_id cx cy w h`; label segmentation asli tetap aman dan tidak ditimpa.
