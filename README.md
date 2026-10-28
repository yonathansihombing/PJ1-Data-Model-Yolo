# YOLO Object Detection — Dataset, Training, dan Evaluasi

## 1. Ruang lingkup

Repositori ini menggabungkan modul kamera, pengelolaan anotasi, pipeline object detection YOLO, eksperimen training, dan evaluasi. Pipeline deteksi utama menggunakan konfigurasi eksperimen YOLO26n. Modul segmentation merupakan jalur terpisah dan tidak boleh dianggap sebagai hasil eksperimen object detection.

Dokumentasi ini menjelaskan alur berdasarkan berkas yang ada di repositori. Hasil training yang sudah tersimpan tidak otomatis membuktikan bahwa eksperimen dapat diulang pada komputer lain; dependensi, dataset, checkpoint, dan perangkat keras tetap perlu diperiksa.

## 2. Kelas dan format anotasi

Konfigurasi dataset saat ini menetapkan kelas berikut:

| ID | Nama kelas |
|---:|---|
| 0 | `field` |
| 1 | `box_hitam` |
| 2 | `box_putih` |

Sumber nama kelas untuk konversi polygon adalah `datasets/raw_yolo/classes.txt`. Skrip `prepare_polygon_detection.py` akan membangun ulang `configs/data.yaml` dari daftar tersebut ketika proses persiapan berhasil. Jangan mengubah urutan kelas setelah anotasi dibuat tanpa memeriksa dan memperbarui seluruh label.

Repositori menggunakan dua format anotasi yang berbeda:

- **Segmentation polygon:** file di `datasets/raw_yolo/labels_seg/`; berisi ID kelas dan titik polygon yang dinormalisasi.
- **Object detection:** file YOLO berisi `class_id x_center y_center width height`; disimpan pada `datasets/object_detection/labels/{train,val,test}/`.

Konversi polygon menjadi object detection menghasilkan kotak pembatas sumbu-sejajar (*axis-aligned bounding box*, AABB). Bentuk polygon asli tidak disimpan sebagai bentuk kotak yang diputar. Skrip konversi tidak mengubah label polygon sumber.

## 3. Struktur utama

- `configs/`: konfigurasi dataset, training, dan inference.
- `datasets/raw_yolo/`: gambar sumber, `classes.txt`, dan anotasi polygon.
- `datasets/object_detection/`: dataset deteksi yang sudah dibagi menjadi train, validation, dan test.
- `datasets/reports/`: laporan pemeriksaan dataset jika dibuat oleh pipeline.
- `runs/`: hasil training dan evaluasi, termasuk metrik serta checkpoint.
- `segment/`: hasil training segmentation yang tersimpan.
- `main.py`: menu utama aplikasi kamera.
- `dataset_workflow.py`: pengelolaan kelas, anotasi, dan review label.
- `prepare_polygon_detection.py`: konversi polygon ke kotak deteksi dan pembagian dataset.
- `prepare_dataset.py`: utilitas dataset, termasuk pemeriksaan, split, dan konversi COCO.
- `train_yolo.py` dan `evaluate_yolo.py`: eksperimen object detection dan evaluasi.
- `train_segmentation.py`: training segmentation dengan konfigurasi dataset segmentation yang terpisah.

Sebagian direktori berisi artefak hasil yang dihasilkan saat program dijalankan. Isi direktori tersebut dapat berbeda pada checkout baru.

## 4. Persiapan lingkungan Windows

Jalankan perintah dari direktori repositori. Pastikan Python yang dipakai adalah versi yang didukung oleh versi PyTorch dan Ultralytics yang dipilih.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Jika PowerShell menolak aktivasi skrip, gunakan Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Instalasi PyTorch untuk CPU atau GPU

Pasang PyTorch **sebelum** memasang seluruh dependensi proyek. Pilih perintah sesuai sistem operasi, versi Python, dan perangkat dari pemilih resmi: https://pytorch.org/get-started/locally/

Untuk GPU NVIDIA, gunakan paket PyTorch dengan runtime CUDA yang didukung oleh instalasi PyTorch tersebut. Toolkit CUDA yang terpasang secara terpisah tidak otomatis menjamin PyTorch dapat menggunakan GPU.

Verifikasi instalasi:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA tersedia:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'tidak tersedia')"
```

Jika `CUDA tersedia: False`, training masih dapat berjalan di CPU, tetapi biasanya lebih lambat. Periksa kompatibilitas driver NVIDIA, paket PyTorch, dan versi Python sebelum mencoba memasang ulang.

### Instal dependensi proyek

```powershell
python -m pip install -r requirements.txt
```

Verifikasi paket utama:

```powershell
python -c "import cv2, yaml, ultralytics; print('OpenCV:', cv2.__version__); print('Ultralytics:', ultralytics.__version__)"
```

File `requirements.txt` tidak mengunci versi paket. Karena itu, hasil instalasi dapat berbeda antarwaktu dan komputer. Untuk laporan yang dapat direproduksi, catat versi Python, PyTorch, CUDA yang dilaporkan PyTorch, OpenCV, dan Ultralytics.

## 5. Periksa dataset sumber

Pastikan folder dan berkas berikut tersedia:

- `datasets/raw_yolo/images/`
- `datasets/raw_yolo/labels_seg/`
- `datasets/raw_yolo/classes.txt`

Tinjau gambar dan label sebelum membuat split. Setiap anotasi harus cocok dengan objek pada gambar. Label usulan dari auto-labeling harus direview sebelum diperlakukan sebagai label final. Hindari gambar yang sama atau hampir sama tersebar di split yang berbeda.

## 6. Konversi polygon dan split dataset deteksi

Sebelum menjalankan konversi, pastikan dataset sumber sudah dicadangkan. Jalankan:

```powershell
python prepare_polygon_detection.py
```

Jika nama gambar berupa pola seperti `scene01_frame0001.jpg`, `scene01_frame0002.jpg`, gunakan grup berdasarkan nama adegan:

```powershell
python prepare_polygon_detection.py --group-regex '^(.*)_frame\\d+

Skrip mengonversi polygon menjadi kotak AABB, lalu membagi dataset ke train/validation/test dengan seed bawaan 42 dan memperbarui `configs/data.yaml`. Gambar identik (hash file sama) otomatis ditempatkan dalam split yang sama. Jika nama file memuat identitas adegan/video, gunakan `--group-regex` agar frame dari adegan yang sama tidak tersebar antar-split. Hasil ditulis ke `datasets/object_detection/`, termasuk `split_manifest.csv` dan `split_summary.json`.

Periksa `split_manifest.csv` untuk melihat penempatan gambar dan grup pada setiap split. Pengelompokan dapat membuat rasio aktual sedikit berbeda dari rasio target, karena satu grup tidak boleh dipecah antar-split. Jika keluaran lama sudah ada, jangan menimpanya sebelum memastikan hasil tersebut tidak diperlukan. Opsi `--overwrite` mengizinkan pembuatan ulang keluaran:

```powershell
python prepare_polygon_detection.py --overwrite
```

Opsi tersebut dapat mengganti hasil dataset deteksi sebelumnya. Gunakan hanya setelah membuat cadangan jika data lama perlu dipertahankan.

## 7. Eksperimen object detection

Konfigurasi utama:

| Eksperimen | Model awal | Tujuan |
|---|---|---|
| E1 — `e1_pretrained` | `yolo26n.pt` | Baseline dengan bobot pretrained |
| E2 — `e2_scratch` | `yolo26n.yaml` | Inisialisasi dari arsitektur tanpa bobot pretrained |
| E3 — `e3_finetuning` | Mewarisi E1 | Mengubah hanya `train.lr0` dari 0.01 menjadi 0.005 |

Parameter training E1 dan E2 didefinisikan pada berkas YAML masing-masing. E3 mewarisi E1 sehingga perubahan yang dimaksud hanya learning rate awal. Periksa konfigurasi efektif sebelum membandingkan hasil.

Periksa konfigurasi dan kesiapan dataset tanpa training:

```powershell
python train_yolo.py --exp e1 --dry-run
python train_yolo.py --exp e2 --dry-run
python train_yolo.py --exp e3 --dry-run
```

Jalankan eksperimen satu per satu:

```powershell
python train_yolo.py --exp e1 --allow-download
python train_yolo.py --exp e2
python train_yolo.py --exp e3
```

Opsi `--allow-download` mengizinkan pengunduhan bobot pretrained bila diperlukan. Periksa bantuan skrip dengan `python train_yolo.py --help` jika opsi atau konfigurasi berubah. Skrip menolak penggunaan ulang nama hasil yang sudah ada kecuali opsi penimpaan yang disediakan digunakan. Training menggunakan train dan validation; test tidak dipakai untuk memilih bobot selama proses training.

Hasil disimpan di `runs/<nama_eksperimen>/`, termasuk `weights/best.pt`, `weights/last.pt`, `results.csv`, grafik, dan metadata bila proses selesai.

## 8. Evaluasi model

Jalankan evaluasi validation untuk membandingkan eksperimen:

```powershell
python evaluate_yolo.py
```

Gunakan hasil validation untuk memilih model. Jalankan final test setelah model final dipilih, sesuai opsi yang didukung skrip:

```powershell
python evaluate_yolo.py --final-test
```

Skrip menolak pengujian final berulang kecuali `--force-retest` digunakan. Jangan menggunakan hasil test berulang untuk memilih model karena hal itu mengubah fungsi test set sebagai evaluasi akhir. Periksa `python evaluate_yolo.py --help` untuk parameter pemilihan model yang tersedia pada versi kode saat ini.

Metrik yang dilaporkan meliputi precision, recall, mAP50, mAP50–95, waktu inferensi, serta hasil per kelas jika artefaknya tersedia. Jangan mengisi metrik yang belum dihasilkan program.

## 9. Inferensi melalui aplikasi kamera

Jalankan menu utama:

```powershell
python main.py
```

Menu yang berkaitan dengan YOLO pada kode saat ini meliputi:

- **[6]** YOLO Object Detection.
- **[7]** Capture gambar dataset dari kamera.
- **[8]** Training / Transfer Learning YOLO.
- **[9]** Pengelolaan dataset, kelas, anotasi, dan auto-labeling.
- **[10]** Evaluasi validation dan perbandingan E1/E2/E3.
- **[11]** Final test.
- **[12]** Pembuatan laporan TXT/CSV/README dari hasil yang tersedia.

Ikuti petunjuk di terminal untuk setiap menu. Pastikan checkpoint yang dipilih memang hasil training untuk kelas yang dimaksud. Hasil YOLO berupa kotak dalam koordinat piksel; koordinat tersebut bukan ukuran dunia nyata atau milimeter.

Untuk inferensi gambar secara langsung menggunakan Ultralytics, contoh berikut dapat dipakai di luar menu aplikasi setelah checkpoint tersedia:

```python
from ultralytics import YOLO

model = YOLO("runs/e1_pretrained/weights/best.pt")
results = model.predict(source="gambar_uji.jpg", conf=0.25, save=True)
```

Ganti path checkpoint dan gambar sesuai berkas yang tersedia. Contoh ini tidak menggantikan pipeline eksperimen yang mencatat konfigurasi dan metadata.

## 10. Jalur segmentation

`train_segmentation.py` adalah jalur terpisah dari E1/E2/E3 object detection. Skrip ini secara default mencari `datasets/segmentation/data.yaml` dan menggunakan `yolo11n-seg.pt`. Pastikan konfigurasi dataset segmentation benar-benar tersedia sebelum menjalankannya:

```powershell
python train_segmentation.py --help
```

Jangan menganggap konfigurasi segmentation tersedia hanya karena folder hasil training segmentation ada di repositori. Periksa file konfigurasi sumber dan label polygon sebelum menjalankan ulang training.

## 11. Troubleshooting

- **PyTorch tidak menemukan GPU:** periksa `torch.cuda.is_available()`, driver NVIDIA, kompatibilitas paket PyTorch, dan versi Python.
- **Modul tidak ditemukan:** aktifkan virtual environment yang benar, lalu jalankan `python -m pip install -r requirements.txt`.
- **Dataset atau label tidak ditemukan:** periksa nama folder, `classes.txt`, pasangan gambar-label, dan path di `configs/data.yaml`.
- **Checkpoint tidak ditemukan:** periksa folder `runs/` dan pastikan training berhasil menyimpan `weights/best.pt`.
- **Kelas model tidak sesuai:** periksa nama kelas yang tersimpan dalam checkpoint dan samakan dengan konfigurasi dataset yang dipakai.
- **Hasil evaluasi berbeda:** catat versi dependensi, konfigurasi efektif, seed, isi dataset, dan checkpoint. Perbedaan lingkungan dapat memengaruhi hasil.

## 12. Batas interpretasi hasil

Metrik tinggi pada dataset kecil tidak menjamin model akan bekerja sama baiknya pada kondisi baru. Evaluasi perlu mempertimbangkan jumlah gambar, variasi sudut, pencahayaan, latar belakang, kualitas anotasi, distribusi kelas, dan potensi kemiripan antara gambar train dan test.

Laporan harus membedakan hasil eksperimen utama dari eksperimen tambahan. Angka dari log lama tidak boleh dianggap sebagai hasil eksekusi ulang. Jika data atau log tidak tersedia untuk diverifikasi, nyatakan keterbatasan tersebut secara eksplisit.

```

Regex di atas memakai grup pertama (`scene01`) sebagai identitas grup. Sesuaikan regex dengan pola nama file dataset Anda. Jika tidak diberikan, hanya gambar dengan isi file identik yang otomatis digabung sebagai satu grup.

Skrip mengonversi polygon menjadi kotak AABB, lalu membagi dataset ke train/validation/test dengan seed bawaan 42 dan memperbarui `configs/data.yaml`. Gambar identik (hash file sama) otomatis ditempatkan dalam split yang sama. Jika nama file memuat identitas adegan/video, gunakan `--group-regex` agar frame dari adegan yang sama tidak tersebar antar-split. Hasil ditulis ke `datasets/object_detection/`, termasuk `split_manifest.csv` dan `split_summary.json`.

Jika keluaran lama sudah ada, jangan menimpanya sebelum memastikan hasil tersebut tidak diperlukan. Opsi `--overwrite` mengizinkan pembuatan ulang keluaran:

```powershell
python prepare_polygon_detection.py --overwrite
```

Opsi tersebut dapat mengganti hasil dataset deteksi sebelumnya. Gunakan hanya setelah membuat cadangan jika data lama perlu dipertahankan.

## 7. Eksperimen object detection

Konfigurasi utama:

| Eksperimen | Model awal | Tujuan |
|---|---|---|
| E1 — `e1_pretrained` | `yolo26n.pt` | Baseline dengan bobot pretrained |
| E2 — `e2_scratch` | `yolo26n.yaml` | Inisialisasi dari arsitektur tanpa bobot pretrained |
| E3 — `e3_finetuning` | Mewarisi E1 | Mengubah hanya `train.lr0` dari 0.01 menjadi 0.005 |

Parameter training E1 dan E2 didefinisikan pada berkas YAML masing-masing. E3 mewarisi E1 sehingga perubahan yang dimaksud hanya learning rate awal. Periksa konfigurasi efektif sebelum membandingkan hasil.

Periksa konfigurasi dan kesiapan dataset tanpa training:

```powershell
python train_yolo.py --exp e1 --dry-run
python train_yolo.py --exp e2 --dry-run
python train_yolo.py --exp e3 --dry-run
```

Jalankan eksperimen satu per satu:

```powershell
python train_yolo.py --exp e1 --allow-download
python train_yolo.py --exp e2
python train_yolo.py --exp e3
```

Opsi `--allow-download` mengizinkan pengunduhan bobot pretrained bila diperlukan. Periksa bantuan skrip dengan `python train_yolo.py --help` jika opsi atau konfigurasi berubah. Skrip menolak penggunaan ulang nama hasil yang sudah ada kecuali opsi penimpaan yang disediakan digunakan. Training menggunakan train dan validation; test tidak dipakai untuk memilih bobot selama proses training.

Hasil disimpan di `runs/<nama_eksperimen>/`, termasuk `weights/best.pt`, `weights/last.pt`, `results.csv`, grafik, dan metadata bila proses selesai.

## 8. Evaluasi model

Jalankan evaluasi validation untuk membandingkan eksperimen:

```powershell
python evaluate_yolo.py
```

Gunakan hasil validation untuk memilih model. Jalankan final test setelah model final dipilih, sesuai opsi yang didukung skrip:

```powershell
python evaluate_yolo.py --final-test
```

Skrip menolak pengujian final berulang kecuali `--force-retest` digunakan. Jangan menggunakan hasil test berulang untuk memilih model karena hal itu mengubah fungsi test set sebagai evaluasi akhir. Periksa `python evaluate_yolo.py --help` untuk parameter pemilihan model yang tersedia pada versi kode saat ini.

Metrik yang dilaporkan meliputi precision, recall, mAP50, mAP50–95, waktu inferensi, serta hasil per kelas jika artefaknya tersedia. Jangan mengisi metrik yang belum dihasilkan program.

## 9. Inferensi melalui aplikasi kamera

Jalankan menu utama:

```powershell
python main.py
```

Menu yang berkaitan dengan YOLO pada kode saat ini meliputi:

- **[6]** YOLO Object Detection.
- **[7]** Capture gambar dataset dari kamera.
- **[8]** Training / Transfer Learning YOLO.
- **[9]** Pengelolaan dataset, kelas, anotasi, dan auto-labeling.
- **[10]** Evaluasi validation dan perbandingan E1/E2/E3.
- **[11]** Final test.
- **[12]** Pembuatan laporan TXT/CSV/README dari hasil yang tersedia.

Ikuti petunjuk di terminal untuk setiap menu. Pastikan checkpoint yang dipilih memang hasil training untuk kelas yang dimaksud. Hasil YOLO berupa kotak dalam koordinat piksel; koordinat tersebut bukan ukuran dunia nyata atau milimeter.

Untuk inferensi gambar secara langsung menggunakan Ultralytics, contoh berikut dapat dipakai di luar menu aplikasi setelah checkpoint tersedia:

```python
from ultralytics import YOLO

model = YOLO("runs/e1_pretrained/weights/best.pt")
results = model.predict(source="gambar_uji.jpg", conf=0.25, save=True)
```

Ganti path checkpoint dan gambar sesuai berkas yang tersedia. Contoh ini tidak menggantikan pipeline eksperimen yang mencatat konfigurasi dan metadata.

## 10. Jalur segmentation

`train_segmentation.py` adalah jalur terpisah dari E1/E2/E3 object detection. Skrip ini secara default mencari `datasets/segmentation/data.yaml` dan menggunakan `yolo11n-seg.pt`. Pastikan konfigurasi dataset segmentation benar-benar tersedia sebelum menjalankannya:

```powershell
python train_segmentation.py --help
```

Jangan menganggap konfigurasi segmentation tersedia hanya karena folder hasil training segmentation ada di repositori. Periksa file konfigurasi sumber dan label polygon sebelum menjalankan ulang training.

## 11. Troubleshooting

- **PyTorch tidak menemukan GPU:** periksa `torch.cuda.is_available()`, driver NVIDIA, kompatibilitas paket PyTorch, dan versi Python.
- **Modul tidak ditemukan:** aktifkan virtual environment yang benar, lalu jalankan `python -m pip install -r requirements.txt`.
- **Dataset atau label tidak ditemukan:** periksa nama folder, `classes.txt`, pasangan gambar-label, dan path di `configs/data.yaml`.
- **Checkpoint tidak ditemukan:** periksa folder `runs/` dan pastikan training berhasil menyimpan `weights/best.pt`.
- **Kelas model tidak sesuai:** periksa nama kelas yang tersimpan dalam checkpoint dan samakan dengan konfigurasi dataset yang dipakai.
- **Hasil evaluasi berbeda:** catat versi dependensi, konfigurasi efektif, seed, isi dataset, dan checkpoint. Perbedaan lingkungan dapat memengaruhi hasil.

## 12. Batas interpretasi hasil

Metrik tinggi pada dataset kecil tidak menjamin model akan bekerja sama baiknya pada kondisi baru. Evaluasi perlu mempertimbangkan jumlah gambar, variasi sudut, pencahayaan, latar belakang, kualitas anotasi, distribusi kelas, dan potensi kemiripan antara gambar train dan test.

Laporan harus membedakan hasil eksperimen utama dari eksperimen tambahan. Angka dari log lama tidak boleh dianggap sebagai hasil eksekusi ulang. Jika data atau log tidak tersedia untuk diverifikasi, nyatakan keterbatasan tersebut secara eksplisit.
