# YOLO Object Detection — Dataset, Training, dan Evaluasi Model

Repository ini berisi dataset, konfigurasi, hasil eksperimen, dan bobot model YOLO untuk proyek deteksi objek. Repository disusun agar proses pengumpulan data, persiapan dataset, pelatihan model, dan evaluasi dapat didokumentasikan serta digunakan kembali.

## 1. Tujuan Proyek

Proyek ini bertujuan untuk:

- Menyiapkan dataset gambar beserta anotasi objek dalam format YOLO.
- Membagi dataset menjadi data training, validation, dan testing.
- Melakukan eksperimen pelatihan model YOLO dengan beberapa konfigurasi.
- Mengevaluasi performa model berdasarkan hasil pengujian.
- Menyimpan bobot model terbaik untuk digunakan pada proses inferensi atau deteksi objek.

## 2. Kelas Objek

Berdasarkan konfigurasi dataset yang digunakan, kelas objek terdiri atas:

| ID | Nama kelas |
|---:|---|
| 0 | `field` |
| 1 | `box_hitam` |
| 2 | `box_putih` |

ID kelas pada anotasi harus sesuai dengan urutan kelas pada `configs/data.yaml`.

**Catatan:** Setiap gambar harus memiliki anotasi yang sesuai dengan objek sebenarnya. Jika kelas atau anotasi diubah, dataset perlu divalidasi kembali sebelum training.

## 3. Struktur Repository

```text
PJ1-Data-Model-Yolo/
│
├── README.md
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   └── ... konfigurasi eksperimen
│
├── datasets/
│   ├── raw_yolo/
│   │   ├── classes.txt
│   │   ├── images/
│   │   └── labels_seg/
│   │
│   ├── object_detection/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── split_summary.json
│   │
│   └── reports/
│
└── runs/
    ├── e1_pretrained/
    ├── e2_scratch/
    ├── e3_finetuning/
    ├── e1_fixed_lr/
    ├── e3_fixed_lr/
    └── _evaluation/
```

Struktur di atas merupakan gambaran folder utama yang digunakan dalam proyek. File tambahan dapat tersedia sesuai hasil eksperimen dan pembaruan dataset.

## 4. Fungsi Setiap Folder

### `configs/`

Berisi konfigurasi dataset dan eksperimen training. File `data.yaml` digunakan untuk menentukan lokasi dataset, direktori training, validation, testing, serta nama kelas.

### `datasets/raw_yolo/`

Berisi data mentah yang menjadi sumber dataset. Folder ini digunakan untuk menyimpan gambar dan anotasi sebelum atau selama proses persiapan dataset.

Data mentah sebaiknya dipertahankan agar gambar dapat diperiksa, anotasi diperbaiki, dan dataset diperbarui tanpa harus mengumpulkan ulang semua data.

### `datasets/object_detection/`

Berisi dataset yang telah disiapkan untuk pelatihan model. Data dibagi menjadi:

- **Train:** digunakan untuk melatih model.
- **Validation:** digunakan untuk memantau performa selama pengembangan model.
- **Test:** digunakan untuk mengevaluasi model pada data pengujian.

Gambar dan file anotasi harus memiliki pasangan nama file yang sesuai.

### `datasets/reports/`

Berisi laporan pemeriksaan dataset, termasuk hasil validasi data yang tersedia.

### `runs/`

Berisi hasil eksperimen training, metrik, grafik, file pendukung evaluasi, dan bobot model yang dihasilkan.

## 5. Format Dataset YOLO

Anotasi object detection YOLO disimpan dalam file `.txt`, dengan satu baris untuk setiap objek:

```text
class_id x_center y_center width height
```

Koordinat posisi dan ukuran kotak pembatas dinormalisasi terhadap ukuran gambar, sehingga nilainya berada pada rentang 0 sampai 1.

Contoh format:

```text
1 0.500000 0.450000 0.300000 0.250000
```

Contoh tersebut hanya menunjukkan format anotasi, bukan anotasi aktual dari gambar dalam dataset.

Pastikan bahwa:

1. Setiap ID kelas sesuai dengan `data.yaml`.
2. Koordinat anotasi berada pada rentang yang valid.
3. Nama file gambar dan label cocok.
4. Objek yang terlihat pada gambar telah diberi anotasi sesuai kebutuhan.
5. Data testing tidak digunakan sebagai data training.

## 6. Konfigurasi Dataset

Contoh konfigurasi `configs/data.yaml`:

```yaml
path: datasets/object_detection
train: images/train
val: images/val
test: images/test

names:
  0: field
  1: box_hitam
  2: box_putih
```

Path dataset dibuat relatif terhadap repository agar lebih mudah digunakan di komputer lain.

Jika direktori dataset dipindahkan atau nama kelas berubah, konfigurasi harus diperiksa dan disesuaikan sebelum training.

## 7. Eksperimen Training

Repository menyimpan hasil beberapa eksperimen:

| Eksperimen | Tujuan |
|---|---|
| `e1_pretrained` | Training menggunakan model pretrained |
| `e2_scratch` | Training dari konfigurasi model tanpa pretrained |
| `e3_finetuning` | Eksperimen fine-tuning dengan konfigurasi yang disiapkan |
| `e1_fixed_lr` | Eksperimen tambahan dengan pengaturan learning rate tetap |
| `e3_fixed_lr` | Eksperimen tambahan dengan pengaturan learning rate tetap |

Hasil setiap eksperimen dapat mencakup bobot model, metrik, grafik, dan file pendukung lainnya, bergantung pada hasil yang tersimpan.

Perbandingan model sebaiknya mempertimbangkan precision, recall, mAP50, mAP50-95, serta hasil pengujian pada data test. Model terbaik tidak ditentukan hanya berdasarkan nama eksperimen atau ukuran file bobot.

## 8. Instalasi Lingkungan

Gunakan Python dan lingkungan virtual yang kompatibel dengan versi Ultralytics, PyTorch, serta perangkat keras yang digunakan.

Contoh langkah instalasi pada Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install ultralytics pyyaml
```

Instalasi PyTorch perlu disesuaikan dengan dukungan CPU atau GPU pada komputer yang digunakan. Untuk penggunaan GPU NVIDIA, periksa kompatibilitas versi PyTorch dan CUDA sebelum memulai training.

## 9. Training Model

Perintah training dapat dijalankan menggunakan Ultralytics YOLO. Contoh berikut merupakan pola perintah dasar; nama model dan parameter harus disesuaikan dengan konfigurasi eksperimen yang ingin dijalankan.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="configs/data.yaml",
    epochs=30,
    imgsz=416,
    batch=16
)
```

Perintah tersebut merupakan contoh training dasar, bukan jaminan bahwa seluruh parameter sama dengan eksperimen yang tersimpan dalam repository.

Untuk mereproduksi eksperimen tertentu, gunakan konfigurasi eksperimen yang sesuai dan periksa nilai model, pretrained, epochs, image size, batch size, seed, serta parameter lainnya.

## 10. Evaluasi Model

Evaluasi dilakukan menggunakan data test yang terpisah dari data training.

Contoh evaluasi menggunakan bobot model:

```python
from ultralytics import YOLO

model = YOLO("runs/e1_pretrained/weights/best.pt")

metrics = model.val(
    data="configs/data.yaml",
    split="test"
)

print(metrics)
```

Sesuaikan path bobot dengan eksperimen yang akan diuji. Pastikan model yang dipilih memiliki kelas yang sama dengan konfigurasi dataset.

Hasil evaluasi perlu digunakan untuk mengetahui kemampuan model mendeteksi objek, kesalahan klasifikasi, objek yang terlewat, dan kemungkinan deteksi palsu.

## 11. Menggunakan Bobot Model

File `best.pt` merupakan bobot model yang dipilih berdasarkan hasil validasi selama training. File tersebut dapat digunakan untuk inferensi.

Contoh deteksi pada gambar:

```python
from ultralytics import YOLO

model = YOLO("runs/e1_pretrained/weights/best.pt")

results = model.predict(
    source="contoh_gambar.jpg",
    conf=0.25,
    save=True
)
```

Ganti `contoh_gambar.jpg` dengan path gambar yang akan diuji. Ganti path checkpoint jika menggunakan model dari eksperimen lain.

Sebelum digunakan, periksa nama kelas yang tersimpan dalam checkpoint untuk memastikan model mendukung kelas `field`, `box_hitam`, dan `box_putih` sesuai kebutuhan.

## 12. Praktik Pengelolaan Dataset

- Simpan data mentah secara terpisah dari dataset hasil pembagian.
- Periksa anotasi sebelum memasukkan gambar ke proses training.
- Gunakan pembagian train, validation, dan test yang jelas.
- Hindari gambar duplikat atau gambar yang hampir identik tersebar di beberapa split, karena dapat membuat evaluasi terlalu optimistis.
- Catat konfigurasi setiap eksperimen agar hasilnya dapat dibandingkan.
- Jangan memasukkan path absolut komputer pribadi ke file konfigurasi yang akan dibagikan.
- Simpan bobot model yang diperlukan dan hindari mengunggah checkpoint yang tidak dibutuhkan.

## 13. Batasan dan Pengembangan Selanjutnya

Performa model bergantung pada kualitas dan jumlah data, variasi pencahayaan, sudut kamera, kondisi latar belakang, serta ketepatan anotasi.

Pengembangan selanjutnya dapat mencakup penambahan variasi gambar untuk setiap kelas, pemeriksaan kualitas anotasi, evaluasi model secara konsisten, serta pengujian deteksi menggunakan kamera secara langsung.

## 14. Catatan Repository

Repository ini berfungsi sebagai dokumentasi dataset, konfigurasi, dan hasil eksperimen. Perintah dan path yang digunakan perlu diperiksa kembali sesuai file aktual yang tersedia sebelum menjalankan training atau evaluasi.

**Tujuan akhir:** menyediakan alur kerja yang terdokumentasi untuk menyiapkan dataset, melatih model YOLO, membandingkan hasil eksperimen, dan menggunakan model untuk deteksi objek.
