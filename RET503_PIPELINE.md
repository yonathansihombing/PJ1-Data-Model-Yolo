# RET503 — Pipeline Dataset, Training, dan Evaluasi YOLO

## 1. Tujuan dan batas pipeline

Pipeline eksperimen RET503 menggunakan **object detection**. Data sumber proyek kamera berupa anotasi segmentation polygon di `datasets/raw_yolo/labels_seg/`; `prepare_polygon_detection.py` mengubah setiap polygon menjadi kotak pembatas sumbu-sejajar (*axis-aligned bounding box*, AABB), lalu membagi gambar menjadi train, validation, dan test. Label polygon sumber tidak diubah.

Jalur segmentation tetap terpisah. Hasil segmentation tidak boleh dicampur dengan metrik eksperimen object detection E1/E2/E3.

## 2. Konfigurasi yang digunakan

- `configs/data.yaml`: lokasi dataset deteksi dan nama kelas; dibuat ulang oleh skrip persiapan berdasarkan `datasets/raw_yolo/classes.txt`.
- `configs/e1_pretrained.yaml`: baseline dengan `yolo26n.pt`, pretrained.
- `configs/e2_scratch.yaml`: `yolo26n.yaml`, inisialisasi tanpa bobot pretrained.
- `configs/e3_finetuning.yaml`: mewarisi E1 dan mengubah satu hyperparameter, `train.lr0` dari 0.01 menjadi 0.005.
- `configs/inference.yaml`: checkpoint dan parameter inferensi.

Kelas harus mempunyai urutan yang sama pada `classes.txt`, anotasi, dan konfigurasi dataset. Jangan mengganti urutan kelas setelah anotasi dibuat tanpa memvalidasi ulang semua label.

## 3. Siapkan lingkungan (Windows PowerShell)

Jalankan dari direktori repositori:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Pasang PyTorch sesuai sistem operasi, versi Python, dan CPU/GPU melalui pemilih resmi: https://pytorch.org/get-started/locally/

Setelah itu, instal dependensi proyek:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU tidak tersedia')"
python -c "import cv2, yaml, ultralytics; print('OpenCV', cv2.__version__, '| Ultralytics', ultralytics.__version__)"
```

Jika CUDA tidak tersedia, periksa driver NVIDIA, kompatibilitas PyTorch, dan versi Python. Toolkit CUDA yang terpasang sendiri tidak menjamin bahwa PyTorch dapat menggunakan GPU.

## 4. Periksa sumber dan siapkan dataset

Sebelum memproses data, pastikan berkas berikut tersedia:

- `datasets/raw_yolo/images/`
- `datasets/raw_yolo/labels_seg/`
- `datasets/raw_yolo/classes.txt`

Periksa setiap anotasi dan pastikan label usulan auto-labeling telah direview. Jalankan konversi dan split:

```powershell
python prepare_polygon_detection.py
```

Rasio bawaan adalah 70% train, 20% validation, dan 10% test dengan seed 42. Skrip akan gagal jika menemukan gambar yang tidak terbaca atau label tidak valid/hilang, kecuali gambar tanpa label secara eksplisit diizinkan menggunakan `--allow-unlabeled`. Gunakan opsi tersebut hanya untuk gambar latar yang memang tidak mengandung objek kelas target.

Jika folder keluaran sudah berisi data, skrip menolak menimpanya. Untuk membuat ulang dataset deteksi:

```powershell
python prepare_polygon_detection.py --overwrite
```

**Perhatian:** `--overwrite` menghapus keluaran dataset deteksi yang ada sebelum membuat ulang. Cadangkan data yang perlu dipertahankan terlebih dahulu.

Hasil utama: `datasets/object_detection/images/{train,val,test}/`, `datasets/object_detection/labels/{train,val,test}/`, `datasets/object_detection/split_summary.json`, dan `configs/data.yaml`.

Pembagian dilakukan secara acak per gambar. Jika beberapa gambar berasal dari video atau adegan yang sama, split acak per gambar berisiko menyebabkan kebocoran data. Untuk evaluasi yang lebih kuat, kelompokkan gambar dari sesi/adegan yang sama ke split yang sama.

## 5. Dry-run dan training

Jalankan dry-run untuk memeriksa konfigurasi dan kesiapan dataset tanpa melatih model:

```powershell
python train_yolo.py --exp e1 --dry-run
python train_yolo.py --exp e2 --dry-run
python train_yolo.py --exp e3 --dry-run
```

Jika pemeriksaan berhasil, jalankan eksperimen:

```powershell
python train_yolo.py --exp e1 --allow-download
python train_yolo.py --exp e2
python train_yolo.py --exp e3
```

`--allow-download` mengizinkan pengunduhan bobot pretrained yang diperlukan E1/E3 jika belum tersedia di lokasi bobot yang diperiksa skrip. E2 menggunakan definisi arsitektur tanpa pretrained. Gunakan `python train_yolo.py --help` untuk memeriksa opsi yang tersedia.

| Eksperimen | Tujuan | Perbedaan utama |
|---|---|---|
| E1 | Baseline pretrained | `yolo26n.pt` |
| E2 | Training dari inisialisasi acak | `yolo26n.yaml`, tanpa pretrained |
| E3 | Menguji perubahan learning rate | E1 dengan `lr0=0.005`, dibandingkan E1 `lr0=0.01` |

E1/E2 sebaiknya memakai parameter training lain yang sama. E3 harus berbeda dari E1 hanya pada satu hyperparameter yang ditetapkan. Jangan membandingkan hasil jika konfigurasi efektif, data, atau prosedur evaluasi berbeda tanpa mencatat perbedaan tersebut.

Hasil disimpan di `runs/<nama_eksperimen>/`, biasanya termasuk `weights/best.pt`, `weights/last.pt`, `results.csv`, grafik, konfigurasi efektif, dan metadata.

## 6. Evaluasi validation dan final test

Evaluasi validation dan perbandingan E1/E2/E3:

```powershell
python evaluate_yolo.py
```

Pilih model berdasarkan hasil validation, terutama mAP50–95, serta periksa precision, recall, hasil per kelas, dan waktu inferensi.

Setelah model final dipilih, jalankan test satu kali. Contoh memilih E1 secara eksplisit:

```powershell
python evaluate_yolo.py --final-test --select E1
```

Ganti E1 dengan E2 atau E3 hanya jika eksperimen tersebut mempunyai hasil validation yang valid. Jika tidak memberikan `--select`, skrip memilih eksperimen valid dengan mAP50–95 validation tertinggi. Opsi `--force-retest` mengizinkan test ulang dan menandainya sebagai pelanggaran prosedur; hindari penggunaannya untuk laporan final.

Hasil evaluasi disimpan di `runs/summary_val.csv`, `runs/comparison.md`, dan artefak terkait. Hasil test disimpan di `runs/final_test.json` dan `runs/final_test.csv`.

## 7. Konversi YOLO ke COCO (opsional)

Untuk mengekspor anotasi detection ke COCO JSON:

```powershell
python prepare_dataset.py yolo-to-coco --root datasets/object_detection --out datasets/object_detection/annotations_coco.json
```

Perintah ini adalah ekspor satu arah. Jangan menyatakan validasi *round-trip* YOLO↔COCO telah berhasil sebelum menjalankan konversi balik ke direktori sementara dan membandingkan jumlah objek serta koordinat dengan toleransi pembulatan.

## 8. Jalur segmentation (terpisah)

`train_segmentation.py` secara default mencari `datasets/segmentation/data.yaml` dan menggunakan `yolo11n-seg.pt`. Pastikan konfigurasi data tersebut benar-benar tersedia sebelum training. Keberadaan artefak di folder `segment/` saja tidak membuktikan bahwa dataset sumber segmentation saat ini siap digunakan.

Periksa opsi:

```powershell
python train_segmentation.py --help
```

## 9. Catatan reproducibility

Sebelum menyerahkan laporan, simpan versi Python, PyTorch, Ultralytics, status CUDA, konfigurasi eksperimen, seed, ringkasan dataset, dan artefak hasil. Bedakan metrik yang berasal dari log lama dengan hasil training atau evaluasi yang baru dijalankan. Jangan mengarang metrik yang tidak tersedia dan jangan memakai test set untuk memilih model.
