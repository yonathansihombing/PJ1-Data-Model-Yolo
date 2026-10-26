# RET503 — Pipeline dari data ke model YOLO

## Penting: dua jalur yang berbeda

Project camera menyimpan anotasi Polygon/segmentation (`datasets/raw_yolo/labels_seg`). Namun latihan RET503 yang dinilai di sini memakai **object detection** dengan label YOLO `class_id cx cy w h`. Skrip `prepare_polygon_detection.py` mengubah setiap polygon menjadi kotak AABB pembungkus dan menyimpan hasil deteksi di `datasets/object_detection/labels/`. Label polygon asli tidak diubah. Segmentation tetap menjadi fitur tambahan, bukan pengganti E1/E2/E3 detection.

## Berkas konfigurasi yang disertakan

- `configs/data.yaml` — akan ditulis ulang berdasarkan kelas yang ada di `datasets/raw_yolo/classes.txt` ketika split berhasil.
- `configs/e1_pretrained.yaml` — YOLOv8n pretrained, baseline.
- `configs/e2_scratch.yaml` — YOLOv8n arsitektur `.yaml`, tanpa pretrained; parameter training selain sumber bobot sama dengan E1.
- `configs/e3_finetuning.yaml` — sama dengan E1 kecuali `train.lr0` diubah dari `0.01` menjadi `0.005` (satu hyperparameter).
- `configs/inference.yaml` — konfigurasi inference.

## Urutan penggunaan (PowerShell, dari folder project)

1. Instal dependensi:
   ```powershell
   python -m pip install -r requirements.txt
   ```
   Jika PyTorch belum ada atau instalasi GPU diperlukan, pasang PyTorch yang sesuai dari petunjuk resmi PyTorch terlebih dahulu.

2. Pastikan `datasets/raw_yolo/images/`, `datasets/raw_yolo/labels_seg/`, dan `datasets/raw_yolo/classes.txt` sudah ada. Label final harus direview; label usulan yang kosong/tidak yakin jangan dianggap sebagai anotasi valid.

3. Validasi + konversi polygon ke kotak deteksi + split 70/20/10:
   ```powershell
   python prepare_polygon_detection.py
   ```
   Jika `datasets/object_detection` sudah berisi hasil sebelumnya dan memang ingin dibuat ulang:
   ```powershell
   python prepare_polygon_detection.py --overwrite
   ```
   Output mencakup `datasets/object_detection/images/{train,val,test}`, `labels/{train,val,test}`, `split_summary.json`, dan `configs/data.yaml`. Default seed `42` membuat split reproducible.

4. Cek konfigurasi dan data tanpa menjalankan training:
   ```powershell
   python train_yolo.py --exp e1 --dry-run
   python train_yolo.py --exp e2 --dry-run
   python train_yolo.py --exp e3 --dry-run
   ```

5. Jalankan eksperimen terpisah:
   ```powershell
   python train_yolo.py --exp e1 --allow-download
   python train_yolo.py --exp e2
   python train_yolo.py --exp e3
   ```
   `--allow-download` hanya dibutuhkan jika `weights/yolov8n.pt` belum tersedia. E2 memerlukan definisi arsitektur `yolov8n.yaml` dari versi Ultralytics terpasang. Untuk membandingkan secara adil, jangan mengubah parameter lain antar eksperimen. Jangan memakai `--exp all` jika ingin mengawasi dan menyimpan log setiap eksperimen satu per satu.

6. Evaluasi validation untuk perbandingan:
   ```powershell
   python evaluate_yolo.py
   ```
   Pilih model final berdasarkan validation, bukan test.

7. Jalankan final test **sekali saja** setelah model final dipilih:
   ```powershell
   python evaluate_yolo.py --final-test --select E1
   ```
   Ganti E1 dengan E2/E3 sesuai model yang dipilih. Hasil, metrik dan ringkasan akan ditulis di `runs/`.

8. Konversi dataset detection YOLO ke COCO JSON (opsional untuk bukti konversi):
   ```powershell
   python prepare_dataset.py yolo-to-coco --root datasets/object_detection --out datasets/object_detection/annotations_coco.json
   ```
   Untuk uji round-trip yang kuat, konversi JSON tersebut kembali ke folder dataset sementara lalu bandingkan jumlah objek dan koordinat (toleransi pembulatan). Jangan menimpa dataset sumber.

## Hasil eksperimen yang harus dikumpulkan

Simpan output asli dari pelatihan, jangan dibuat manual: `runs/e1_pretrained/`, `runs/e2_scratch/`, `runs/e3_finetuning/` termasuk `weights/best.pt`, `weights/last.pt`, `results.csv`, grafik, confusion matrix, `args.yaml`, `experiment_meta.json`; serta file evaluasi `runs/summary_val.csv`, tabel perbandingan dan `runs/final_test.json`/`.csv` (nama aktual mengikuti `evaluate_yolo.py`). Tambahkan screenshot log terminal dan catat versi Python, Ultralytics, PyTorch, hardware, durasi, seed dan jumlah data.

## Batasan yang harus dinyatakan di laporan

- ZIP template tidak menyertakan foto pribadi, label final, checkpoint hasil training, atau metrik eksperimen karena semua itu harus berasal dari dataset/komputer pengguna dan eksperimen yang benar-benar dijalankan.
- Jika instruksi PPT mewajibkan 200 gambar per kelas dan tiga kelas COCO, dataset 90 gambar `field`, `box_hitam`, `box_putih` **belum memenuhi syarat tersebut**. Perlu dataset sesuai pilihan kelas COCO dan jumlah target yang diwajibkan; jangan mengklaim terpenuhi hanya karena pipeline bisa dijalankan.
- Split acak per gambar sesuai skrip. Jika foto berurutan dari video/adegan yang sama, gunakan split berbasis grup untuk mencegah kebocoran data antar split.
- E1/E2/E3 harus benar-benar dijalankan. Berkas konfigurasi dan kode bukan bukti bahwa training sukses atau bahwa model punya metrik tertentu.
