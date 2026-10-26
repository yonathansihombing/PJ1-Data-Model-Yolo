# Checklist audit RET503 — Yonatan Andreas (4222411028)

| Item | Lokasi/nama yang diharapkan | Status audit | Catatan |
|---|---|---|---|
| Kode aplikasi kamera | `main.py` dan modul terkait dalam ZIP | Tersedia | Pertahankan fungsi kamera/kalibrasi/field/segmentation |
| Pretrained YOLO | `yolo26n.pt` di ZIP | Tersedia | File ada dalam arsip |
| Konfigurasi E1 | `configs/e1_pretrained.yaml` | Tersedia | Pretrained YOLO26n, lr0 0.01, optimizer auto |
| Konfigurasi E2 | `configs/e2_scratch.yaml` | Tersedia | Scratch, pretrained false |
| Konfigurasi E3 | `configs/e3_finetuning.yaml` | Tersedia | Satu perubahan: lr0 0.005 |
| Dataset config | `configs/data.yaml` | Tersedia | Tiga kelas; dataset aktual tidak ada dalam ZIP |
| Dataset + label | `datasets/object_detection/` | Belum tersedia | Perlu arsip lokal aktual; jangan regenerasi/ubah split tanpa izin |
| Hasil E1 `results.png` | `runs/e1_pretrained/results.png` | Belum tersedia di ZIP | Wajib dari hasil training asli |
| Hasil E3 `results.png` | `runs/e3_finetuning/results.png` | Belum tersedia di ZIP | Wajib dari hasil training asli |
| E1/E3 `results.csv` | Folder run masing-masing | Belum tersedia di ZIP | Ambil dari run asli |
| Bobot E1/E3 | `weights/best.pt` dan bila memungkinkan `last.pt` | Belum tersedia di ZIP | Jangan dibuat dari metrik |
| Hasil validation | `runs/summary_val.csv`, `runs/comparison.md` | Belum tersedia di ZIP | Metrik sudah diketahui dari terminal |
| Final test JSON/CSV | `runs/final_test.json`, `runs/final_test.csv` | Belum tersedia di ZIP | Hasil sudah dicatat; jangan retest |
| `hasil_praktik_4222411028.txt` | File ini | Dibuat sebagai draf | Angka dari terminal; belum diverifikasi dengan artefak asli |
| `hasil_praktik_4222411028.csv` | File ini | Dibuat sebagai draf | Angka dari terminal; belum diverifikasi dengan artefak asli |
| README analisis | `README.md` | Dibuat sebagai draf | Menyebut keterbatasan secara eksplisit |
| PPT dosen | `RET503_Data dan YOLO` | Belum ditemukan | Perlu diunggah untuk audit persyaratan teknis lengkap |
| Tabel pengumpulan | Gambar tabel dosen | Belum ditemukan pada audit ini | Rangkuman ketentuan tersedia dari pesan pengguna |
| Round-trip YOLO↔COCO | Bukti konversi dan pemeriksaan | Belum diverifikasi | Jangan mengklaim selesai sebelum dijalankan |

## Keputusan
Paket ini belum dinyatakan final/siap dikumpulkan karena dataset, grafik asli E1/E3, artefak training, hasil evaluasi tersimpan, dan PPT dosen belum dapat diverifikasi dalam ZIP sumber. Tidak ada training baru atau pengujian ulang test set yang dilakukan.
