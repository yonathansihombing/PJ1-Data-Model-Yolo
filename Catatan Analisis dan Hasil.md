# RET503 YOLO — Catatan Analisis dan Hasil

**Nama:** Yonatan Andreas  
**NIM:** 4222411028

## Status dokumen
Dokumen ini adalah draf laporan hasil yang merangkum metrik dari output terminal pengguna. ZIP `camera_RET503_pipeline_revisi.zip` berisi kode dan konfigurasi, tetapi tidak menyertakan dataset `datasets/object_detection/` atau hasil eksperimen di `runs/`. Karena itu, draf ini belum menjadi paket pengumpulan yang sepenuhnya terverifikasi. Tidak ada training yang dijalankan ulang dan test set tidak diubah.

## Konfigurasi eksperimen dari ZIP
- **E1:** `yolo26n.pt`, `pretrained: true`, 50 epoch maksimum, imgsz 640, batch 8, seed 42, optimizer `auto`, `lr0: 0.01`.
- **E2:** `yolo26n.yaml`, `pretrained: false`, parameter training lain disamakan dengan E1.
- **E3:** mewarisi E1 dan mengubah satu parameter, `lr0` dari 0.01 menjadi 0.005.
- **Dataset config:** tiga kelas (`field`, `box_hitam`, `box_putih`), split path `images/train`, `images/val`, `images/test`.

## Perbandingan validation — eksperimen utama
| Eksperimen | Precision | Recall | mAP50 | mAP50–95 | Inferensi (ms/gambar) | Training (menit) |
|---|---:|---:|---:|---:|---:|---:|
| E1 pretrained | 0.9690 | 0.9921 | 0.9950 | 0.8737 | 6.75 | 3.35 |
| E2 scratch | 0.3181 | 0.5675 | 0.3967 | 0.1289 | 24.18 | 3.11 |
| E3 tuning | 0.9690 | 0.9921 | 0.9950 | 0.8737 | 28.56 | 3.20 |

Metrik di atas disalin dari output terminal sebelumnya. E1 dan E3 memiliki metrik validation yang sama pada angka yang dilaporkan; tidak ada bukti dari metrik tersebut bahwa E3 lebih unggul dari E1. E2 dari scratch jauh lebih rendah pada dataset ini.

## Final test — E1
Test set berisi 9 gambar dan 9 objek. Hasil tercatat: precision 0.9282, recall 0.9333, mAP50 0.9700, mAP50–95 0.9160, inferensi 4.89 ms/gambar.

### Hasil per kelas
| Kelas | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| field | 0.9935 | 1.0000 | 0.9950 | 0.9450 |
| box_hitam | 0.8144 | 0.8000 | 0.9200 | 0.8079 |
| box_putih | 0.9766 | 1.0000 | 0.9950 | 0.9950 |

Kelas `box_hitam` menjadi prioritas perbaikan karena metrik per kelas paling rendah di antara tiga kelas. Karena test set sangat kecil, hasil ini perlu ditafsirkan hati-hati.

## Eksperimen fixed-LR tambahan
E1 fixed-LR (`AdamW`, lr0 0.01) dan E3 fixed-LR (`AdamW`, lr0 0.005) sama-sama menghasilkan validation P=0.5802, R=0.2698, mAP50=0.4286, mAP50–95=0.4129. Kedua run ini berbeda dari eksperimen utama dan harus dilaporkan sebagai eksperimen tambahan, tidak menggantikan hasil E1/E3 utama.

## Integritas test set
Final test telah dijalankan sekali pada 2026-10-26 18:42:02, dengan E1 dipilih manual melalui `--select E1`. Jangan jalankan ulang `--force-retest` dan jangan menggunakan hasil test untuk tuning/pemilihan model.

## Keterbatasan dan item yang belum diverifikasi
1. Dataset dan label aktual tidak terdapat di ZIP audit, sehingga jumlah 90 gambar (63 train, 18 val, 9 test) belum diverifikasi dari file aktual pada tahap ini.
2. Artefak asli `runs/e1_pretrained/` dan `runs/e3_finetuning/` termasuk `results.png`, `results.csv`, `best.pt`, `last.pt`, dan `args.yaml` belum tersedia di ZIP.
3. PPT dosen dan gambar tabel pengumpulan belum ditemukan di berkas yang tersedia untuk audit ini; ketentuan teknis di luar informasi yang diberikan belum bisa dicocokkan.
4. Belum ada validasi round-trip YOLO↔COCO yang dapat dibuktikan dari ZIP.

## Cara menjalankan (tanpa mengulang training)
- Evaluasi validation (aman untuk pemilihan, tidak menggunakan test): `py evaluate_yolo.py`
- Final test **jangan dijalankan lagi** karena hasil test sudah tercatat.

Untuk membuat paket final benar-benar siap dikumpulkan, gabungkan dataset dan artefak run asli dari komputer lokal ke paket pengumpulan setelah diverifikasi. Jangan membuat ulang `results.png` dari angka ringkasan atau mengarang artefak training.
