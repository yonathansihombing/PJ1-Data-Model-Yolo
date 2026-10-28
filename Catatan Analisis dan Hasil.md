# RET503 YOLO — Analisis Hasil Eksperimen

**Nama:** Yonatan Andreas  
**NIM:** 4222411028

## 1. Ruang lingkup dan status data

Dokumen ini merangkum angka eksperimen yang telah dicatat dalam laporan sebelumnya dan artefak hasil yang tersimpan di repositori. Angka di bawah **bukan hasil training ulang yang dilakukan saat revisi dokumentasi ini**. Perubahan dokumentasi tidak mengubah dataset, checkpoint, atau hasil eksperimen.

Repositori saat ini memuat artefak eksperimen di `runs/`, termasuk hasil E1/E2/E3 dan file evaluasi. Keberadaan artefak tidak dengan sendirinya membuktikan bahwa eksperimen dapat direproduksi pada lingkungan baru. Reproduksi memerlukan dataset yang cocok, dependensi yang kompatibel, konfigurasi yang sama, dan pemeriksaan checkpoint.

## 2. Konfigurasi eksperimen utama

| Eksperimen | Model | Pretrained | Epoch maksimum | Image size | Batch | Seed | Learning rate awal |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 — pretrained | `yolo26n.pt` | Ya | 50 | 640 | 8 | 42 | 0.01 |
| E2 — scratch | `yolo26n.yaml` | Tidak | 50 | 640 | 8 | 42 | 0.01 |
| E3 — fine-tuning | Mewarisi E1 | Ya | 50 | 640 | 8 | 42 | 0.005 |

E2 menggunakan arsitektur tanpa bobot pretrained. E3 mewarisi konfigurasi E1 dan mengubah satu parameter, yaitu learning rate awal. Parameter lainnya perlu tetap sama agar perbandingan E1 dan E3 mengisolasi perubahan tersebut.

Konfigurasi dataset yang tersimpan menetapkan kelas `field`, `box_hitam`, dan `box_putih`, dengan direktori `images/train`, `images/val`, dan `images/test`.

## 3. Hasil validation yang tercatat

Tabel berikut mempertahankan angka yang tercatat pada laporan sebelumnya. Angka harus dicocokkan dengan `runs/summary_val.csv` dan hasil eksperimen terkait sebelum digunakan sebagai angka final dalam publikasi atau penilaian.

| Eksperimen | Precision | Recall | mAP50 | mAP50–95 | Inferensi (ms/gambar) | Training (menit) |
|---|---:|---:|---:|---:|---:|---:|
| E1 pretrained | 0.9690 | 0.9921 | 0.9950 | 0.8737 | 6.75 | 3.35 |
| E2 scratch | 0.3181 | 0.5675 | 0.3967 | 0.1289 | 24.18 | 3.11 |
| E3 tuning | 0.9690 | 0.9921 | 0.9950 | 0.8737 | 28.56 | 3.20 |

Berdasarkan angka yang dilaporkan, E1 dan E3 memiliki metrik validation yang sama pada presisi empat desimal. Data tersebut tidak menunjukkan peningkatan akurasi E3 terhadap E1. Waktu inferensi yang tercatat berbeda, tetapi perbandingan kecepatan perlu dilakukan dengan prosedur, perangkat, ukuran gambar, dan kondisi runtime yang sama. E2 mencatat nilai metrik lebih rendah pada dataset ini; hal tersebut tidak membuktikan bahwa model scratch selalu lebih buruk pada semua dataset.

## 4. Hasil final test E1 yang tercatat

Laporan sebelumnya mencatat test set berisi 9 gambar dan 9 objek, dengan hasil berikut:

| Metrik | Nilai tercatat |
|---|---:|
| Precision | 0.9282 |
| Recall | 0.9333 |
| mAP50 | 0.9700 |
| mAP50–95 | 0.9160 |
| Inferensi (ms/gambar) | 4.89 |

### Hasil per kelas

| Kelas | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| `field` | 0.9935 | 1.0000 | 0.9950 | 0.9450 |
| `box_hitam` | 0.8144 | 0.8000 | 0.9200 | 0.8079 |
| `box_putih` | 0.9766 | 1.0000 | 0.9950 | 0.9950 |

Di antara tiga kelas, `box_hitam` memiliki precision, recall, AP50, dan AP50–95 yang paling rendah pada tabel tersebut. Kelas ini layak menjadi prioritas pemeriksaan anotasi dan penambahan variasi data. Namun, jumlah test yang hanya 9 gambar dan 9 objek terlalu kecil untuk mendukung klaim generalisasi yang kuat. Kesimpulan perlu dibatasi pada data yang diuji.

**Penting:** sebelum mengutip angka final test, cocokkan angka dan jumlah sampel dengan `runs/final_test.json`, `runs/final_test.csv`, serta artefak per kelas. Jika nilainya berbeda, gunakan artefak evaluasi yang tervalidasi dan perbarui tabel ini.

## 5. Eksperimen fixed learning rate tambahan

Laporan sebelumnya mencatat E1 fixed-LR (`AdamW`, `lr0=0.01`) dan E3 fixed-LR (`AdamW`, `lr0=0.005`) memiliki hasil validation yang sama:

| Eksperimen | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| E1 fixed-LR | 0.5802 | 0.2698 | 0.4286 | 0.4129 |
| E3 fixed-LR | 0.5802 | 0.2698 | 0.4286 | 0.4129 |

Kedua eksperimen ini merupakan percobaan tambahan dan tidak boleh dicampur dengan E1/E3 utama yang menggunakan konfigurasi optimizer `auto`. Hasilnya perlu dibaca bersama konfigurasi efektif, log training, dan ringkasan evaluasi yang sesuai.

## 6. Kesimpulan

1. Pada angka validation yang tercatat, E1 dan E3 menghasilkan metrik yang sama; belum ada bukti bahwa perubahan `lr0` pada E3 meningkatkan metrik deteksi.
2. E2 mencatat performa lebih rendah dibandingkan E1/E3 pada dataset ini.
3. Hasil test E1 yang tercatat menunjukkan nilai agregat tinggi, tetapi ukuran test yang kecil membatasi kekuatan kesimpulan.
4. `box_hitam` memiliki metrik per kelas terendah pada tabel test yang tercatat dan perlu diperiksa lebih lanjut.
5. Kesimpulan ini bersifat deskriptif untuk hasil yang tercatat. Training ulang, pengujian pada data baru, atau klaim bahwa model siap untuk kondisi operasional memerlukan bukti pengujian tambahan.

## 7. Batas verifikasi

Revisi ini memperbaiki konsistensi dan bahasa laporan; tidak menjalankan ulang training atau test set. Validasi round-trip YOLO↔COCO juga belum dapat dinyatakan berhasil tanpa bukti eksekusi konversi balik dan perbandingan hasil. Catat hasil pemeriksaan tersebut sebelum menyatakan pipeline terverifikasi sepenuhnya.
