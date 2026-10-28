# Perbandingan eksperimen (split VAL)

| Eksperimen | Status | Perubahan vs E1 | imgsz | Precision | Recall | mAP@50 | mAP@50-95 | ms/gambar | menit latih |
|---|---|---|---|---|---|---|---|---|---|
| E1 | OK | - | 640 | 0.969 | 0.9921 | 0.995 | 0.8737 | 6.75 | 3.35 |
| E2 | OK | model: yolo26n.pt -> yolo26n.yaml; pretrained: True -> False | 640 | 0.3181 | 0.5675 | 0.3967 | 0.1289 | 24.18 | 3.11 |
| E3 | OK | train.lr0: 0.01 -> 0.005 | 640 | 0.969 | 0.9921 | 0.995 | 0.8737 | 28.56 | 3.2 |

Model terpilih (mAP@50-95 val tertinggi): **E1**

Angka berasal dari evaluasi nyata `best.pt` pada split val. Test set belum/tidak dipakai di tabel ini.
