# Studio 2 — Klasifikasi Sampah · Kelompok 3

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No  | Nama Lengkap                 | NIM       | Kontribusi utama |
| --- | ---------------------------- | --------- | ---------------- |
| 1   | Elfa Noviana Sari            | 123140066 |                  |
| 2   | Mohd.Musyaffa Alief Athallah | 123140184 |                  |
| 3   | Revolusi Al Ghifari          | 123140199 |                  |

## Status Pengerjaan

| Tahap                 | Status      | Berkas                             |
| --------------------- | ----------- | ---------------------------------- |
| Data loading          | Selesai     | `notebooks/01_data_loading.ipynb`  |
| EDA                   | Selesai     | `notebooks/02_eda.ipynb`           |
| Preprocessing         | Selesai     | `notebooks/03_preprocessing.ipynb` |
| Pelatihan model       | Selesai     | `notebooks/04_train_baseline_resnet18.ipynb`, `notebooks/05_train_efficientnet_b0.ipynb` |
| Evaluasi & submission | Selesai     | `submission/model.py`, `submission/model.pth` |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: **EfficientNet-B0 pretrained ImageNet** (fine-tuning penuh). Dipilih karena Macro-F1 validasi lokal terbaik di antara model yang dicoba, sekaligus paling ringan. Pembanding: baseline ResNet-18 pretrained ImageNet (dari starter kit).
- **Detail arsitektur EfficientNet-B0**:
  - **Backbone (feature extractor)**: EfficientNet-B0 versi torchvision (`torchvision.models.efficientnet_b0`), bobot awal **pretrained ImageNet-1K** (1.000 kelas). Terdiri dari 16 blok MBConv (Mobile Inverted Bottleneck Convolution) dengan *squeeze-and-excitation* dan *compound scaling* (lebar, kedalaman, resolusi diskalakan bersamaan). Blok terakhir menghasilkan feature map **1280 kanal** (`model.features`).
  - **Head (classifier)**: bagian `model.classifier` (bawaan EfficientNet) diganti menjadi:
    - `Dropout(p=0,3)` — regularisasi (naik dari 0,2 bawaan) untuk menahan overfitting pada dataset kecil;
    - `Linear(1280 → 5)` — memetakan 1280 fitur menjadi logits 5 kelas sampah, menggantikan layer 1000 kelas ImageNet.
  - **Perbedaan dengan ResNet-18**: ResNet-18 memakai *residual block* sederhana dan head `Linear(512 → 5)`; EfficientNet-B0 memakai MBConv + SE dan head `Linear(1280 → 5)`. EfficientNet-B0 jauh lebih efisien (4,0 juta vs 11,2 juta parameter) namun ImageNet top-1 lebih tinggi (77,7% vs 69,8%).
  - **Skema fine-tuning**: seluruh bobot dilatih (*full fine-tuning*, bukan hanya head), dengan `AdamW` (lr 5e-4, weight decay 1e-4) dan `CosineAnnealingLR` (T_max 15 epoch). Learning rate kecil dipakai agar fitur pretrained ikut menyesuaikan tanpa rusak.
  - **Output**: logits `[batch, 5]`, urutan kelas alfabetis (0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam) — sama dengan urutan `ImageFolder`.
  - **Kontrak submission**: `get_model()` di `submission/model.py` membangun arsitektur yang sama tetapi dengan `weights=None` (tanpa mengunduh bobot); bobot terbaik dimuat dari `model.pth` oleh `evaluate.py`.
- **Strategi pelatihan**: sudah dijalankan.
  - checkpoint terbaik dipilih berdasarkan **Macro-F1** pada split validasi lokal, bukan akurasi;
  - `label_smoothing` untuk meredam label noise;
  - penanganan ketidakseimbangan kelas: saat ini belum ada penanganan khusus (rasio kelas terbesar/terkecil ±1,84× dianggap masih wajar). **Rencana selanjutnya**: membandingkan tanpa penanganan, _class-weighted loss_, dan `WeightedRandomSampler`.
- **Data tambahan**: tidak ada.
- **Ukuran input & preprocessing**:
  - input 224×224, normalisasi statistik ImageNet;
  - train: `RandomResizedCrop(scale 0,5–1,0)`, flip horizontal, `ColorJitter`;
  - validasi: `Resize(256)` + `CenterCrop(224)` (`get_transform()` di `submission/model.py`).
  - **Rencana selanjutnya**: menguji alternatif tanpa crop (`Resize` persegi) karena center crop memotong gambar yang sangat lebar.

### Temuan data (ringkasan EDA)

- Train set: **2.753 gambar**, 5 kelas: `botol_plastik` 523, `gelas_plastik` 521, `kertas_kardus` 365, `saset_kemasan` 673, `styrofoam` 671 (rasio kelas terbesar/terkecil ±1,84×).
- Resolusi cukup besar (median sisi pendek ±700–800 px); hanya 1,2% gambar berukuran < 224 px.
- Warna tidak membedakan kelas (klasifikasi dari fitur warna saja ±29%, hampir sama dengan tebakan acak).
- Banyak gambar katalog/iklan dengan teks dan watermark → ada risiko model memakai teks sebagai jalan pintas.
- 16 pasangan near-duplicate (dHash 256-bit), semuanya satu kelas. Split dibuat **per grup duplikat** supaya tidak ada kebocoran antara train dan validasi.
- Label noise sekitar **0,5%** (13 gambar jelas salah label, hasil cek visual dari kandidat k-nearest neighbor). Data dibiarkan apa adanya dan ditangani dengan `label_smoothing`.

### Split validasi lokal

15% per kelas (stratified, seed 42), menjaga grup duplikat tetap utuh: **2.343 gambar train** (`my_train/`) dan **410 gambar validasi** (`my_val/`). Validation set resmi disimpan dosen dan tidak pernah dilihat.

## Hasil

| Putaran            | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
| ------------------ | ------ | ------------------------------- | ---------------------------- |
| Preview 1 (14 Okt) | [`7ca6be1`](https://github.com/Escavora/studio-proyek2-cnn-klasifikasi-sampah/commit/7ca6be18ac6ced32b6617ca35faa83811c58cdc8) | 0,9581 (EfficientNet-B0)        |                              |
| Preview 2 (21 Okt) |        |                                 |                              |
| Final (25 Okt)     |        |                                 |                              |

> Kolom **Commit** diisi hash commit yang dijadikan submission pada putaran tersebut (yang diambil asisten saat mengevaluasi).

Model yang di-submit: **EfficientNet-B0** (Macro-F1 lokal **0,9581**, akurasi **0,9585**, `model.pth` 16,3 MB).
Pembanding baseline **ResNet-18** pada split yang sama: Macro-F1 **0,9229**, akurasi 0,9244.

F1 per kelas (EfficientNet-B0) — `botol_plastik` 0,9548 · `gelas_plastik` 0,9625 · `kertas_kardus` 0,9533 · `saset_kemasan` 0,9641 · `styrofoam` 0,9557.

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- **ResNet-18 → EfficientNet-B0** berhasil menaikkan Macro-F1 lokal dari **0,9229** ke **0,9581** (+0,0352), padahal parameter jauh lebih sedikit (4,0 juta vs 11,2 juta). Karena itu EfficientNet-B0 dipilih sebagai model final.
- Epoch ditambah (8 → 15) karena backbone lebih kecil baru konvergen setelah belasan epoch.
- Kelas `kertas_kardus` paling sulit (contoh paling sedikit dan visual paling beragam).

## Cara Reproduksi

Dataset **tidak disertakan** di repositori (sesuai ketentuan tugas). Unduh dari tautan di halaman kompetisi, lalu ekstrak sehingga ada `train/<nama_kelas>/*.jpg` di root repositori.

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# jalankan berurutan: notebooks/01_data_loading.ipynb -> 02_eda.ipynb -> 03_preprocessing.ipynb
# (03 membuat my_train/ dan my_val/)

python evaluate.py --submission ./submission --data ./my_val   # setelah submission/model.pth tersedia
```

`requirements.txt` memakai PyTorch build CUDA 12.8 (Windows + GPU NVIDIA).

## Struktur Repositori

```
notebooks/
  01_data_loading.ipynb     # cek struktur dan integritas dataset
  02_eda.ipynb              # analisis data (distribusi, ukuran, duplikat, label noise)
  03_preprocessing.ipynb    # split per grup duplikat, transform, DataLoader
  04_train_baseline_resnet18.ipynb    # baseline pembanding (ResNet-18)
  05_train_efficientnet_b0.ipynb      # model final (EfficientNet-B0)
submission/
  model.py          # get_model() dan get_transform() (EfficientNet-B0)
  model.pth         # state_dict model terbaik (16,3 MB)
evaluate.py         # salinan skrip evaluasi resmi
train_baseline.py   # baseline starter kit (ResNet-18)
requirements.txt    # dependensi lingkungan pelatihan
```

Folder `train/`, `my_train/`, `my_val/`, dan `data/` ada di `.gitignore` dan tidak ikut diunggah.
