# Studio 2 — Klasifikasi Sampah · Kelompok 3

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No  | Nama Lengkap                 | NIM       | Kontribusi utama |
| --- | ---------------------------- | --------- | ---------------- |
| 1   | Elfa Noviana Sari            | 123140066 | Segera diisi     |
| 2   | Mohd.Musyaffa Alief Athallah | 123140184 | Segera diisi     |
| 3   | Revolusi Al Ghifari          | 123140199 | Segera diisi     |

## Status Pengerjaan

| Tahap                 | Status      | Berkas                             |
| --------------------- | ----------- | ---------------------------------- |
| Data loading          | Selesai     | `notebooks/01_data_loading.ipynb`  |
| EDA                   | Selesai     | `notebooks/02_eda.ipynb`           |
| Preprocessing         | Selesai     | `notebooks/03_preprocessing.ipynb` |
| Pelatihan model       | Belum mulai | —                                  |
| Evaluasi & submission | Belum mulai | `submission/model.pth` belum ada   |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: belum final. Rencana awal: baseline ResNet-18 pretrained ImageNet (dari starter kit), lalu dibandingkan dengan backbone lain.
- **Strategi pelatihan**: belum dijalankan. Rencana:
  - checkpoint terbaik dipilih berdasarkan **Macro-F1** pada split validasi lokal, bukan akurasi;
  - `label_smoothing` untuk meredam label noise;
  - penanganan ketidakseimbangan kelas dibandingkan secara terpisah: tanpa penanganan, _class-weighted loss_, dan `WeightedRandomSampler`.
- **Data tambahan**: tidak ada.
- **Ukuran input & preprocessing**:
  - input 224×224, normalisasi statistik ImageNet;
  - train: `RandomResizedCrop(scale 0,5–1,0)`, flip horizontal, `ColorJitter`;
  - validasi: `Resize(256)` + `CenterCrop(224)` (`get_transform()` di `submission/model.py`);
  - alternatif tanpa crop (`Resize` persegi) akan diuji karena center crop memotong gambar yang sangat lebar.

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
| Preview 1 (14 Okt) |        |                                 |                              |
| Preview 2 (21 Okt) |        |                                 |                              |
| Final (25 Okt)     |        |                                 |                              |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal): belum ada, pelatihan belum dimulai.

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
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik (belum ada)
evaluate.py         # salinan skrip evaluasi resmi
train_baseline.py   # baseline starter kit (ResNet-18)
requirements.txt    # dependensi lingkungan pelatihan
```

Folder `train/`, `my_train/`, `my_val/`, dan `data/` ada di `.gitignore` dan tidak ikut diunggah.
