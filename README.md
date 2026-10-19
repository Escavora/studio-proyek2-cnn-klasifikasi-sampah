<div align="center">

# Studio 2 — Klasifikasi Sampah · Aroma Karsa

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-ConvNeXt--Tiny-2E7D32)
![Macro-F1](https://img.shields.io/badge/Macro--F1%20lokal-0%2C9774-1565C0)

[Anggota](#anggota-kelompok) · [Status](#status-pengerjaan) · [Pendekatan](#ringkasan-pendekatan) · [Data](#temuan-data-ringkasan-eda) · [Hasil](#hasil) · [Reproduksi](#cara-reproduksi) · [Struktur](#struktur-repositori)

</div>

---

## Anggota Kelompok

<table>
  <thead>
    <tr>
      <th rowspan="2">No</th>
      <th rowspan="2">Nama Lengkap</th>
      <th rowspan="2">NIM</th>
      <th colspan="3">Kontribusi Utama</th>
    </tr>
    <tr>
      <th>Preview 1</th>
      <th>Preview 2</th>
      <th>Preview 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1</td>
      <td>Elfa Noviana Sari</td>
      <td align="center">123140066</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td>Mohd.Musyaffa Alief Athallah</td>
      <td align="center">123140184</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td>Revolusi Al Ghifari</td>
      <td align="center">123140199</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

## Status Pengerjaan

| Tahap                 | Status | Berkas                                                                                   |
| --------------------- | :----: | ---------------------------------------------------------------------------------------- |
| Data loading          | ✅ Selesai | `notebooks/01_data_loading.ipynb`                                                    |
| EDA                   | ✅ Selesai | `notebooks/02_eda.ipynb`                                                             |
| Preprocessing         | ✅ Selesai | `notebooks/03_preprocessing.ipynb`                                                   |
| Pelatihan model       | ✅ Selesai | `notebooks/04_train_baseline_resnet18.ipynb`, `notebooks/05_train_efficientnet_b0.ipynb`, `notebooks/07_train_convnext_tiny.ipynb` |
| Evaluasi & submission | ✅ Selesai | `submission/model.py`, `submission/model.pth`                                        |

---

## Ringkasan Pendekatan

| Aspek               | Pilihan                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| Arsitektur          | ConvNeXt-Tiny pretrained ImageNet-21k (timm), *full fine-tuning* bertahap |
| Pembanding          | ResNet-18, EfficientNet-B0, EfficientNetV2-S (pretrained ImageNet)      |
| Optimizer           | `AdamW` (LR backbone 1e-4, head 1e-3, weight decay 0,05) + warmup 1 epoch + cosine |
| Regularisasi        | augmentasi kuat, Mixup/CutMix, `label_smoothing` 0,1, EMA bobot          |
| Pemilihan checkpoint | Macro-F1 terbaik pada split validasi lokal (bobot biasa atau EMA)      |
| Input               | 224×224, normalisasi statistik ImageNet                                 |
| Data tambahan       | Tidak ada                                                               |

### Arsitektur / backbone

**ConvNeXt-Tiny** dari timm (`convnext_tiny.fb_in22k_ft_in1k`): dilatih di ImageNet-21k (±14 juta gambar, 21 ribu kelas) lalu di-fine-tune ke ImageNet-1k. Dipilih karena Macro-F1 validasi lokal tertinggi di antara backbone yang dibandingkan (lihat [Hasil](#hasil)). Fitur dari pretraining ImageNet-21k lebih umum, sehingga lebih cocok untuk dataset kecil seperti ini.

<details>
<summary><b>Detail arsitektur ConvNeXt-Tiny</b></summary>

<br>

- **Backbone (feature extractor)**: CNN modern yang mengadopsi desain Vision Transformer. *Stem* konvolusi 4×4 stride 4 (*patchify*), lalu 4 stage dengan 3-3-9-3 blok dan lebar 96-192-384-768 kanal. Tiap blok: depthwise conv 7×7 → LayerNorm → MLP *inverted bottleneck* (4×) dengan GELU → *layer scale*, plus koneksi residual.
- **Head (classifier)**: global average pooling → LayerNorm → `Linear(768 → 5)`, menggantikan layer 1.000 kelas ImageNet (`num_classes=5` di timm).
- **Ukuran**: 27,8 juta parameter. `model.pth` disimpan dalam fp16 (55,7 MB) supaya muat di GitHub; `load_state_dict` otomatis meng-cast ke fp32.
- **Kenapa cocok untuk GPU kecil**: ConvNeXt memakai LayerNorm, bukan BatchNorm, sehingga tetap stabil dengan batch kecil (16 per langkah, *grad accumulation* 2 → batch efektif 32 di GPU 4 GB).
- **Output**: logits `[batch, 5]`, urutan kelas alfabetis (0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam) — sama dengan urutan `ImageFolder`.
- **Kontrak submission**: `get_model()` di `submission/model.py` membangun arsitektur yang sama dengan `pretrained=False` (tanpa mengunduh bobot); bobot dimuat dari `model.pth` oleh `evaluate.py`. Paket `timm` dicantumkan di `submission/requirements.txt`.

</details>

### Strategi pelatihan

Resep fine-tuning di `notebooks/07_train_convnext_tiny.ipynb` (laptop RTX 3050 4 GB, ±19 menit):

1. **Fase head** (2 epoch): backbone dibekukan, hanya head baru yang dilatih (LR 1e-3), supaya head acak tidak merusak fitur pretrained.
2. **Fase penuh** (20 epoch): semua layer dilatih dengan LR berbeda (backbone 1e-4, head 1e-3), bias/norm tanpa weight decay, warmup linear 1 epoch lalu cosine per langkah.
3. **AMP** (fp16) + `channels_last` supaya cepat dan hemat memori GPU.
4. **EMA** bobot (decay 0,995) dievaluasi tiap epoch di samping bobot biasa.
5. Checkpoint terbaik dipilih berdasarkan **Macro-F1** pada split validasi lokal, bukan akurasi; `label_smoothing` 0,1 untuk meredam label noise.

Penanganan ketidakseimbangan kelas: belum ada penanganan khusus (rasio kelas terbesar/terkecil ±1,84× dianggap masih wajar). **Rencana selanjutnya**: membandingkan _class-weighted loss_ dan `WeightedRandomSampler`.

### Data tambahan

Tidak ada.

### Ukuran input & preprocessing

- input 224×224, normalisasi statistik ImageNet;
- train: `RandomResizedCrop(scale 0,5–1,0)`, flip horizontal, `ColorJitter`, ditambah `TrivialAugmentWide` dan `RandomErasing(p=0,25)` (`strong=True`) untuk mengurangi ketergantungan model pada teks/watermark katalog;
- Mixup/CutMix pada 50% batch saat training (`get_batch_mix()`);
- validasi: `Resize(256)` + `CenterCrop(224)` (`get_transform()` di `submission/model.py`).
- split dan kedua transform didefinisikan sekali di `preprocessing.py` (tahap preprocessing), lalu di-import oleh notebook training; augmentasi diterapkan *on-the-fly* oleh `DataLoader`, tidak disimpan ke disk.
- **Rencana selanjutnya**: menguji alternatif tanpa crop (`Resize` persegi) karena center crop memotong gambar yang sangat lebar.

---

## Temuan data (ringkasan EDA)

- Train set: **2.753 gambar**, 5 kelas: `botol_plastik` 523, `gelas_plastik` 521, `kertas_kardus` 365, `saset_kemasan` 673, `styrofoam` 671 (rasio kelas terbesar/terkecil ±1,84×).
- Resolusi cukup besar (median sisi pendek ±700–800 px); hanya 1,2% gambar berukuran < 224 px.
- Warna tidak membedakan kelas (klasifikasi dari fitur warna saja ±29%, hampir sama dengan tebakan acak).
- Banyak gambar katalog/iklan dengan teks dan watermark → ada risiko model memakai teks sebagai jalan pintas.
- 16 pasangan near-duplicate (dHash 256-bit), semuanya satu kelas. Split dibuat **per grup duplikat** supaya tidak ada kebocoran antara train dan validasi.
- Label noise sekitar **0,5%** (13 gambar jelas salah label, hasil cek visual dari kandidat k-nearest neighbor). Data dibiarkan apa adanya dan ditangani dengan `label_smoothing`.

### Split validasi lokal

15% per kelas (stratified, seed 42), menjaga grup duplikat tetap utuh: **2.343 gambar train** (`my_train/`) dan **410 gambar validasi** (`my_val/`). Validation set resmi disimpan dosen dan tidak pernah dilihat.

> [!NOTE]
> Tidak ada gambar yang dibuang saat preprocessing. Angka **2.753** adalah total dataset `train/`, sedangkan **2.343** hanya bagian train setelah split (2.343 + 410 = 2.753, dicek dengan `assert` di `03_preprocessing.ipynb`). Gambar near-duplicate juga tidak dihapus, hanya dikelompokkan supaya satu grup selalu berada di split yang sama.

| Kelas           | `train/` (asli) | `my_train/` | `my_val/` |
| --------------- | :-------------: | :---------: | :-------: |
| `botol_plastik` | 523             | 445         | 78        |
| `gelas_plastik` | 521             | 443         | 78        |
| `kertas_kardus` | 365             | 311         | 54        |
| `saset_kemasan` | 673             | 573         | 100       |
| `styrofoam`     | 671             | 571         | 100       |
| **Total**       | **2.753**       | **2.343**   | **410**   |

---

## Hasil

| Putaran            | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
| ------------------ | :----: | :-----------------------------: | :--------------------------: |
| Preview 1 (14 Okt) | [`7ca6be1`](https://github.com/Escavora/studio-proyek2-cnn-klasifikasi-sampah/commit/7ca6be18ac6ced32b6617ca35faa83811c58cdc8) | 0,9581 (EfficientNet-B0) |  |
| Preview 2 (21 Okt) |        | 0,9774 (ConvNeXt-Tiny)          |                              |
| Final (25 Okt)     |        |                                 |                              |

> [!TIP]
> Kolom **Commit** diisi hash commit yang dijadikan submission pada putaran tersebut (yang diambil asisten saat mengevaluasi).

Model yang di-submit: **ConvNeXt-Tiny** (Macro-F1 lokal **0,9774**, akurasi **0,9780**, `model.pth` 55,7 MB fp16). Evaluasi `evaluate.py` pada `my_val/` selesai dalam ±25 detik.

Perbandingan pada split validasi yang sama (`my_val/`, 410 gambar):

| Model                       | Resep fine-tuning | Macro-F1   | Akurasi    | Parameter |
| --------------------------- | ----------------- | :--------: | :--------: | :-------: |
| ResNet-18 (baseline)        | lama (notebook 04) | 0,9229    | 0,9244     | 11,2 juta |
| EfficientNet-B0             | lama (notebook 05) | 0,9581    | 0,9585     | 4,0 juta  |
| EfficientNet-B0             | baru               | 0,9597    | 0,9610     | 4,0 juta  |
| EfficientNetV2-S            | baru               | 0,9632    | 0,9634     | 20,2 juta |
| **ConvNeXt-Tiny** (IN-21k)  | baru (notebook 07) | **0,9774** | **0,9780** | 27,8 juta |

F1 per kelas (ConvNeXt-Tiny):

| `botol_plastik` | `gelas_plastik` | `kertas_kardus` | `saset_kemasan` | `styrofoam` |
| :-------------: | :-------------: | :-------------: | :-------------: | :---------: |
| 0,9873          | 0,9873          | 0,9636          | 0,9583          | 0,9901      |

**Catatan eksperimen singkat** (apa yang dicoba, apa yang berhasil/gagal):

- **ResNet-18 → EfficientNet-B0** menaikkan Macro-F1 lokal dari **0,9229** ke **0,9581** (+0,0352) dengan parameter jauh lebih sedikit; dipakai sebagai submission Preview 1.
- **Resep fine-tuning baru** (head dulu, LR berbeda backbone/head, warmup + cosine, AMP, EMA, augmentasi kuat + Mixup/CutMix) hanya menaikkan EfficientNet-B0 sedikit (0,9581 → 0,9597), masih dalam rentang variasi acak antar-run (±0,004).
- **Ganti backbone ke ConvNeXt-Tiny pretrained ImageNet-21k** memberi lompatan terbesar (**+0,019**). EfficientNetV2-S (ImageNet-1k) hanya +0,005 dan 3× lebih lambat dilatih.
- Run ulang ConvNeXt-Tiny dengan notebook 07 menghasilkan 0,9765 (checkpoint EMA), selisih < 1 gambar dari submission (0,9774), jadi resepnya stabil. Skor 0,9774 dipilih dari epoch terbaik di `my_val`, sehingga sedikit optimis; perkiraan realistis di validation set tersembunyi ±0,97.
- TTA flip horizontal tidak menaikkan skor (tetap 0,9774), jadi tidak dipakai.
- Kesalahan tersisa paling banyak di `saset_kemasan` (tertukar dengan kelas lain); `styrofoam`, `botol_plastik`, dan `gelas_plastik` hampir sempurna.

---

## Cara Reproduksi

> [!IMPORTANT]
> Dataset **tidak disertakan** di repositori (sesuai ketentuan tugas). Unduh dari tautan di halaman kompetisi, lalu ekstrak sehingga ada `train/<nama_kelas>/*.jpg` di root repositori.

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# jalankan berurutan: notebooks/01_data_loading.ipynb -> 02_eda.ipynb -> 03_preprocessing.ipynb
# (03 membuat my_train/ dan my_val/)
# lalu notebooks/07_train_convnext_tiny.ipynb (GPU CUDA, ±20 menit) untuk melatih ulang model submission

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
  05_train_efficientnet_b0.ipynb      # EfficientNet-B0 (submission Preview 1)
  07_train_convnext_tiny.ipynb        # model final (ConvNeXt-Tiny)
submission/
  model.py          # get_model() dan get_transform() (ConvNeXt-Tiny, timm)
  model.pth         # state_dict model terbaik (fp16, 55,7 MB)
  requirements.txt  # timm
preprocessing.py    # split per grup duplikat, transform, Mixup/CutMix; dipakai notebook 03-07
evaluate.py         # salinan skrip evaluasi resmi
train_baseline.py   # baseline starter kit (ResNet-18)
requirements.txt    # dependensi lingkungan pelatihan
```

Folder `train/`, `my_train/`, `my_val/`, dan `data/` ada di `.gitignore` dan tidak ikut diunggah.
