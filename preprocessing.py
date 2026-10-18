import random
import shutil

import numpy as np
from PIL import Image
from torchvision import transforms

from submission.model import get_transform  # transform validasi resmi (kontrak evaluate.py)

SEED = 42
VAL_FRAC = 0.15
MAX_DIST = 30  # batas jarak dHash untuk near-duplicate (sama dengan EDA)

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # statistik ImageNet (backbone pretrained)


# ---------- Split validasi lokal ----------

def dhash(path, size=16):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def find_duplicate_groups(files, max_dist=MAX_DIST):
    # Kembalikan (group_id per gambar, daftar pasangan near-duplicate).
    hashes = np.stack([dhash(p) for p in files])
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    dist = np.zeros((len(files), len(files)), dtype=np.uint16)
    for b in range(hashes.shape[1]):
        dist += popcount[hashes[:, b][:, None] ^ hashes[:, b][None, :]]
    pairs = list(zip(*np.where(np.triu(dist <= max_dist, k=1))))

    # Union-find: setiap gambar awalnya grup sendiri, lalu gabungkan tiap pasangan duplikat
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(files))]), pairs


def split_by_group(labels, group_id, val_frac=VAL_FRAC, seed=SEED):
    # Per kelas, acak urutan grup lalu isi val sampai ±val_frac; satu grup selalu utuh.
    # Kembalikan dict index gambar -> "train" / "val".
    rng = random.Random(seed)
    split_of = {}
    for c in np.unique(labels):  # urut alfabetis, sama dengan CLASSES
        idx_c = np.where(labels == c)[0]
        groups = sorted(set(group_id[idx_c]))
        rng.shuffle(groups)
        target, n_val = int(len(idx_c) * val_frac), 0
        for g in groups:
            members = idx_c[group_id[idx_c] == g]
            to_val = n_val < target
            n_val += len(members) if to_val else 0
            for i in members:
                split_of[i] = "val" if to_val else "train"
    return split_of


def write_split(files, labels, split_of, train_dir, val_dir):
    # Hapus folder split lama lalu salin ulang ke <split>/<kelas>/ (format ImageFolder).
    for d in (train_dir, val_dir):
        if d.exists():
            shutil.rmtree(d)
    for i, p in enumerate(files):
        out = (val_dir if split_of[i] == "val" else train_dir) / labels[i]
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out / p.name)


# ---------- Transform ----------

def get_train_transform():
    # Augmentasi hasil 03_preprocessing.ipynb bagian 4 (Transform).
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_val_transform():
    # Sama persis dengan get_transform() di submission/model.py.
    return get_transform()
