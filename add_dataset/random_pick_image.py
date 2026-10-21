import os
import random
import shutil

def process_random_images(source_dir, target_dir, max_limit=362, mode='copy'):
    """
    source_dir : Folder asal yang berisi gambar
    target_dir : Folder baru tujuan
    max_limit  : Jumlah maksimum gambar yang diambil (default: 308)
    mode       : 'copy' untuk menyalin, 'move' untuk memindahkan
    """
    # 1. Ekstensi gambar yang didukung
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    
    # 2. Cek apakah folder asal ada
    if not os.path.exists(source_dir):
        print(f"Error: Folder '{source_dir}' tidak ditemukan!")
        return

    # 3. Filter hanya file gambar
    all_files = os.listdir(source_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    total_images = len(image_files)
    print(f"Total gambar ditemukan di folder asal: {total_images}")
    
    if total_images == 0:
        print("Tidak ada gambar yang cocok ditemukan.")
        return

    # 4. Ambil sampel acak tanpa duplikasi (maksimal 308)
    sample_size = min(total_images, max_limit)
    selected_images = random.sample(image_files, sample_size)

    # 5. Buat folder baru jika belum ada
    os.makedirs(target_dir, exist_ok=True)

    # 6. Eksekusi salin atau pindah
    for img in selected_images:
        src_path = os.path.join(source_dir, img)
        dest_path = os.path.join(target_dir, img)
        
        if mode == 'move':
            shutil.move(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)

    action_text = "dipindahkan" if mode == 'move' else "disalin"
    print(f"Selesai! {len(selected_images)} gambar berhasil {action_text} ke folder '{target_dir}'.")

# ==========================================
# Cara Penggunaan:
# ==========================================
folder_asal = 'DATASET_SAMPAH/Plastik'        # Ganti dengan path folder asal kamu
folder_tujuan = 'output_random/saset_kemasan' # Nama folder baru yang akan dibuat

# Opsi A: Menyalin gambar (direkomendasikan)
process_random_images(folder_asal, folder_tujuan, max_limit=69, mode='copy')

# process_random_images(folder_asal, folder_tujuan, max_limit=308, mode='move')