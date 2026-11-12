#!/usr/bin/env python3
"""
Helper script untuk mengunduh dan mengekstrak dataset fashion terlabeli
secara otomatis dari server ke folder lokal 'dataset/'.
"""

import os
import sys
import zipfile
import urllib.request

DATASET_URL = "http://202.10.47.34/slides/dataset_fashion_labeled.zip"
TARGET_ZIP = "dataset_fashion_labeled.zip"
EXTRACT_DIR = "dataset"

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = f"\rMengunduh dataset: {percent:5.1f}% ({readsofar // (1024*1024):d} MB / {totalsize // (1024*1024):d} MB)"
        sys.stderr.write(s)
        if readsofar >= totalsize:
            sys.stderr.write("\n")
    else:
        sys.stderr.write(f"\rMengunduh: {readsofar // (1024*1024):d} MB")

def main():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)

    if not os.path.exists(TARGET_ZIP):
        print(f"Mengunduh dataset dari: {DATASET_URL}...")
        try:
            urllib.request.urlretrieve(DATASET_URL, TARGET_ZIP, reporthook)
            print("Unduhan selesai!")
        except Exception as e:
            print(f"Gagal mengunduh otomatis: {e}")
            print(f"Silakan unduh manual dari browser di: {DATASET_URL} dan simpan di folder ini dengan nama '{TARGET_ZIP}'.")
            return

    print(f"Mengekstrak {TARGET_ZIP} ke folder '{EXTRACT_DIR}'...")
    with zipfile.ZipFile(TARGET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print("\nEkstraksi selesai! Struktur dataset yang tersedia:")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        level = root.replace(EXTRACT_DIR, '').count(os.sep)
        indent = ' ' * 4 * level
        folder = os.path.basename(root) or EXTRACT_DIR
        num_files = len([f for f in files if f.endswith('.jpg')])
        info = f" ({num_files} foto)" if num_files > 0 else ""
        print(f"{indent}📁 {folder}/{info}")

if __name__ == "__main__":
    main()
