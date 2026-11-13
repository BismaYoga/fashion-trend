# Fashion Trend: YOLOv8 Cropping + VGG16 Classification + K-Means Color Analysis

Repositori ini berisi rancangan arsitektur dan notebook *end-to-end* untuk memprediksi dan menganalisis tren item fesyen dari media sosial menggunakan kombinasi **YOLOv8 (Object Detection & Cropping)**, **VGG-16 (Transfer Learning Classification)**, dan **K-Means Clustering (Dominant Color Extraction)**.

Proyek ini dirancang sebagai implementasi modul analisis visual pada penelitian **Sistem Pakar Prediksi Tren Fashion dari Media Sosial (Universitas Udayana)**.

---

## Alur Pipeline End-to-End (Sesuai Notebook)

Setiap notebook (`train_vgg16_baju.ipynb` & `train_vgg16_tas.ipynb`) menjalankan 5 tahapan terintegrasi:

1. **Deteksi & Auto-Cropping Dataset dengan YOLOv8 (`yolov8n.pt`)**
   - Setiap citra pada dataset mentah (`dataset/baju` atau `dataset/tas`) dideteksi menggunakan **YOLOv8**.
   - Untuk **Tas**: YOLOv8 memotong area objek tas (`handbag`, `backpack`, `suitcase`) atau area pembawa tas pada objek `person`.
   - Untuk **Baju**: YOLOv8 mendeteksi `person` dan memotong fokus area pakaian/torso sehingga latar belakang jalanan/gedung tereliminasi.
   - Seluruh hasil potongan di-*resize* ke ukuran standar input VGG16 (**`224x224` piksel**) dan disimpan ke folder `dataset_cropped/`.

2. **Pembagian Data & Augmentasi**
   - Dataset hasil crop YOLOv8 (`dataset_cropped/`) dibagi secara seimbang menjadi **70% Training**, **15% Validation**, dan **15% Testing**.

3. **Training Model Transfer Learning VGG16**
   - Menggunakan *pre-trained* **VGG16 (ImageNet weights)** dengan *fine-tuning* pada blok konvolusi akhir (`features[24:]`) dan *Custom Classifier Head*.
   - Menyimpan model terbaik secara otomatis ke `best_vgg16_baju.pth` dan `best_vgg16_tas.pth`.
   - Menampilkan kurva *Loss & Accuracy*, *Confusion Matrix*, serta *Classification Report* (*Precision, Recall, F1-Score*).

4. **Uji Coba Model (Inference Pipeline: YOLOv8 Crop -> VGG16 Predict -> K-Means Color)**
   - Saat menguji foto baru, alur yang dijalankan sama persis:
     1. Foto dideteksi dan di-*crop* menggunakan **YOLOv8** lalu diubah ukurannya menjadi **`224x224`** sesuai input model.
     2. Potongan gambar diprediksi kelasnya oleh model **VGG16** yang telah dilatih beserta nilai *Confidence Score* (%).
     3. Potongan objek dianalisis komposisi warna dominannya menggunakan algoritma **K-Means Clustering (`sklearn.cluster.KMeans`)** untuk menghasilkan palet warna dominan, kode **HEX (`#RRGGBB`)**, nama warna fesyen terdekat, serta persentase proporsinya.

---

## Model & Distribusi Kelas Dataset

### 1. Model Tas (`tas`) — Total: 380 Gambar (Distribusi Seimbang)
- `tote_bag` (**127 gambar**): Tas jinjing santai berukuran sedang-besar
- `handbag` (**127 gambar**): Tas jinjing / tas tangan wanita
- `sling_bag` (**126 gambar**): Tas selempang / shoulder bag / crossbody

### 2. Model Baju (`baju`) — Total: 247 Gambar (Distribusi Seimbang)
- `casual_top` (**83 gambar**): Kaos (T-shirt), kemeja, blouse santai, crop top, streetwear
- `outerwear` (**82 gambar**): Jaket, blazer, mantel, cardigan, sweater, hoodie
- `dress` (**82 gambar**): Gaun wanita, maxi dress, skirt set, gamis pesta

---

## Struktur Repositori

```
fashion-trend/
├── train_vgg16_baju.ipynb      # Notebook lengkap YOLOv8 Crop + Train VGG16 + Uji Coba & K-Means (Baju)
├── train_vgg16_tas.ipynb       # Notebook lengkap YOLOv8 Crop + Train VGG16 + Uji Coba & K-Means (Tas)
├── prepare_dataset.py          # Script otomatis unduh & ekstrak dataset
├── labels_manifest.csv         # Data manifest label seluruh 627 foto (CSV)
├── labels_manifest.json        # Data manifest label seluruh 627 foto (JSON)
├── requirements.txt            # Dependensi PyTorch, Ultralytics YOLOv8, OpenCV, Scikit-learn
└── README.md                   # Dokumentasi lengkap penggunaan
```

---

## Cara Menjalankan di Lokal / Google Colab

```bash
git clone https://github.com/BismaYoga/fashion-trend.git
cd fashion-trend
pip install -r requirements.txt
jupyter notebook
```
Buka `train_vgg16_baju.ipynb` atau `train_vgg16_tas.ipynb`, lalu jalankan seluruh cell dari atas ke bawah (*Run All*). Pada bagian akhir notebook (Bagian 5), tersedia cell khusus `test_fashion_pipeline(...)` untuk menguji foto baru lengkap dengan deteksi YOLOv8, prediksi VGG16, dan ekstraksi warna K-Means.
