# Fashion Trend: YOLOv8 Cropping + Dual VGG16 (Baju & Tas) + K-Means Color + Streamlit

Repositori ini berisi rancangan arsitektur, **1 Master Notebook terpadu (`fashion_trend_pipeline.ipynb`)**, dan **Aplikasi Pengujian Web Interaktif Streamlit (`app.py`)** untuk memprediksi serta menganalisis tren item fesyen dari media sosial menggunakan kombinasi:
- **YOLOv8 (`yolov8n.pt`)**: Deteksi objek fesyen dan *auto-cropping* sesuai ukuran input model (`224x224`).
- **Dual VGG-16 (`best_vgg16_baju.pth` & `best_vgg16_tas.pth`)**: Klasifikasi *Transfer Learning* untuk kedua model (**Baju** dan **Tas**) dalam satu *pipeline*.
- **K-Means Clustering (`sklearn.cluster.KMeans`)**: Ekstraksi palet warna dominan, kode HEX, nama warna fesyen, dan persentase proporsi warna.
- **Streamlit (`app.py`)**: Antarmuka web untuk menguji kedua model sekaligus menggunakan foto baru.

Proyek ini dirancang sebagai implementasi modul analisis visual pada penelitian **Sistem Pakar Prediksi Tren Fashion dari Media Sosial (Universitas Udayana)**.

---

## Struktur Repositori

```
fashion-trend/
├── fashion_trend_pipeline.ipynb  # Master Notebook Terpadu (YOLOv8 Crop + Train Baju & Tas + Uji K-Means + Streamlit)
├── app.py                        # Aplikasi Web Streamlit untuk menguji kedua model (Baju & Tas) sekaligus
├── prepare_dataset.py            # Script otomatis unduh & ekstrak dataset
├── labels_manifest.csv           # Data manifest label seluruh 627 foto (CSV)
├── labels_manifest.json          # Data manifest label seluruh 627 foto (JSON)
├── requirements.txt              # Dependensi PyTorch, Ultralytics YOLOv8, OpenCV, Scikit-learn, Streamlit
└── README.md                     # Dokumentasi lengkap penggunaan
```

---

## Model & Distribusi Kelas Dataset

### 1. Model Baju (`baju`) — Total: 247 Gambar (Distribusi Seimbang)
- `casual_top` (**83 gambar**): Kaos (T-shirt), kemeja, blouse santai, crop top, streetwear
- `outerwear` (**82 gambar**): Jaket, blazer, mantel, cardigan, sweater, hoodie
- `dress` (**82 gambar**): Gaun wanita, maxi dress, skirt set, gamis pesta

### 2. Model Tas (`tas`) — Total: 380 Gambar (Distribusi Seimbang)
- `tote_bag` (**127 gambar**): Tas jinjing santai berukuran sedang-besar
- `handbag` (**127 gambar**): Tas jinjing / tas tangan wanita
- `sling_bag` (**126 gambar**): Tas selempang / shoulder bag / crossbody

---

## Cara Menjalankan Notebook & Aplikasi Streamlit

### 1. Kloning & Instalasi Dependensi
```bash
git clone https://github.com/BismaYoga/fashion-trend.git
cd fashion-trend
pip install -r requirements.txt
```

### 2. Jalankan Master Notebook (`fashion_trend_pipeline.ipynb`)
```bash
jupyter notebook fashion_trend_pipeline.ipynb
```
Jalankan seluruh cell dari atas ke bawah (*Run All*). Notebook ini akan:
1. Mengunduh dataset dan melakukan **Auto-Cropping YOLOv8 (`224x224`)** untuk kategori **Baju** dan **Tas**.
2. Melatih model **VGG16 Baju** (`best_vgg16_baju.pth`) dan **VGG16 Tas** (`best_vgg16_tas.pth`) secara berurutan lengkap dengan kurva *Loss/Accuracy*, *Confusion Matrix*, dan *Classification Report*.
3. Menguji kedua model dengan alur **YOLOv8 Crop $\rightarrow$ Prediksi VGG16 $\rightarrow$ Analisis Warna K-Means**.

### 3. Jalankan Pengujian Interaktif dengan Streamlit (`app.py`)
Setelah bobot model terbentuk (atau langsung dari terminal), jalankan:
```bash
streamlit run app.py
```
Melalui dashboard Streamlit, Anda dapat mengunggah foto dan memilih mode pengujian **Kedua Model Sekaligus (Baju & Tas)**, **Hanya Model Baju**, atau **Hanya Model Tas**.
