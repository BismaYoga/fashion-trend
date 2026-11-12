# Fashion Trend VGG16 Classification

Repositori ini berisi rancangan arsitektur dan notebook pelatihan **Transfer Learning menggunakan VGG-16** untuk memprediksi dan mengklasifikasikan tren item fesyen dari media sosial.

Proyek ini dirancang sebagai implementasi modul klasifikasi citra pada penelitian **Sistem Pakar Prediksi Tren Fashion dari Media Sosial (Universitas Udayana)**.

---

## 🎯 Model & Label Klasifikasi

Terdapat **2 Model VGG16 Independen**, masing-masing memiliki 3 kelas target:

### 1. Model Tas (`tas`)
- `handbag`: Tas jinjing / tas tangan wanita
- `sling_bag`: Tas selempang / shoulder bag / crossbody
- `tote_bag`: Tas jinjing santai berukuran sedang-besar

### 2. Model Baju (`baju`)
- `casual_top`: Kaos (T-shirt), kemeja, blouse santai, crop top, streetwear
- `outerwear`: Jaket, blazer, mantel, cardigan, sweater, hoodie
- `dress`: Gaun wanita, maxi dress, skirt set, gamis pesta

---

## 📂 Struktur Repositori

```
fashion-trend/
├── train_vgg16_tas.ipynb       # Notebook pelatihan model Tas (3 kelas)
├── train_vgg16_baju.ipynb      # Notebook pelatihan model Baju (3 kelas)
├── prepare_dataset.py          # Script otomatis unduh & ekstrak dataset
├── labels_manifest.csv         # Data manifest label seluruh 627 foto (CSV)
├── labels_manifest.json        # Data manifest label seluruh 627 foto (JSON)
├── requirements.txt            # Dependensi PyTorch, Torchvision, Scikit-learn
└── README.md                   # Petunjuk lengkap penggunaan
```

---

## 🚀 Panduan Menjalankan di Komputer Lokal / Google Colab

### 1. Kloning Repositori
```bash
git clone https://github.com/BismaYoga/fashion-trend.git
cd fashion-trend
```

### 2. Pasang Dependensi
Pastikan lingkungan Python Anda (disarankan menggunakan virtualenv atau Conda) telah terpasang:
```bash
pip install -r requirements.txt
```

> **Catatan Pengguna GPU (NVIDIA):**  
> Jika memiliki GPU NVIDIA, pastikan PyTorch terpasang dengan dukungan CUDA untuk performa pelatihan yang jauh lebih cepat (Notebook mendeteksi CUDA secara otomatis).

### 3. Unduh & Siapkan Dataset
Jalankan script helper untuk menarik dataset gambar yang sudah terorganisir per folder kelas:
```bash
python prepare_dataset.py
```
*(Atau jika script di atas dijalankan di dalam notebook, cell ke-3 akan mengunduh dan mengekstrak dataset secara otomatis).*

Dataset terlabeli juga dapat diunduh manual melalui portal berikut:
- **Direct Download ZIP:** [http://202.10.47.34/slides/dataset_fashion_labeled.zip](http://202.10.47.34/slides/dataset_fashion_labeled.zip) *(221 MB)*
- **Halaman Portal:** [http://202.10.47.34/slides/download.html](http://202.10.47.34/slides/download.html)

### 4. Jalankan Pelatihan Model
Buka Jupyter Notebook atau VS Code:
```bash
jupyter notebook
```
- Buka `train_vgg16_tas.ipynb` untuk melatih model klasifikasi tas.
- Buka `train_vgg16_baju.ipynb` untuk melatih model klasifikasi baju.
- Jalankan seluruh cell dari atas ke bawah (*Run All*).

---

## 🧠 Arsitektur Transfer Learning VGG-16

1. **Backbone:** Pre-trained VGG-16 (ImageNet weights) dengan layer konvolusi yang dibekukan (*frozen feature extraction*) untuk mempertahankan representasi visual dasar.
2. **Classifier Head:**
   - `Linear(25088 -> 1024)` + `BatchNorm1d` + `ReLU` + `Dropout(0.5)`
   - `Linear(1024 -> 512)` + `BatchNorm1d` + `ReLU` + `Dropout(0.4)`
   - `Linear(512 -> 3)` (Output probabilitas untuk 3 kelas target)
3. **Optimasi & Regularisasi:**
   - Loss Function: `CrossEntropyLoss`
   - Optimizer: `AdamW` (learning rate: `1e-4`, weight decay: `1e-3`)
   - Scheduler: `ReduceLROnPlateau` (otomatis menurunkan learning rate jika validation loss stagnan)
   - Checkpoint: Menyimpan bobot terbaik (`best_vgg16_tas.pth` / `best_vgg16_baju.pth`) berdasarkan performa validation accuracy tertinggi.

---

## 📊 Metrik Evaluasi

Sesuai dengan perancangan pada **Bab 3.4 Proposal Final Project**, notebook menghasilkan:
- **Akurasi Global (Overall Accuracy)**
- **Classification Report per Kelas:** Precision, Recall, dan F1-Score
- **Confusion Matrix Heatmap:** Untuk menganalisis distribusi kesalahan prediksi antarkelas secara visual
- **Visualisasi Inferensi:** Uji coba prediksi pada citra baru lengkap dengan diagram tingkat kepercayaan (*confidence score*).
