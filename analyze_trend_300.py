import os
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from ultralytics import YOLO
from sklearn.cluster import KMeans

# =====================================================================
# 1. KONFIGURASI PATH & PARAMETER
# =====================================================================
IMAGE_FOLDER = r"C:\Users\Bisma\Downloads\dataset_fashion_instagram_300\images"
MAX_IMAGES = 300  # Jumlah maksimal gambar yang dianalisis (set None jika ingin semua)

MODEL_PATHS = {
    "baju": "best_vgg16_baju.pth",
    "tas": "best_vgg16_tas.pth"
}

DEFAULT_CLASSES = {
    "baju": ["casual_top", "dress", "outerwear"],
    "tas": ["handbag", "sling_bag", "tote_bag"]
}

INPUT_SIZE = (224, 224)
SEED = 42

FASHION_COLOR_PALETTE = {
    "Hitam (Black)": (25, 25, 25),
    "Putih (White)": (245, 245, 245),
    "Abu-Abu Tua (Charcoal)": (75, 75, 80),
    "Abu-Abu Muda (Light Grey)": (185, 185, 190),
    "Krem / Beige": (225, 208, 185),
    "Putih Gading (Ivory)": (248, 242, 228),
    "Cokelat Muda (Tan / Camel)": (193, 154, 107),
    "Cokelat Tua (Dark Brown)": (92, 64, 51),
    "Cokelat Kopi (Espresso)": (58, 40, 32),
    "Biru Dongker (Navy Blue)": (28, 46, 74),
    "Biru Denim (Denim Blue)": (70, 105, 145),
    "Biru Muda (Sky / Baby Blue)": (155, 195, 225),
    "Merah Marun (Burgundy)": (115, 28, 38),
    "Merah Menyala (Crimson Red)": (195, 35, 45),
    "Merah Muda (Dusty Pink)": (215, 155, 165),
    "Merah Muda Cerah (Blush Pink)": (242, 195, 205),
    "Hijau Sage (Sage Green)": (158, 175, 152),
    "Hijau Olive (Olive Green)": (105, 115, 65),
    "Hijau Zamrud (Emerald)": (25, 110, 75),
    "Kuning Mustard": (215, 165, 45),
    "Oranye Terracotta": (195, 95, 65),
    "Ungu Lavender / Lilac": (185, 165, 205)
}

COLOR_HEX_MAP = {
    name: "#{:02X}{:02X}{:02X}".format(*rgb)
    for name, rgb in FASHION_COLOR_PALETTE.items()
}

val_test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# 2. INISIALISASI PERANGKAT & MODEL (YOLOv8 + DUAL VGG16)
# =====================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_vgg16_classifier(num_classes):
    try:
        model = vgg16(weights=VGG16_Weights.DEFAULT)
    except Exception:
        model = vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )
    return model

def load_vgg_model(category, device):
    ckpt_path = MODEL_PATHS[category]
    classes = list(DEFAULT_CLASSES[category])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"File model '{ckpt_path}' tidak ditemukan! "
            f"Pastikan '{ckpt_path}' berada di folder yang sama dengan script ini."
        )

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        classes = ckpt.get("class_names", classes)
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model = build_vgg16_classifier(len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes

# =====================================================================
# 3. FUNGSI DETEKSI YOLOv8, PREDIKSI VGG16, DAN WARNA K-MEANS
# =====================================================================
BAG_COCO_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
PERSON_COCO_CLASS = 0            # person

def analyze_single_image(img_path, yolo_model, model_baju, classes_baju, model_tas, classes_tas, device):
    image_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(image_pil)
    h, w, _ = img_rgb.shape

    results = yolo_model.predict(source=img_rgb, conf=0.20, verbose=False)
    boxes = results[0].boxes

    has_bag_box = False
    has_person_box = False
    bag_box = None
    apparel_box = None

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        # Cek objek Tas
        bag_indices = [i for i, c in enumerate(clss) if c in BAG_COCO_CLASSES]
        if len(bag_indices) > 0:
            has_bag_box = True
            best_idx = max(bag_indices, key=lambda i: confs[i] * ((xyxy[i][2]-xyxy[i][0])*(xyxy[i][3]-xyxy[i][1])))
            bag_box = tuple(xyxy[best_idx])

        # Cek objek Person (untuk Baju)
        person_indices = [i for i, c in enumerate(clss) if c == PERSON_COCO_CLASS]
        if len(person_indices) > 0:
            has_person_box = True
            best_idx = max(person_indices, key=lambda i: (xyxy[i][2]-xyxy[i][0])*(xyxy[i][3]-xyxy[i][1]))
            px1, py1, px2, py2 = xyxy[best_idx]
            pw, ph = (px2 - px1), (py2 - py1)
            apparel_box = (
                max(0, px1 - 0.03 * pw),
                max(0, py1 + 0.12 * ph),
                min(w, px2 + 0.03 * pw),
                min(h, py1 + 0.92 * ph)
            )
            if bag_box is None:
                bag_box = (
                    max(0, px1 - 0.05 * pw),
                    max(0, py1 + 0.18 * ph),
                    min(w, px2 + 0.05 * pw),
                    min(h, py1 + 0.90 * ph)
                )

    # Fallback Center Crop jika tidak ada deteksi spesifik
    mx, my = int(0.075 * w), int(0.075 * h)
    center_box = (mx, my, w - mx, h - my)
    if apparel_box is None:
        apparel_box = center_box
    if bag_box is None:
        bag_box = center_box

    def crop_and_resize(box):
        x1, y1, x2, y2 = [int(v) for v in box]
        if (x2 - x1) < 32 or (y2 - y1) < 32:
            x1, y1, x2, y2 = 0, 0, w, h
        cropped = image_pil.crop((x1, y1, x2, y2)).convert("RGB")
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        return cropped.resize(INPUT_SIZE, resample)

    crop_baju = crop_and_resize(apparel_box)
    crop_tas = crop_and_resize(bag_box)

    # Prediksi VGG16 Baju & Tas
    with torch.no_grad():
        tensor_baju = val_test_transforms(crop_baju).unsqueeze(0).to(device)
        probs_baju = torch.softmax(model_baju(tensor_baju), dim=1)[0].cpu().numpy()
        idx_baju = int(np.argmax(probs_baju))
        pred_baju = classes_baju[idx_baju]
        conf_baju = float(probs_baju[idx_baju]) * 100.0

        tensor_tas = val_test_transforms(crop_tas).unsqueeze(0).to(device)
        probs_tas = torch.softmax(model_tas(tensor_tas), dim=1)[0].cpu().numpy()
        idx_tas = int(np.argmax(probs_tas))
        pred_tas = classes_tas[idx_tas]
        conf_tas = float(probs_tas[idx_tas]) * 100.0

    # Tentukan kategori utama gambar ini (apakah fokus Tas atau Baju)
    if has_bag_box and not has_person_box:
        primary_category = "tas"
    elif has_person_box and not has_bag_box:
        primary_category = "baju" if conf_baju >= (conf_tas - 10.0) else "tas"
    else:
        primary_category = "baju" if conf_baju >= conf_tas else "tas"

    active_crop = crop_baju if primary_category == "baju" else crop_tas
    primary_sub_item = pred_baju if primary_category == "baju" else pred_tas
    primary_conf = conf_baju if primary_category == "baju" else conf_tas

    # Ekstraksi Warna Dominan dengan K-Means (Top 2 warna per gambar)
    dominant_colors = extract_kmeans_colors(active_crop, n_clusters=3)
    top1_color = dominant_colors[0]
    top2_color = dominant_colors[1] if len(dominant_colors) > 1 else dominant_colors[0]

    return {
        "filename": os.path.basename(img_path),
        "primary_category": primary_category,
        "primary_item": primary_sub_item,
        "primary_confidence": round(primary_conf, 2),
        "pred_baju": pred_baju,
        "conf_baju": round(conf_baju, 2),
        "pred_tas": pred_tas,
        "conf_tas": round(conf_tas, 2),
        "dominant_color_1": top1_color["name"],
        "dominant_hex_1": top1_color["hex"],
        "dominant_pct_1": round(top1_color["percentage"], 1),
        "dominant_color_2": top2_color["name"],
        "dominant_hex_2": top2_color["hex"],
        "dominant_pct_2": round(top2_color["percentage"], 1),
    }

def get_closest_fashion_color_name(rgb_tuple):
    r, g, b = rgb_tuple
    return min(
        FASHION_COLOR_PALETTE.keys(),
        key=lambda name: (r - FASHION_COLOR_PALETTE[name][0]) ** 2
        + (g - FASHION_COLOR_PALETTE[name][1]) ** 2
        + (b - FASHION_COLOR_PALETTE[name][2]) ** 2
    )

def extract_kmeans_colors(cropped_pil, n_clusters=3):
    img_np = np.array(cropped_pil.convert("RGB"))
    h, w, _ = img_np.shape
    my, mx = int(0.125 * h), int(0.125 * w)
    core = img_np[my:h - my, mx:w - mx]
    pixels = core.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    total = len(labels)

    res = []
    for idx in np.argsort(-counts):
        rgb = tuple(int(v) for v in np.clip(centroids[idx], 0, 255))
        res.append({
            "rgb": rgb,
            "hex": "#{:02X}{:02X}{:02X}".format(*rgb),
            "percentage": (counts[idx] / total) * 100.0,
            "name": get_closest_fashion_color_name(rgb)
        })
    return res

# =====================================================================
# 4. EKSEKUSI ANALISIS BATCH & PEMBUATAN INSIGHT TREN
# =====================================================================
def main():
    print("=" * 75)
    print("SISTEM ANALISIS TREN FASHION INSTAGRAM (YOLOv8 + VGG16 + K-MEANS)")
    print("=" * 75)

    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan di: {IMAGE_FOLDER}\n"
            f"Pastikan path IMAGE_FOLDER sudah sesuai dengan lokasi di laptop Anda."
        )

    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    all_files = sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(valid_exts)
    ])

    if MAX_IMAGES is not None:
        all_files = all_files[:MAX_IMAGES]

    if len(all_files) == 0:
        raise RuntimeError(f"Tidak ada file gambar (.jpg/.png) di dalam {IMAGE_FOLDER}")

    device = get_device()
    print(f"Perangkat Komputasi : {device}")
    print(f"Lokasi Dataset      : {IMAGE_FOLDER}")
    print(f"Jumlah Gambar Uji   : {len(all_files)} gambar\n")

    print("Memuat model YOLOv8 & Dual VGG16 (Baju & Tas)...")
    yolo_model = YOLO("yolov8n.pt")
    model_baju, classes_baju = load_vgg_model("baju", device)
    model_tas, classes_tas = load_vgg_model("tas", device)
    print("Seluruh model berhasil dimuat!\n")

    records = []
    for img_path in tqdm(all_files, desc="Menganalisis Gambar"):
        try:
            rec = analyze_single_image(
                img_path, yolo_model, model_baju, classes_baju, model_tas, classes_tas, device
            )
            records.append(rec)
        except Exception as e:
            print(f"Gagal memproses {os.path.basename(img_path)}: {e}")

    df = pd.DataFrame(records)
    csv_out = "hasil_analisis_300_gambar.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------
    # PERHITUNGAN STATISTIK & INSIGHT TREN
    # -----------------------------------------------------------------
    total_analyzed = len(df)
    baju_counts = df["pred_baju"].value_counts()
    tas_counts = df["pred_tas"].value_counts()
    primary_item_counts = df["primary_item"].value_counts()

    # Warna dominan gabungan (Top 1 & Top 2 berbobot)
    color_counter = Counter()
    for _, row in df.iterrows():
        color_counter[row["dominant_color_1"]] += 1.0
        color_counter[row["dominant_color_2"]] += 0.5

    top_colors = color_counter.most_common(10)
    total_color_weight = sum(color_counter.values())

    # Kombinasi Item + Warna Dominan 1
    combo_counter = Counter(
        f"{row['primary_item']} + {row['dominant_color_1']}"
        for _, row in df.iterrows()
    )
    top_combos = combo_counter.most_common(8)

    # Warna terpopuler per sub-kategori
    color_by_baju = {}
    for cls_b in classes_baju:
        sub = df[df["pred_baju"] == cls_b]
        if len(sub) > 0:
            color_by_baju[cls_b] = sub["dominant_color_1"].value_counts().index[0]

    color_by_tas = {}
    for cls_t in classes_tas:
        sub = df[df["pred_tas"] == cls_t]
        if len(sub) > 0:
            color_by_tas[cls_t] = sub["dominant_color_1"].value_counts().index[0]

    top_baju_name = baju_counts.index[0]
    top_baju_pct = (baju_counts.iloc[0] / total_analyzed) * 100.0
    top_tas_name = tas_counts.index[0]
    top_tas_pct = (tas_counts.iloc[0] / total_analyzed) * 100.0
    top_color_1 = top_colors[0][0]
    top_color_2 = top_colors[1][0] if len(top_colors) > 1 else top_colors[0][0]
    top_color_3 = top_colors[2][0] if len(top_colors) > 2 else top_colors[0][0]

    # -----------------------------------------------------------------
    # CETAK LAPORAN INSIGHT TREN KE TERMINAL
    # -----------------------------------------------------------------
    report_lines = []
    def log_line(text=""):
        print(text)
        report_lines.append(text)

    log_line("\n" + "=" * 75)
    log_line(f"LAPORAN INSIGHT TREN FASHION SAAT INI ({total_analyzed} GAMBAR INSTAGRAM)")
    log_line("=" * 75)

    log_line("\n1. TREN KATEGORI BAJU (MODEL VGG16 BAJU):")
    for cls_name, cnt in baju_counts.items():
        pct = (cnt / total_analyzed) * 100.0
        fav_col = color_by_baju.get(cls_name, "-")
        log_line(f"   - {cls_name:<14} : {cnt:>3} gambar ({pct:>5.1f}%) | Warna Terpopuler: {fav_col}")

    log_line("\n2. TREN KATEGORI TAS (MODEL VGG16 TAS):")
    for cls_name, cnt in tas_counts.items():
        pct = (cnt / total_analyzed) * 100.0
        fav_col = color_by_tas.get(cls_name, "-")
        log_line(f"   - {cls_name:<14} : {cnt:>3} gambar ({pct:>5.1f}%) | Warna Terpopuler: {fav_col}")

    log_line("\n3. TOP 8 WARNA DOMINAN YANG SEDANG TREN (K-MEANS CLUSTERING):")
    for rank, (col_name, weight) in enumerate(top_colors[:8], 1):
        pct = (weight / total_color_weight) * 100.0
        hex_val = COLOR_HEX_MAP.get(col_name, "#888888")
        log_line(f"   {rank}. {col_name:<28} ({hex_val}) : {pct:>5.1f}% pangsa visual")

    log_line("\n4. TOP 6 KOMBINASI ITEM & WARNA PALING BANYAK MUNCUL:")
    for rank, (combo_name, cnt) in enumerate(top_combos[:6], 1):
        pct = (cnt / total_analyzed) * 100.0
        log_line(f"   {rank}. {combo_name:<38} : {cnt:>3} foto ({pct:>4.1f}%)")

    log_line("\n5. KESIMPULAN & REKOMENDASI INSIGHT TREN SAAT INI:")
    log_line(
        f"   - Tren Pakaian (Apparel) saat ini didominasi oleh '{top_baju_name.upper()}' "
        f"({top_baju_pct:.1f}%), dengan kecenderungan warna utama {color_by_baju.get(top_baju_name, top_color_1)}."
    )
    log_line(
        f"   - Tren Tas (Bags) saat ini dipimpin oleh model '{top_tas_name.upper()}' "
        f"({top_tas_pct:.1f}%), yang paling sering dipadukan dengan warna {color_by_tas.get(top_tas_name, top_color_1)}."
    )
    log_line(
        f"   - Secara keseluruhan, palet warna pasar sedang mengarah kuat ke nuansa "
        f"'{top_color_1}', '{top_color_2}', dan '{top_color_3}'."
    )
    log_line("=" * 75)
    log_line(f"File CSV detail disimpan ke     : {os.path.abspath(csv_out)}")

    # Simpan laporan ke file Markdown
    md_out = "laporan_insight_trend_fashion.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"File Laporan Insight disimpan ke : {os.path.abspath(md_out)}")

    # -----------------------------------------------------------------
    # VISUALISASI DASHBOARD GRAFIK TREN (4 PANEL)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Dashboard Insight Tren Fashion Instagram ({total_analyzed} Gambar)\n"
        f"Pipeline: YOLOv8 Auto-Crop + Dual VGG16 + K-Means Color Clustering",
        fontsize=15, fontweight="bold"
    )

    # Panel 1: Tren Baju
    axes[0, 0].bar(baju_counts.index, baju_counts.values, color=["#2563eb", "#3b82f6", "#60a5fa"][:len(baju_counts)], edgecolor="#1e3a8a")
    axes[0, 0].set_title("1. Distribusi Tren Kategori Baju (VGG16 Baju)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Jumlah Gambar")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(baju_counts.values):
        axes[0, 0].text(i, v + max(1, total_analyzed * 0.01), f"{v} ({v/total_analyzed*100:.1f}%)", ha="center", fontweight="bold")

    # Panel 2: Tren Tas
    axes[0, 1].bar(tas_counts.index, tas_counts.values, color=["#059669", "#10b981", "#34d399"][:len(tas_counts)], edgecolor="#064e3b")
    axes[0, 1].set_title("2. Distribusi Tren Kategori Tas (VGG16 Tas)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Jumlah Gambar")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(tas_counts.values):
        axes[0, 1].text(i, v + max(1, total_analyzed * 0.01), f"{v} ({v/total_analyzed*100:.1f}%)", ha="center", fontweight="bold")

    # Panel 3: Top 8 Warna Dominan (Bar diwarnai sesuai kode HEX asli)
    top8_colors = top_colors[:8][::-1]
    c_labels = [c[0] for c in top8_colors]
    c_pcts = [(c[1] / total_color_weight) * 100.0 for c in top8_colors]
    c_hexes = [COLOR_HEX_MAP.get(c[0], "#64748b") for c in top8_colors]

    bars = axes[1, 0].barh(c_labels, c_pcts, color=c_hexes, edgecolor="#1e293b", linewidth=1.2)
    axes[1, 0].set_title("3. Top 8 Tren Warna Dominan (K-Means)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Pangsa Visual (%)")
    axes[1, 0].set_xlim(0, max(c_pcts) * 1.22)
    axes[1, 0].grid(axis="x", linestyle="--", alpha=0.5)
    for bar, p in zip(bars, c_pcts):
        axes[1, 0].text(p + 0.5, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va="center", fontweight="bold")

    # Panel 4: Top Kombinasi Item + Warna
    top6_combos = top_combos[:6][::-1]
    combo_labels = [c[0] for c in top6_combos]
    combo_vals = [c[1] for c in top6_combos]

    bars4 = axes[1, 1].barh(combo_labels, combo_vals, color="#8b5cf6", edgecolor="#4c1d95")
    axes[1, 1].set_title("4. Top 6 Kombinasi Item & Warna Terpopuler", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Jumlah Kemunculan (Foto)")
    axes[1, 1].set_xlim(0, max(combo_vals) * 1.22)
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.5)
    for bar, v in zip(bars4, combo_vals):
        axes[1, 1].text(v + 0.3, bar.get_y() + bar.get_height()/2, f"{v} foto ({v/total_analyzed*100:.1f}%)", va="center", fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    chart_out = "dashboard_trend_fashion_300.png"
    plt.savefig(chart_out, dpi=300)
    print(f"Grafik Dashboard disimpan ke     : {os.path.abspath(chart_out)}\n")
    plt.show()

if __name__ == "__main__":
    main()
