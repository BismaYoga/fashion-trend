import os
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Mencegah bug file-watcher Streamlit terhadap modul internal PyTorch (torch.classes)
try:
    torch.classes.__path__ = []
except Exception:
    pass

# =====================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Fashion Trend AI — YOLOv8 + VGG16 + K-Means",
    layout="wide"
)

st.title("Fashion Trend Analyzer: YOLOv8 + Dual VGG16 (Baju & Tas) + K-Means Color")
st.write(
    "Unggah foto untuk mendeteksi dan memotong bagian objek dengan **YOLOv8**, "
    "memprediksi klasifikasi menggunakan **VGG16 (Baju & Tas)**, "
    "serta menganalisis palet warna dominan menggunakan **K-Means Clustering**."
)

INPUT_SIZE = (224, 224)
SEED = 42

DEFAULT_CLASSES = {
    "baju": ["casual_top", "dress", "outerwear"],
    "tas": ["handbag", "sling_bag", "tote_bag"]
}

MODEL_PATHS = {
    "baju": "best_vgg16_baju.pth",
    "tas": "best_vgg16_tas.pth"
}

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

val_test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# 2. FUNGSI PEMUATAN MODEL (YOLOv8 & VGG16)
# =====================================================================
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

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

@st.cache_resource
def load_vgg_model(category):
    device = get_device()
    ckpt_path = MODEL_PATHS[category]
    classes = list(DEFAULT_CLASSES[category])

    if not os.path.exists(ckpt_path):
        model = build_vgg16_classifier(len(classes)).to(device)
        model.eval()
        return model, classes, False

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    # Mendukung format dictionary baru maupun state_dict langsung
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        classes = ckpt.get("class_names", classes)
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model = build_vgg16_classifier(len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes, True

# =====================================================================
# 3. FUNGSI DETEKSI & CROPPING YOLOv8 + ANALISIS WARNA K-MEANS
# =====================================================================
def detect_and_crop_yolov8(image_pil, yolo_model, category_mode="baju", target_size=(224, 224), conf_thres=0.20):
    img_rgb = np.array(image_pil.convert("RGB"))
    h, w, _ = img_rgb.shape
    results = yolo_model.predict(source=img_rgb, conf=conf_thres, verbose=False)
    boxes = results[0].boxes

    BAG_COCO_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
    PERSON_COCO_CLASS = 0            # person
    selected_box = None
    det_label = "center_fallback"
    det_conf = 1.0

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        if category_mode == "tas":
            bag_indices = [i for i, c in enumerate(clss) if c in BAG_COCO_CLASSES]
            if len(bag_indices) > 0:
                best_idx = max(bag_indices, key=lambda i: confs[i] * ((xyxy[i][2]-xyxy[i][0])*(xyxy[i][3]-xyxy[i][1])))
                selected_box = tuple(xyxy[best_idx])
                det_label = f"yolo_{yolo_model.names[clss[best_idx]]}"
                det_conf = float(confs[best_idx])
            else:
                person_indices = [i for i, c in enumerate(clss) if c == PERSON_COCO_CLASS]
                if len(person_indices) > 0:
                    best_idx = max(person_indices, key=lambda i: (xyxy[i][2]-xyxy[i][0])*(xyxy[i][3]-xyxy[i][1]))
                    px1, py1, px2, py2 = xyxy[best_idx]
                    pw, ph = (px2 - px1), (py2 - py1)
                    selected_box = (
                        max(0, px1 - 0.05 * pw),
                        max(0, py1 + 0.18 * ph),
                        min(w, px2 + 0.05 * pw),
                        min(h, py1 + 0.90 * ph)
                    )
                    det_label = "yolo_person_bag_region"
                    det_conf = float(confs[best_idx])
        else:
            person_indices = [i for i, c in enumerate(clss) if c == PERSON_COCO_CLASS]
            if len(person_indices) > 0:
                best_idx = max(person_indices, key=lambda i: (xyxy[i][2]-xyxy[i][0])*(xyxy[i][3]-xyxy[i][1]))
                px1, py1, px2, py2 = xyxy[best_idx]
                pw, ph = (px2 - px1), (py2 - py1)
                selected_box = (
                    max(0, px1 - 0.03 * pw),
                    max(0, py1 + 0.12 * ph),
                    min(w, px2 + 0.03 * pw),
                    min(h, py1 + 0.92 * ph)
                )
                det_label = "yolo_person_apparel"
                det_conf = float(confs[best_idx])
            else:
                best_idx = int(np.argmax(confs))
                selected_box = tuple(xyxy[best_idx])
                det_label = f"yolo_{yolo_model.names[clss[best_idx]]}"
                det_conf = float(confs[best_idx])

    if selected_box is None:
        mx, my = int(0.075 * w), int(0.075 * h)
        selected_box = (mx, my, w - mx, h - my)
        det_label = "center_object_crop"
        det_conf = 1.0

    x1, y1, x2, y2 = selected_box
    bw, bh = (x2 - x1), (y2 - y1)
    x1 = int(max(0, x1 - 0.04 * bw))
    y1 = int(max(0, y1 - 0.04 * bh))
    x2 = int(min(w, x2 + 0.04 * bw))
    y2 = int(min(h, y2 + 0.04 * bh))
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        x1, y1, x2, y2 = 0, 0, w, h

    cropped_pil = image_pil.crop((x1, y1, x2, y2)).convert("RGB")
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    cropped_resized_pil = cropped_pil.resize(target_size, resample_filter)

    return cropped_resized_pil, (x1, y1, x2, y2), {"method": det_label, "confidence": det_conf}

def get_closest_fashion_color_name(rgb_tuple):
    r, g, b = rgb_tuple
    return min(
        FASHION_COLOR_PALETTE.keys(),
        key=lambda name: (r - FASHION_COLOR_PALETTE[name][0]) ** 2
        + (g - FASHION_COLOR_PALETTE[name][1]) ** 2
        + (b - FASHION_COLOR_PALETTE[name][2]) ** 2
    )

def analyze_dominant_colors_kmeans(cropped_pil, n_clusters=3):
    img_np = np.array(cropped_pil.convert("RGB"))
    h, w, _ = img_np.shape
    my, mx = int(0.125 * h), int(0.125 * w)
    core_region = img_np[my:h - my, mx:w - mx]
    pixels = core_region.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    total_pixels = len(labels)

    results = []
    for idx in np.argsort(-counts):
        rgb = tuple(int(v) for v in np.clip(centroids[idx], 0, 255))
        hex_code = "#{:02X}{:02X}{:02X}".format(*rgb)
        pct = (counts[idx] / total_pixels) * 100.0
        results.append({
            "rgb": rgb,
            "hex": hex_code,
            "percentage": pct,
            "name": get_closest_fashion_color_name(rgb)
        })
    return results

def show_image_compat(img, caption=None):
    """Kompatibel untuk semua versi Streamlit (lama maupun baru)."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

# =====================================================================
# 4. ANTARMUKA UTAMA & EKSEKUSI PREDIKSI
# =====================================================================
st.sidebar.header("Pengaturan Pengujian")
test_mode = st.sidebar.radio(
    "Pilih Mode Model:",
    options=[
        "Kedua Model Sekaligus (Baju & Tas)",
        "Hanya Model Baju (VGG16 Baju)",
        "Hanya Model Tas (VGG16 Tas)"
    ],
    index=0
)
n_clusters_color = st.sidebar.slider("Jumlah Warna Dominan (K-Means):", min_value=2, max_value=5, value=3)

st.sidebar.markdown("---")
st.sidebar.subheader("Status File Model (.pth)")
for cat_key, path_val in MODEL_PATHS.items():
    if os.path.exists(path_val):
        st.sidebar.success(f"{cat_key.upper()}: Ditemukan ({path_val})")
    else:
        st.sidebar.warning(f"{cat_key.upper()}: Belum ada ({path_val})")

uploaded_file = st.file_uploader("Unggah Foto Fesyen (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    orig_image = Image.open(uploaded_file).convert("RGB")
    yolo_detector = load_yolo()
    device = get_device()

    categories_to_run = ["baju", "tas"]
    if "Hanya Model Baju" in test_mode:
        categories_to_run = ["baju"]
    elif "Hanya Model Tas" in test_mode:
        categories_to_run = ["tas"]

    for cat in categories_to_run:
        st.markdown("---")
        st.subheader(f"Hasil Analisis Model: {cat.upper()}")

        cropped_224, (x1, y1, x2, y2), det_info = detect_and_crop_yolov8(
            orig_image, yolo_detector, category_mode=cat, target_size=INPUT_SIZE
        )
        vgg_model, class_list, is_trained = load_vgg_model(cat)

        if not is_trained:
            st.warning(
                f"Catatan: File `{MODEL_PATHS[cat]}` belum ditemukan di folder yang sama dengan `app.py`. "
                f"Pastikan file `{MODEL_PATHS[cat]}` berada satu folder dengan `app.py`."
            )

        input_tensor = val_test_transforms(cropped_224).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = vgg_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        pred_label = class_list[top_idx]
        pred_conf = float(probs[top_idx]) * 100.0
        colors_info = analyze_dominant_colors_kmeans(cropped_224, n_clusters=n_clusters_color)

        # Gambar Bounding Box pada Citra Asli
        boxed_img = orig_image.copy()
        draw = ImageDraw.Draw(boxed_img)
        box_color = "#2563eb" if cat == "baju" else "#10b981"
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=5)

        col1, col2, col3, col4 = st.columns([1.2, 1.0, 1.3, 1.5])

        with col1:
            st.markdown("**1. Deteksi YOLOv8**")
            show_image_compat(boxed_img)
            st.caption(f"Metode: `{det_info['method']}` | Conf: `{det_info['confidence']:.2f}`")

        with col2:
            st.markdown(f"**2. Hasil Crop ({INPUT_SIZE[0]}x{INPUT_SIZE[1]})**")
            show_image_compat(cropped_224)
            st.caption(f"BBox: `({x1}, {y1}) - ({x2}, {y2})`")

        with col3:
            st.markdown(f"**3. Prediksi VGG16 ({cat.upper()})**")
            st.metric(label="Prediksi Kelas", value=pred_label.upper(), delta=f"{pred_conf:.2f}% Confidence")
            for c_name, p_val in zip(class_list, probs * 100.0):
                st.write(f"`{c_name}`: **{p_val:.1f}%**")
                st.progress(min(max(int(round(p_val)), 0), 100))

        with col4:
            st.markdown(f"**4. Warna Dominan (K-Means Top {n_clusters_color})**")
            for idx_c, c_item in enumerate(colors_info, 1):
                swatch_html = (
                    f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;'
                    f'padding:8px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;">'
                    f'<div style="width:42px;height:42px;border-radius:6px;background:{c_item["hex"]};'
                    f'border:1px solid #334155;flex-shrink:0;"></div>'
                    f'<div style="font-size:13px;line-height:1.4;color:#0f172a;">'
                    f'<strong>#{idx_c} {c_item["name"]}</strong><br>'
                    f'HEX: <code>{c_item["hex"]}</code> | Proporsi: <strong>{c_item["percentage"]:.1f}%</strong>'
                    f'</div></div>'
                )
                st.markdown(swatch_html, unsafe_allow_html=True)
