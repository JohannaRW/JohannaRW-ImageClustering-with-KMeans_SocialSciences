import numpy as np
import joblib, os
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
import csv
import re
from Save_CSVs import export_all_filenames_with_clusters
from Graph_PCA import plot_clusters_pca

from PIL import Image, ImageOps, ImageDraw, ImageFont
import pillow_heif
pillow_heif.register_heif_opener()

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ====== Image folder ======

image_folder = ""
img_paths = (
    glob(os.path.join(image_folder, "*.jpeg"))
    + glob(os.path.join(image_folder, "*.jpg"))
    + glob(os.path.join(image_folder, "*.webp"))
    + glob(os.path.join(image_folder, "*.png"))
    + glob(os.path.join(image_folder, "*.heic"))
    + glob(os.path.join(image_folder, "*.heif"))
)
img_paths = sorted(img_paths)

# ====== Load or extract features ======
feature_folder = os.path.join(image_folder, "features")
os.makedirs(feature_folder, exist_ok=True)

# ====== Feature extraction: layer selection ======
# Options: "fc1" or "avg"
feature_mode = input("Choose feature mode (fc1 / avg): ").strip().lower()

if feature_mode == "fc1":
    print("[INFO] Extracting features from layer fc1 (4096D)")
    base_model = VGG16(weights="imagenet", include_top=True)
    feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
    feature_dim = 4096

elif feature_mode == "avg":
    print("[INFO] Extracting features with Global Average Pooling (512D)")
    feature_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    feature_dim = 512

else:
    raise ValueError("Invalid input: please choose 'fc1' or 'avg'.")


# ====== Compute or load features ======
features, feat_files = [], []
img_size = (224, 224)

for i, imgpath in enumerate(img_paths, 1):
    base = os.path.basename(imgpath)
    stem = os.path.splitext(base)[0]
    savepath = os.path.join(feature_folder, f"{stem}_{feature_mode}_feat.pkl")

    if os.path.exists(savepath):
        vec = joblib.load(savepath)
    else:
        img = load_img(imgpath, target_size=img_size)
        arr = img_to_array(img)[None, ...]
        arr = preprocess_input(arr)
        vec = feature_model.predict(arr, verbose=0)
        joblib.dump(vec, savepath)

    features.append(vec[0])
    feat_files.append(savepath)

    if i % 50 == 0 or i == len(img_paths):
        print(f"[EXTRACT] {i}/{len(img_paths)} processed …")

X = np.vstack(features)
print("[INFO] Feature matrix:", X.shape)

# ====== PCA to 200D ======
pca = PCA(n_components=200, random_state=42)
X_reduced = pca.fit_transform(X)

# ====== Elbow & Silhouette ======
inertias = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_reduced)
    inertias.append(km.inertia_)

plt.figure(); plt.plot(K, inertias, marker="o")
plt.title("Elbow method"); plt.xlabel("k"); plt.ylabel("Inertia"); plt.tight_layout(); plt.show()

sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    sil_scores.append(score)

plt.figure(); plt.plot(list(K_range), sil_scores, marker="o")
plt.title("Silhouette scores"); plt.xlabel("k"); plt.ylabel("Score"); plt.tight_layout(); plt.show()

# ====== AutoElbow (external) ======
print("[INFO] AutoElbow (rupakbob) starting …")
from autoelbow_rupakbob import autoelbow
n = autoelbow.auto_elbow_search(X_reduced)

try:
    k_star = int(n)
except Exception:
    k_star = int(n[0]) if isinstance(n, (list, tuple)) else int(n)
print(f"[INFO] AutoElbow suggests k* = {k_star}")

# ====== User-defined cluster number ======
while True:
    try:
        k_best = int(input("How many clusters (k)? ").strip())
        if k_best >= 1: break
        print("Please enter a number >= 1.")
    except ValueError:
        print("Invalid input.")

# ====== Final clustering ======
km_final = KMeans(n_clusters=k_best, random_state=42, n_init="auto")
labels = km_final.fit_predict(X_reduced)
counts = Counter(labels)
print("[INFO] Cluster sizes:", dict(counts))

# ====== Center selection ======
def nearest_to_kmeans_center(X, labels, kmeans_model, top_n=10):
    results = {}
    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]
        center = kmeans_model.cluster_centers_[cl]
        dists = np.linalg.norm(X[idxs] - center, axis=1)
        nearest = idxs[np.argsort(dists)[:top_n]]
        results[cl] = list(nearest)
    return results

def nearest_to_medoid(X, labels, top_n=10):
    results = {}
    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]
        cluster_points = X[idxs]
        dists = np.linalg.norm(cluster_points[:, None] - cluster_points[None, :], axis=2)
        medoid_idx = np.argmin(dists.sum(axis=0))
        medoid = cluster_points[medoid_idx]
        dist_to_medoid = np.linalg.norm(cluster_points - medoid, axis=1)
        nearest = idxs[np.argsort(dist_to_medoid)[:top_n]]
        results[cl] = list(nearest)
    return results

# ====== Panel rendering function ======
def _extract_date_from_filename(fname):
    """
    Tries to find a date in format YYYY-MM-DD in the filename.
    If none is found → return filename.
    """
    m = re.search(r'(20\d{2})[-_\.](\d{2})[-_\.](\d{2})', fname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return os.path.basename(fname)

# ==== Render panels ====
def render_clusters_panel_10(nearest_idx_map, img_paths, out_path, thumb=200, padding=16):
    cluster_ids = list(nearest_idx_map.keys())  # Order same as KMeans labels
    n_cols = 10
    n_rows = len(cluster_ids)
    cell = thumb + padding + 60  # extra space for text
    width = padding + n_cols * cell
    height = 50 + n_rows * cell
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for r, cl in enumerate(cluster_ids):
        idxs = nearest_idx_map[cl]  # sorted by centrality

        # Total size per cluster from 'labels'
        cluster_size = np.sum(labels == cl)

        # Title with cluster ID + total size
        draw.text(
            (padding, 20 + r * cell),
            f"Cluster {cl:02d} (n={cluster_size})",
            fill=(0, 0, 0),
            font=font
        )

        for c, i in enumerate(idxs[:n_cols]):
            try:
                im = Image.open(img_paths[i])
                im = ImageOps.exif_transpose(im)
                im.thumbnail((thumb, thumb))

                x = padding + c * cell
                y = 40 + r * cell

                # Paste image
                panel.paste(im, (x, y))

                # Caption centered
                text = _extract_date_from_filename(img_paths[i])
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                text_x = x + (thumb - text_w) // 2
                text_y = y + thumb + 10
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

            except Exception as e:
                print(f"[WARN] Error with {img_paths[i]}: {e}")
                pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panel.save(out_path)
    print(f"[INFO] Panel saved: {out_path}")

# ====== Generate panels ======
preview_dir = "cluster_previews"
nearest = 10

nearest_idx_map_kmeans = nearest_to_kmeans_center(X_reduced, labels, km_final, top_n=nearest)
nearest_idx_map_medoid = nearest_to_medoid(X_reduced, labels, top_n=nearest)

render_clusters_panel_10(nearest_idx_map_kmeans, img_paths,
                         out_path=os.path.join(preview_dir, "clusters_kmeans10_panel.png"))

render_clusters_panel_10(nearest_idx_map_medoid, img_paths,
                         out_path=os.path.join(preview_dir, "clusters_medoid10_panel.png"))




# ====== Export filenames of 10 most typical images per cluster ======

def save_cluster_filenames(nearest_idx_map, img_paths, out_csv):
    """
    Saves the filenames of the 10 most representative images per cluster.
    """
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "rank", "filename"])
        for cl in sorted(nearest_idx_map.keys()):
            idxs = nearest_idx_map[cl]
            for rank, i in enumerate(idxs[:10], start=1):
                writer.writerow([cl, rank, os.path.basename(img_paths[i])])
    print(f"[INFO] CSV file saved: {out_csv}")

# KMeans variant
save_cluster_filenames(
    nearest_idx_map_kmeans,
    img_paths,
    os.path.join(preview_dir, "cluster_kmeans10_filenames.csv")
)

# Medoid variant
save_cluster_filenames(
    nearest_idx_map_medoid,
    img_paths,
    os.path.join(preview_dir, "cluster_medoid10_filenames.csv")
)

# ====== Graph_PCA ======

plot_clusters_pca(X_reduced, labels, kmeans_model=km_final)

# ====== Export CSV with all images ======
export_all_filenames_with_clusters(
    labels,
    img_paths,
    os.path.join(preview_dir, "all_images_with_clusters.csv"))

import shutil

def export_typical_images(nearest_idx_map, img_paths, out_dir, method="kmeans"):
    """
    Saves the 10 most typical images per cluster in separate folders.
    """
    os.makedirs(out_dir, exist_ok=True)
    for cl, idxs in nearest_idx_map.items():
        cluster_dir = os.path.join(out_dir, f"cluster_{cl:02d}_{method}")
        os.makedirs(cluster_dir, exist_ok=True)
        for rank, i in enumerate(idxs[:10], start=1):
            src = img_paths[i]
            fname = f"{rank:02d}_" + os.path.basename(src)
            dst = os.path.join(cluster_dir, fname)
            shutil.copy(src, dst)
        print(f"[INFO] {len(idxs[:10])} images for cluster  {cl} → {cluster_dir}")


# Call for your KMeans variant:
export_typical_images(nearest_idx_map_kmeans, img_paths, "cluster_typical_images", method="kmeans")
