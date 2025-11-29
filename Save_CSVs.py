import csv
import os

def export_all_filenames_with_clusters(labels, img_paths, out_csv):
    """
    Speichert alle Bilder mit ihrem Cluster-Label in eine CSV-Datei,
    sortiert nach Cluster-ID und Dateiname.
    """
    rows = []
    for i, path in enumerate(img_paths):
        rows.append([labels[i], os.path.basename(path), path])

    # Sortieren nach Cluster-ID, dann Dateiname
    rows.sort(key=lambda x: (x[0], x[1]))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "filename", "full_path"])
        writer.writerows(rows)

    print(f"[INFO] CSV mit allen {len(img_paths)} Bildern gespeichert: {out_csv}")
