# 06_h_clustering_testing.py
# check the result and structure of the Ward clusters.
#
# produces (all prefixed 06_, saved into the "06 Results of Tested Clusters"
# folder created next to this script):
#   - 06_clusters_summary.csv : per-cluster size, cohesion, and top words (a quick label)
#   - 06_cluster_examples.txt : the most representative paper titles in each cluster
#   - 06_cluster_scatter.png  : a 2D map of all papers, coloured by cluster
#   - prints an overall silhouette score
#
# run this AFTER 05 (it reads 05_paper_clusters.csv from 05's results folder).

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to file, no display window
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------- CONFIG ----------
# this script's folder: ...\Hierarchical Clustering
BASE_DIR = Path(__file__).resolve().parent

# inputs that live in SIBLING folders (one level up, then in):
SIM_DIR = BASE_DIR.parent / "Similarity Metrics"                  # embeddings live here
DATASET_DIR = BASE_DIR.parent / "Neurips Dataset" / "Dataset"     # original csv lives here

EMBEDDINGS_FILE = SIM_DIR / "04 MiniLM Results" / "embeddings.npy"                              # from the 04 script
CLUSTERS_FILE   = BASE_DIR / "05 Clustering Results" / "05_paper_clusters.csv"  # from the 05 script
ORIGINAL_CSV    = DATASET_DIR / "neurips_accepted_submissions.csv"
ID_COL          = "Number"
TITLE_COL       = "Title"
ABSTRACT_COL    = "Abstract"

# all results go into this sub-folder, created next to this script automatically
OUTPUT_DIR = BASE_DIR / "06 Results of Tested Clusters"
OUTPUT_DIR.mkdir(exist_ok=True)

N_TITLES   = 5        # representative titles to show per cluster
N_TERMS    = 8        # top words to show per cluster
PROJECTION = "pca"    # "pca" (fast) or "tsne" (slower, usually clearer separation)
# ----------------------------

# fail clearly if any input is missing
for f in (EMBEDDINGS_FILE, CLUSTERS_FILE, ORIGINAL_CSV):
    if not f.exists():
        print(f"could not find: {f}")
        print("check the paths in the CONFIG section at the top of this script.")
        raise SystemExit(1)

# load embeddings and clusters (same row order), then attach titles/abstracts
embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
clusters = pd.read_csv(CLUSTERS_FILE)
assert len(embeddings) == len(clusters), "embeddings.npy and 05_paper_clusters.csv are misaligned"

original = pd.read_csv(ORIGINAL_CSV)
clusters = clusters.merge(original[[ID_COL, TITLE_COL, ABSTRACT_COL]], on=ID_COL, how="left")
assert len(clusters) == len(embeddings), "merge changed the row count (duplicate Numbers?)"
labels = clusters["cluster"].to_numpy()
n = len(clusters)
print(f"{n} papers across {clusters['cluster'].nunique()} clusters")

# ---- 1. overall quality: silhouette (subsampled for speed/memory) ----
sample = min(n, 3000)
sil = silhouette_score(embeddings, labels, metric="cosine", sample_size=sample, random_state=42)
print(f"overall silhouette (cosine, sample={sample}): {sil:.3f}   (higher is better, max 1.0)")

# ---- 2. per-cluster cohesion: how close papers are to their cluster's centre ----
# embeddings are unit length, so cosine similarity = dot product
summary_rows = []
for c in sorted(clusters["cluster"].unique()):
    mask = labels == c
    members = embeddings[mask]
    centroid = members.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)  # back to unit length
    cohesion = float(np.mean(members @ centroid))             # avg similarity to centre
    summary_rows.append({"cluster": int(c), "size": int(mask.sum()),
                         "cohesion": round(cohesion, 3)})
summary = pd.DataFrame(summary_rows)

# ---- 3. top distinctive words per cluster (a quick automatic label) ----
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(clusters[ABSTRACT_COL].fillna("").astype(str))
terms = np.array(tfidf.get_feature_names_out())

top_terms = {}
for c in summary["cluster"]:
    mask = labels == c
    mean_tfidf = np.asarray(X[mask].mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[::-1][:N_TERMS]
    top_terms[c] = ", ".join(terms[top_idx])

summary["top_terms"] = summary["cluster"].map(top_terms)
summary.to_csv(OUTPUT_DIR / "06_clusters_summary.csv", index=False)
print("\n=== cluster summary ===")
print(summary.to_string(index=False))
print("\nsaved 06_clusters_summary.csv")

# ---- 4. representative paper titles per cluster (closest to the centre) ----
lines = []
for c in summary["cluster"]:
    mask = labels == c
    members = embeddings[mask]
    centroid = members.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = members @ centroid
    member_titles = clusters.loc[mask, TITLE_COL].to_numpy()
    order = sims.argsort()[::-1][:N_TITLES]
    lines.append(f"\n--- cluster {c}  ({int(mask.sum())} papers) ---")
    lines.append(f"top words: {top_terms[c]}")
    for t in member_titles[order]:
        lines.append(f"   - {t}")

examples_text = "\n".join(lines)
(OUTPUT_DIR / "06_cluster_examples.txt").write_text(examples_text, encoding="utf-8")
# print safely: the Windows console may not be UTF-8, and some abstracts contain
# characters (e.g. non-breaking hyphens) it can't encode. Replace those for display
# only; the saved file above keeps the real characters.
_enc = sys.stdout.encoding or "utf-8"
print(examples_text.encode(_enc, errors="replace").decode(_enc))
print("\nsaved 06_cluster_examples.txt")

# ---- 5. 2D map of all papers, coloured by cluster ----
if PROJECTION == "tsne":
    # clearer separation but slow on many points
    from sklearn.manifold import TSNE
    coords = TSNE(n_components=2, metric="cosine", init="pca",
                  random_state=42).fit_transform(embeddings)
else:
    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=8, alpha=0.7)
plt.title(f"papers coloured by cluster ({PROJECTION.upper()} projection)")
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_cluster_scatter.png", dpi=150)
print("\nsaved 06_cluster_scatter.png")