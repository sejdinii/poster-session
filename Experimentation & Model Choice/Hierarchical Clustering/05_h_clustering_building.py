# 05_h_clustering_building.py
# Build the Ward tree and OVER-CLUSTER: cut into many more small groups than rooms,
# so the later optimisation step has flexible building blocks to pack into rooms.
#
# Key idea: we deliberately cut into MORE clusters than rooms (but not down to single
# papers). More small clusters = more freedom for the optimiser, while Ward still does
# the cheap work of grouping clearly-similar papers.
#
# The tree is CACHED (05_ward_linkage.npy). On later runs it is loaded instead of
# rebuilt, so you can try different cluster counts instantly: just change NUM_CLUSTERS
# and re-run. (Delete 05_ward_linkage.npy to force a rebuild, e.g. after re-embedding.)
#
# inputs:  embeddings.npy, paper_index.csv   (Similarity Metrics folder)
# outputs: everything lands in the "05 Clustering Results" folder next to this script:
#          05_ward_linkage.npy, 05_ward_dendrogram.png, 05_paper_clusters.csv,
#          05_timing_clustering.csv

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent          # ...\Hierarchical Clustering
SIM_DIR  = BASE_DIR.parent / "Similarity Metrics"   # sibling folder with the embeddings

EMBEDDINGS_FILE = SIM_DIR / "04 MiniLM Results" / "embeddings.npy"
INDEX_FILE      = SIM_DIR / "04 MiniLM Results" / "paper_index.csv"
ID_COL          = "Number"

# all results (including the cached tree) go into this sub-folder,
# created next to this script automatically if its missing
OUTPUT_DIR = BASE_DIR / "05 Clustering Results"
OUTPUT_DIR.mkdir(exist_ok=True)

# OVER-CLUSTERING: the cluster count is tied to the data size instead of a fixed
# number: NUM_CLUSTERS = number of papers / PAPERS_PER_CLUSTER (rounded), i.e. groups
# of ~10 papers on average. It is computed below, once the embeddings are loaded and
# the paper count n is known.
PAPERS_PER_CLUSTER = 10

# silhouette sweep range (CONTEXT ONLY). Over-clustering deliberately goes past the
# silhouette-best point, so treat this as background, not the decision.
K_RANGE = range(5, 51, 5)
# ----------------------------

if not EMBEDDINGS_FILE.exists():
    print(f"could not find: {EMBEDDINGS_FILE}")
    if SIM_DIR.exists():
        print("npy files in that folder:")
        for f in SIM_DIR.glob("*.npy"):
            print("   ", f.name)
    raise SystemExit(1)

timings = {}

# load embeddings + ids
t0 = time.perf_counter()
embeddings = np.load(EMBEDDINGS_FILE).astype(np.float64)
index = pd.read_csv(INDEX_FILE)
timings["load_data"] = time.perf_counter() - t0
n = len(embeddings)
print(f"loaded {n} embeddings of dimension {embeddings.shape[1]}")
assert len(index) == n, "embeddings.npy and paper_index.csv have different lengths"

NUM_CLUSTERS = round(n / PAPERS_PER_CLUSTER)
print(f"cluster count: {n} papers / {PAPERS_PER_CLUSTER} per cluster -> {NUM_CLUSTERS} clusters")

# build the Ward tree, OR load it from cache if we already built it for this data
LINKAGE_FILE = OUTPUT_DIR / "05_ward_linkage.npy"
Z = None
if LINKAGE_FILE.exists():
    cached = np.load(LINKAGE_FILE)
    if cached.shape[0] == n - 1:        # a valid tree for n points has n-1 merges
        t0 = time.perf_counter()
        Z = cached
        timings["load_linkage"] = time.perf_counter() - t0
        print(f"loaded cached Ward tree from {LINKAGE_FILE.name} "
              f"(delete this file to rebuild)")
    else:
        print("cached tree size doesn't match the data; rebuilding...")

if Z is None:
    print("building the Ward linkage tree (heavy step; grows with n^2)...")
    t0 = time.perf_counter()
    Z = linkage(embeddings, method="ward")
    timings["build_linkage"] = time.perf_counter() - t0
    np.save(LINKAGE_FILE, Z)
    print(f"linkage tree built in {timings['build_linkage']:.2f}s and cached")

# silhouette sweep (context only)
print("\nsilhouette scores by number of groups (context only - higher = tighter):")
t0 = time.perf_counter()
sample_size = min(n, 3000)
for k in K_RANGE:
    labels_k = fcluster(Z, t=k, criterion="maxclust")
    score = silhouette_score(embeddings, labels_k, metric="cosine",
                             sample_size=sample_size, random_state=42)
    print(f"  k = {k:>3}:  silhouette = {score:.3f}")
timings["silhouette_sweep"] = time.perf_counter() - t0

# dendrogram (top of the tree)
t0 = time.perf_counter()
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("Ward dendrogram (top 30 merges)")
plt.xlabel("papers / merged groups")
plt.ylabel("merge distance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_ward_dendrogram.png", dpi=150)
timings["dendrogram"] = time.perf_counter() - t0
print("\ndendrogram saved to 05_ward_dendrogram.png")

# cut into NUM_CLUSTERS small groups and save
t0 = time.perf_counter()
labels = fcluster(Z, t=NUM_CLUSTERS, criterion="maxclust")
index["cluster"] = labels
index.to_csv(OUTPUT_DIR / "05_paper_clusters.csv", index=False)
timings["cut_and_save"] = time.perf_counter() - t0
print(f"\nover-clustered into {NUM_CLUSTERS} groups -> saved 05_paper_clusters.csv")

# size distribution - judge the over-clustering by this; it is what the optimiser packs
sizes = index["cluster"].value_counts()
print("\ncluster size distribution:")
print(f"  clusters:        {NUM_CLUSTERS}")
print(f"  smallest:        {int(sizes.min())} papers")
print(f"  largest:         {int(sizes.max())} papers")
print(f"  average:         {sizes.mean():.1f} papers")
print(f"  median:          {int(sizes.median())} papers")
print(f"  singletons (=1): {int((sizes == 1).sum())} clusters")
print(f"  big (>100):      {int((sizes > 100).sum())} clusters")

# timing summary + save
total = sum(timings.values())
print("\n=== timing summary (seconds) ===")
for step, secs in timings.items():
    print(f"  {step:<20} {secs:8.2f}")
print(f"  {'TOTAL':<20} {total:8.2f}")

metrics = pd.DataFrame(
    [{"step": s, "seconds": round(t, 3)} for s, t in timings.items()]
    + [{"step": "TOTAL", "seconds": round(total, 3)}]
)
metrics["n_papers"] = n
metrics["num_clusters"] = NUM_CLUSTERS
metrics.to_csv(OUTPUT_DIR / "05_timing_clustering.csv", index=False)
print("saved timing metrics to 05_timing_clustering.csv")