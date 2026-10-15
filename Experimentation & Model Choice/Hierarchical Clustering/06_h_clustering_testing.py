#stage 2 check, tests the result and structure of the ward clusters from 05

#what it produces (all prefixed 06_, saved into the "06 Results of Tested Clusters"
#folder created next to this script):
#06_clusters_summary.csv holds per-cluster size, cohesion, and top words (a quick label)
#06_cluster_examples.txt holds the most representative paper titles in each cluster
#06_cluster_scatter.png is a 2D map of all papers, coloured by cluster
#and an overall silhouette score gets printed

#run this AFTER 05, it reads 05_paper_clusters.csv from 05s results folder

#the two quality measures used here:
#silhouette (rousseeuw 1987) scores how tight and well separated the groups are
#cohesion is simpler, the average similarity of a clusters papers to their own
#centre, so it asks one question only, is this group about one thing
#the top words come from tf-idf (sparck jones 1972), which upweights words that
#are frequent inside a cluster but rare across the whole collection

#libraries used in this script and where each one comes from:
#pathlib and sys are python standard library, docs.python.org
#numpy does the array math, harris et al. 2020, github.com/numpy/numpy
#pandas reads, merges and writes the csvs, mckinney 2010, github.com/pandas-dev/pandas
#matplotlib draws the scatter picture, hunter 2007, github.com/matplotlib/matplotlib
#sklearn gives tf-idf, silhouette and pca, pedregosa et al. 2011,
#github.com/scikit-learn/scikit-learn
#pca itself goes back to pearson 1901, tsne (the optional slower map) is
#van der maaten & hinton 2008

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  #save figures to file, no display window needed
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#CONFIG
#everything adjustable lives up here so nothing needs hunting deeper in the file

#anchor everything to this scripts own location instead of wherever the
#terminal happens to be standing
BASE_DIR = Path(__file__).resolve().parent  #...\Hierarchical Clustering

#inputs that live in SIBLING folders (one level up, then in):
SIM_DIR = BASE_DIR.parent / "Similarity Metrics"                  #embeddings live here
DATASET_DIR = BASE_DIR.parent / "Neurips Dataset" / "Dataset"     #original csv lives here

EMBEDDINGS_FILE = SIM_DIR / "04 MiniLM Results" / "embeddings.npy"                              #made by 04
CLUSTERS_FILE   = BASE_DIR / "05 Clustering Results" / "05_paper_clusters.csv"  #made by 05
ORIGINAL_CSV    = DATASET_DIR / "neurips_accepted_submissions.csv"
ID_COL          = "Number"  #the column that identifies a paper
TITLE_COL       = "Title"
ABSTRACT_COL    = "Abstract"

#all results go into this sub-folder, created next to this script automatically
#exist_ok=True means if the folder already exists just carry on, dont crash
OUTPUT_DIR = BASE_DIR / "06 Results of Tested Clusters"
OUTPUT_DIR.mkdir(exist_ok=True)

N_TITLES   = 5        #representative titles to show per cluster
N_TERMS    = 8        #top words to show per cluster
PROJECTION = "pca"    #"pca" (fast) or "tsne" (slower, usually clearer separation)

#stop early with a clear message if any input is missing,
#way better than a confusing crash further down
for f in (EMBEDDINGS_FILE, CLUSTERS_FILE, ORIGINAL_CSV):
    if not f.exists():
        print(f"could not find: {f}")
        print("check the paths in the CONFIG section at the top of this script.")
        raise SystemExit(1)

#LOADING
#embeddings and cluster labels share row order, thats the contract from 04 and 05,
#and both asserts below exist to catch it being broken
#the merge attaches titles and abstracts from the original csv by paper number,
#how="left" keeps every clustered paper even if a lookup failed
#the second assert catches duplicate paper numbers, a merge on duplicates would
#silently grow the table and everything after would be quietly wrong
embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
clusters = pd.read_csv(CLUSTERS_FILE)
assert len(embeddings) == len(clusters), "embeddings.npy and 05_paper_clusters.csv are misaligned"

original = pd.read_csv(ORIGINAL_CSV)
clusters = clusters.merge(original[[ID_COL, TITLE_COL, ABSTRACT_COL]], on=ID_COL, how="left")
assert len(clusters) == len(embeddings), "merge changed the row count (duplicate Numbers?)"
labels = clusters["cluster"].to_numpy()
n = len(clusters)
print(f"{n} papers across {clusters['cluster'].nunique()} clusters")

#OVERALL QUALITY, THE SILHOUETTE (rousseeuw 1987)
#one number for the whole clustering, higher = tighter and better separated
#computed on a fixed 3000-paper sample with a fixed random_state, the full
#pairwise computation would be slow and heavy and the sample is plenty
sample = min(n, 3000)
sil = silhouette_score(embeddings, labels, metric="cosine", sample_size=sample, random_state=42)
print(f"overall silhouette (cosine, sample={sample}): {sil:.3f}   (higher is better, max 1.0)")

#PER-CLUSTER COHESION
#how close each clusters papers sit to their own centre
#the centroid is the average vector, normalised back to unit length,
#and since the embeddings are unit length a plain dot product IS the cosine
#similarity, so members @ centroid gives every papers similarity to the centre
#in one go, and the mean of that is the clusters cohesion
#the +1e-12 stops a divide by zero on a degenerate centroid
summary_rows = []
for c in sorted(clusters["cluster"].unique()):
    mask = labels == c
    members = embeddings[mask]
    centroid = members.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)  #back to unit length
    cohesion = float(np.mean(members @ centroid))             #avg similarity to the centre
    summary_rows.append({"cluster": int(c), "size": int(mask.sum()),
                         "cohesion": round(cohesion, 3)})
summary = pd.DataFrame(summary_rows)

#TOP DISTINCTIVE WORDS PER CLUSTER (a quick automatic label)
#tf-idf (sparck jones 1972) scores every word by how frequent it is in a text
#against how rare it is across all texts, so generic words score low and the
#words that actually characterise a cluster score high
#stop_words="english" drops filler like the/and/of, max_features caps the
#vocabulary at the 5000 strongest words so the matrix stays light
#for each cluster the tf-idf rows of its members get averaged and the
#N_TERMS highest-scoring words become its label
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(clusters[ABSTRACT_COL].fillna("").astype(str))
terms = np.array(tfidf.get_feature_names_out())

top_terms = {}
for c in summary["cluster"]:
    mask = labels == c
    mean_tfidf = np.asarray(X[mask].mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[::-1][:N_TERMS]  #argsort ascending, [::-1] flips it, best first
    top_terms[c] = ", ".join(terms[top_idx])

summary["top_terms"] = summary["cluster"].map(top_terms)
summary.to_csv(OUTPUT_DIR / "06_clusters_summary.csv", index=False)
print("\n=== cluster summary ===")
print(summary.to_string(index=False))
print("\nsaved 06_clusters_summary.csv")

#REPRESENTATIVE TITLES PER CLUSTER
#the papers closest to each clusters centre are its most typical members,
#so their titles make a readable sample of what the cluster is about
#same centroid trick as the cohesion block, then the N_TITLES highest
#similarities picked off the sorted order
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
#print safely: the windows console may not be utf-8, and some abstracts contain
#characters (for example non-breaking hyphens) it cant encode
#encode with errors="replace" swaps those for display only,
#the saved file above keeps the real characters
#this is a real crash guard, do not remove it
_enc = sys.stdout.encoding or "utf-8"
print(examples_text.encode(_enc, errors="replace").decode(_enc))
print("\nsaved 06_cluster_examples.txt")

#THE 2D MAP
#384-dimensional vectors cant be drawn, so they get squashed down to 2 numbers
#per paper just for the picture
#pca (pearson 1901) is the fast squash, it keeps the two directions along which
#the papers spread out the most
#tsne (van der maaten & hinton 2008) usually separates groups more clearly but
#is much slower on this many points, the PROJECTION switch up top picks
#the import sits inside the if on purpose, tsne only loads when its wanted
if PROJECTION == "tsne":
    #clearer separation but slow on many points
    from sklearn.manifold import TSNE
    coords = TSNE(n_components=2, metric="cosine", init="pca",
                  random_state=42).fit_transform(embeddings)
else:
    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)

#one dot per paper, coloured by cluster, tab20 cycles through 20 colours,
#s is the dot size and alpha the transparency so dense areas stay readable
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=8, alpha=0.7)
plt.title(f"papers coloured by cluster ({PROJECTION.upper()} projection)")
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_cluster_scatter.png", dpi=150)
print("\nsaved 06_cluster_scatter.png")