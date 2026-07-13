# 04_miniLM_similarity_model.py
# stage 1 of the grouping pipeline: turn abstracts into vectors, then into a
# similarity matrix -- now also records how long each step takes.
#
# what this does:
#   1. load the neurips abstracts
#   2. embed every abstract with all-MiniLM-L6-v2 (one 384-dim vector per paper)
#   3. compute the full pairwise cosine similarity matrix
#   4. save embeddings + similarity matrix + the row->paper mapping to disk
#   5. report timing metrics (printed + saved to timing_embedding.csv)
#
# we STOP here. dissimilarity, clustering, and the room optimisation come later.

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent          # ...\Similarity Metrics
DATASET_DIR = BASE_DIR.parent / "Neurips Dataset" / "Dataset"   # csv from dataset_API.py

INPUT_CSV    = DATASET_DIR / "neurips_accepted_submissions.csv"
ABSTRACT_COL = "Abstract"
ID_COL       = "Number"
MODEL_NAME   = "all-MiniLM-L6-v2"
BATCH_SIZE   = 128

# all results go into this sub-folder, created next to this script automatically
OUTPUT_DIR = BASE_DIR / "04 MiniLM Results"
OUTPUT_DIR.mkdir(exist_ok=True)
# ----------------------------

# we collect (step name -> seconds) here and print/save a summary at the end
timings = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
if device == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")

if not INPUT_CSV.exists():
    print(f"could not find: {INPUT_CSV}")
    print("run dataset_API.py first, or fix the DATASET_DIR line at the top of this script.")
    raise SystemExit(1)

df = pd.read_csv(INPUT_CSV)
print("columns in the file:", df.columns.tolist())
abstracts = df[ABSTRACT_COL].astype(str).tolist()
n = len(abstracts)
print(f"loaded {n} abstracts")

matrix_gb = (n * n * 4) / (1024 ** 3)
print(f"the n x n similarity matrix will be about {matrix_gb:.2f} GB in float32")

# time: load the model
# (note: the FIRST run also downloads the model, which makes this step much slower)
t0 = time.perf_counter()
model = SentenceTransformer(MODEL_NAME, device=device)
timings["load_model"] = time.perf_counter() - t0

# time: embed every abstract (this is the gpu-heavy step)
# encode(convert_to_numpy=True) waits for the gpu to finish before returning,
# so timing it with perf_counter is accurate.
t0 = time.perf_counter()
embeddings = model.encode(
    abstracts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
).astype(np.float32)
timings["embed_abstracts"] = time.perf_counter() - t0
print(f"embeddings shape: {embeddings.shape}")

# time: build the cosine similarity matrix
t0 = time.perf_counter()
similarity = embeddings @ embeddings.T
timings["similarity_matrix"] = time.perf_counter() - t0
print(f"similarity matrix shape: {similarity.shape}")
print(f"sanity check - diagonal should be ~1.0: {np.diag(similarity)[:3]}")

# time: save everything
t0 = time.perf_counter()
np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
np.save(OUTPUT_DIR / "similarity_matrix.npy", similarity)
df[[ID_COL]].to_csv(OUTPUT_DIR / "paper_index.csv", index=False)
timings["save_outputs"] = time.perf_counter() - t0
print("saved: embeddings.npy, similarity_matrix.npy, paper_index.csv")

# ---------- timing summary ----------
total = sum(timings.values())
print("\n=== timing summary (seconds) ===")
for step, secs in timings.items():
    print(f"  {step:<20} {secs:8.2f}")
print(f"  {'TOTAL':<20} {total:8.2f}")

# embedding throughput - a nice normalised number to report
papers_per_sec = n / timings["embed_abstracts"] if timings["embed_abstracts"] > 0 else 0
print(f"\nembedding throughput: {papers_per_sec:.1f} abstracts/second on {device}")

# save the metrics so we can put them in the report
metrics = pd.DataFrame(
    [{"step": s, "seconds": round(t, 3)} for s, t in timings.items()]
    + [{"step": "TOTAL", "seconds": round(total, 3)}]
)
metrics["n_papers"] = n
metrics["device"] = device
metrics["model"] = MODEL_NAME
metrics.to_csv(OUTPUT_DIR / "timing_embedding.csv", index=False)
print("saved timing metrics to timing_embedding.csv")