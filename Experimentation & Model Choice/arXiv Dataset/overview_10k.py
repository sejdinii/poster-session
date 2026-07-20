# dataset_statistics.py
# the numbers chapter 4 needs about the arxiv demonstration dataset: how many
# papers, whats missing, how long the abstracts are, and how many papers per
# group. lives next to sample_arxiv.py, reads from Dataset\ and writes into
# "Dataset Statistics (For the Paper Chapter)".
#
# run it once, then send the txt and the png.

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no window needed, we only save the png
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent          # ...\arXiv Dataset
INPUT = BASE_DIR / "Dataset" / "papers_dataset_random_10K.csv"
PARENT = BASE_DIR / "Dataset" / "papers_dataset.csv"   # the big csv the sample came from
OUT_DIR = BASE_DIR / "Dataset Statistics (For the Paper Chapter)"
OUT_DIR.mkdir(exist_ok=True)                        # already exists, thats fine

if not INPUT.exists():
    print("cannot find the dataset csv at:")
    print(f"  {INPUT}")
    print("if your file has a different name, change INPUT at the top.")
    raise SystemExit(1)

df = pd.read_csv(INPUT)

# abstract length in words, per paper. empty abstracts count as 0 words
abs_words = df["abstract"].fillna("").astype(str).str.split().str.len()

lines = []
lines.append(f"file: {INPUT.name}")
lines.append(f"papers (rows): {len(df)}")
lines.append(f"columns: {', '.join(df.columns)}")
lines.append("")
lines.append("missing values per column:")
for col, n in df.isna().sum().items():
    lines.append(f"  {col}: {n}")
lines.append("")
lines.append("abstract length in words:")
lines.append(f"  mean: {abs_words.mean():.1f}")
lines.append(f"  min:  {abs_words.min()}")
lines.append(f"  max:  {abs_words.max()}")
lines.append("")
lines.append("papers per group:")
counts = df["group"].fillna("(missing)").value_counts()
for group, n in counts.items():
    lines.append(f"  {group}: {n}")

# parent set size, the csv the 10k sample was drawn from
# read with pandas and not a line count because abstracts contain newlines
lines.append("")
if PARENT.exists():
    parent = pd.read_csv(PARENT, usecols=["group"])
    lines.append(f"parent set ({PARENT.name}): {len(parent)} papers")
else:
    lines.append(f"parent set: {PARENT.name} not found in Dataset\\, count skipped")

report = "\n".join(lines)
(OUT_DIR / "dataset_stats.txt").write_text(report, encoding="utf-8")
print(report)

# bar chart of papers per group, sorted, biggest on top so its readable
fig_h = max(4, 0.32 * len(counts))                  # taller when there are many groups
fig, ax = plt.subplots(figsize=(9, fig_h))
counts.sort_values().plot.barh(ax=ax, color="gray")
ax.set_xlabel("papers")
ax.set_title("Sampled arXiv papers per group")
fig.tight_layout()
fig.savefig(OUT_DIR / "papers_per_group.png", dpi=200)

print(f"\nsaved: {OUT_DIR / 'dataset_stats.txt'}")
print(f"saved: {OUT_DIR / 'papers_per_group.png'}")