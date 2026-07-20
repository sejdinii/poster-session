# dataset_statistics.py
# the numbers chapter 4 needs about the dataset itself: how many papers,
# whats missing, how long the abstracts are, and how many papers per area.
# lives next to dataset_API.py, reads from Dataset\ and writes into
# "Dataset Statistics (For the Paper Chapter)".
#
# run it once, then send the txt and the png.

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no window needed, we only save the png
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent          # ...\Neurips Dataset
INPUT = BASE_DIR / "Dataset" / "neurips_accepted_submissions.csv"
OUT_DIR = BASE_DIR / "Dataset Statistics (For the Paper Chapter)"
OUT_DIR.mkdir(exist_ok=True)                        # already exists, thats fine

if not INPUT.exists():
    print("cannot find the dataset csv at:")
    print(f"  {INPUT}")
    print("if your file has a different name, change INPUT at the top.")
    raise SystemExit(1)

df = pd.read_csv(INPUT)

# abstract length in words, per paper. empty abstracts count as 0 words
abs_words = df["Abstract"].fillna("").astype(str).str.split().str.len()

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
lines.append("papers per primary area:")
counts = df["Primary Area"].fillna("(missing)").value_counts()
for area, n in counts.items():
    lines.append(f"  {area}: {n}")

report = "\n".join(lines)
(OUT_DIR / "dataset_stats.txt").write_text(report, encoding="utf-8")
print(report)

# bar chart of papers per area, sorted, biggest on top so its readable
fig_h = max(4, 0.32 * len(counts))                  # taller when there are many areas
fig, ax = plt.subplots(figsize=(9, fig_h))
counts.sort_values().plot.barh(ax=ax, color="gray")
ax.set_xlabel("papers")
ax.set_title("Accepted NeurIPS 2025 papers per primary area")
fig.tight_layout()
fig.savefig(OUT_DIR / "papers_per_area.png", dpi=200)

print(f"\nsaved: {OUT_DIR / 'dataset_stats.txt'}")
print(f"saved: {OUT_DIR / 'papers_per_area.png'}")