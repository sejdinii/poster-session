# 13_compare_methods.py
# Final analysis: compare the optimisation methods (07 exact + 08-12 heuristics) on the
# paired-seed results they each wrote, and run the statistical tests.
#
# Reads <id>_seed_results.csv for each method present (whatever subset you've run), then:
#   - builds one summary table (objective + similarity best/mean/std, time)
#   - computes each heuristic's GAP to 07's proven optimum
#   - runs a Friedman omnibus test across the heuristics (objective), and
#   - pairwise Wilcoxon signed-rank tests with Holm correction
# It is paired by seed, so for it to be valid every method must have been run with the
# SAME TIME_BUDGET and SEEDS.
#
# install once:  pip install scipy
#
# outputs land in their own folder next to this script:
#   13 Heuristics Method Comparison/13_comparison_table.csv
#   13 Heuristics Method Comparison/13_pairwise_wilcoxon.csv

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    print("this script needs scipy:  pip install scipy")
    raise SystemExit(1)

BASE_DIR = Path(__file__).resolve().parent          # ...\Poster Session\Conference Layout

# all outputs go into this sub-folder, created automatically if its missing
OUT_DIR = BASE_DIR / "13 Heuristics Method Comparison"
OUT_DIR.mkdir(exist_ok=True)

# each method now writes its csvs into its own result sub-folder, so 13 reads
# each file from inside that folder
METHODS = {                                          # id -> (display name, results file)
    "07": ("CP-SAT (exact)", "07 ILP Room Layout/07_seed_results.csv"),
    "08": ("GRASP",          "08 GRASP Room Layout/08_seed_results.csv"),
    "09": ("Tabu Search",    "09 Tabu Search Room Layout/09_seed_results.csv"),
    "10": ("Matheuristic",   "10 Matheuristic Room Layout/10_seed_results.csv"),
    "11": ("VNS",            "11 VNS Room Layout/11_seed_results.csv"),
    "12": ("Memetic",        "12 Memetic Room Layout/12_seed_results.csv"),
}
HEURISTICS = ["08", "09", "10", "11", "12"]          # the stochastic methods tested among
METRIC = "objective"                                 # quantity the methods optimise (test target)
ALPHA = 0.05

# --- load whatever result files are present ---
data = {}
for key, (name, fname) in METHODS.items():
    p = BASE_DIR / fname
    if p.exists():
        df = pd.read_csv(p)
        if "seed" in df.columns:
            data[key] = df.set_index("seed").sort_index()

if not data:
    print("no seed_results.csv files found in the method result folders. run 07-12 first.")
    raise SystemExit(1)

present = list(data.keys())
print("found results for:", ", ".join(f"{k} ({METHODS[k][0]})" for k in present))
missing = [k for k in METHODS if k not in data]
if missing:
    print("not run / skipped:", ", ".join(f"{k} ({METHODS[k][0]})" for k in missing))

# --- align all methods on their shared seeds (pairing requires identical seeds) ---
common = sorted(set.intersection(*[set(df.index) for df in data.values()]))
if not common:
    print("\nthe result files share no common seeds; cannot pair them.")
    print("re-run the methods with the SAME SEEDS, then try again.")
    raise SystemExit(1)
for k in present:
    if set(data[k].index) != set(common):
        print(f"note: {k} used a different seed set; restricting to the {len(common)} shared seeds.")
    data[k] = data[k].loc[common]
print(f"comparing on {len(common)} shared seeds: {common}")

# --- optimum reference from 07 (only meaningful if optimality was proven) ---
optimum = None
opt_note = ""
if "07" in data:
    df7 = data["07"]
    if "optimal" in df7.columns:
        is_opt = df7["optimal"].astype(str).str.lower().isin(["true", "1"])
    else:
        is_opt = pd.Series(False, index=df7.index)
    if is_opt.any():
        optimum = float(df7.loc[is_opt, METRIC].max())
        opt_note = f"proven optimum from 07 ({int(is_opt.sum())}/{len(df7)} runs proved optimal)"
    else:
        optimum = float(np.nanmax(df7[METRIC].to_numpy(float)))
        opt_note = ("07 did NOT prove optimality within the budget - using its best value as an "
                    "APPROXIMATE reference (gaps below are lower bounds, not true optimality gaps)")
else:
    optimum = max(float(np.nanmax(data[k][METRIC].to_numpy(float))) for k in present)
    opt_note = "no 07 results - using the best heuristic value as reference (NOT a proven optimum)"

print(f"\nreference: {optimum:.4f}")
print(f"  ({opt_note})")

# --- summary table ---
def stats_for(key):
    df = data[key]
    obj = df[METRIC].to_numpy(float)
    obj = obj[~np.isnan(obj)]
    sim = df["similarity"].to_numpy(float) if "similarity" in df.columns else np.array([np.nan])
    sim = sim[~np.isnan(sim)] if sim.size else np.array([np.nan])
    sec = df["seconds"].to_numpy(float) if "seconds" in df.columns else np.array([np.nan])
    gap_best = (optimum - obj.max()) / optimum * 100 if optimum > 0 and obj.size else np.nan
    gap_mean = (optimum - obj.mean()) / optimum * 100 if optimum > 0 and obj.size else np.nan
    return {
        "id": key, "method": METHODS[key][0],
        "obj_best": round(float(obj.max()), 2) if obj.size else np.nan,
        "obj_mean": round(float(obj.mean()), 2) if obj.size else np.nan,
        "obj_std": round(float(obj.std()), 3) if obj.size else np.nan,
        "gap_best_%": round(float(gap_best), 3),
        "gap_mean_%": round(float(gap_mean), 3),
        "sim_mean": round(float(np.nanmean(sim)), 4) if sim.size else np.nan,
        "mean_s": round(float(np.nanmean(sec)), 2) if sec.size else np.nan,
    }

table = pd.DataFrame([stats_for(k) for k in present])
table.to_csv(OUT_DIR / "13_comparison_table.csv", index=False)

print("\n================== SUMMARY ==================")
print(table.to_string(index=False))
print("(gap_%, = how far below the reference objective; 0 means it matched the optimum)")

# --- statistical tests among the heuristics, on the objective ---
het = [k for k in HEURISTICS if k in data]
print("\n================== STATISTICAL TESTS (objective, paired by seed) ==================")
if len(het) < 2:
    print("need at least two heuristics to compare; nothing to test.")
    raise SystemExit(0)

obj_mat = {k: data[k][METRIC].to_numpy(float) for k in het}
M = np.vstack([obj_mat[k] for k in het])            # methods x seeds
per_seed_spread = np.ptp(M, axis=0)                 # range across methods, per seed
all_tied = bool(np.allclose(per_seed_spread, 0))

if all_tied:
    print("all heuristics produced the SAME objective on every shared seed.")
    print("-> there is nothing to test: no method is better than another on this instance.")
    print("   (this is a legitimate result. to make the methods separate, shorten TIME_BUDGET")
    print("    or raise the cluster count in 05, then re-run everything and this script.)")
    pd.DataFrame([{"A": METHODS[a][0], "B": METHODS[b][0], "median_obj_diff": 0.0,
                   "p_raw": None, "p_holm": None, "note": "identical on all seeds"}
                  for i, a in enumerate(het) for b in het[i + 1:]]
                 ).to_csv(OUT_DIR / "13_pairwise_wilcoxon.csv", index=False)
    print("\nsaved into: 13 Heuristics Method Comparison")
    raise SystemExit(0)

# Friedman omnibus (>= 3 methods)
if len(het) >= 3:
    try:
        chi2, p_fried = stats.friedmanchisquare(*[obj_mat[k] for k in het])
        verdict = "a significant difference" if p_fried < ALPHA else "no significant difference"
        print(f"Friedman test across {len(het)} heuristics: chi2 = {chi2:.3f}, p = {p_fried:.4g}")
        print(f"  -> {verdict} among the heuristics at alpha = {ALPHA}.")
    except Exception as e:
        p_fried = None
        print(f"Friedman test could not be computed: {e}")
else:
    p_fried = None
    print("(only two heuristics present - skipping Friedman, going straight to Wilcoxon.)")

# Pairwise Wilcoxon signed-rank, with Holm correction
pairs = [(a, b) for i, a in enumerate(het) for b in het[i + 1:]]
raw = {}
rows = []
for a, b in pairs:
    da, db = obj_mat[a], obj_mat[b]
    mask = ~(np.isnan(da) | np.isnan(db))
    da, db = da[mask], db[mask]
    diff = da - db
    if np.allclose(diff, 0):
        rows.append({"A": METHODS[a][0], "B": METHODS[b][0], "median_obj_diff": 0.0,
                     "p_raw": None, "note": "identical on all seeds"})
        continue
    try:
        _, p = stats.wilcoxon(da, db)               # zeros dropped by default
        raw[(a, b)] = p
        rows.append({"A": METHODS[a][0], "B": METHODS[b][0],
                     "median_obj_diff": round(float(np.median(diff)), 4),
                     "p_raw": p, "note": ""})
    except Exception as e:
        rows.append({"A": METHODS[a][0], "B": METHODS[b][0],
                     "median_obj_diff": round(float(np.median(diff)), 4),
                     "p_raw": None, "note": f"test failed: {e}"})

# Holm-Bonferroni adjustment over the computed p-values
holm = {}
if raw:
    ordered = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(ordered)
    prev = 0.0
    for rank, (pairkey, p) in enumerate(ordered):
        adj = max(min(1.0, (m - rank) * p), prev)
        holm[pairkey] = adj
        prev = adj

for row in rows:
    a = next(k for k in het if METHODS[k][0] == row["A"])
    b = next(k for k in het if METHODS[k][0] == row["B"])
    row["p_holm"] = round(holm[(a, b)], 4) if (a, b) in holm else None
    if row["p_raw"] is not None:
        row["p_raw"] = round(row["p_raw"], 4)

pair_table = pd.DataFrame(rows)[["A", "B", "median_obj_diff", "p_raw", "p_holm", "note"]]
pair_table.to_csv(OUT_DIR / "13_pairwise_wilcoxon.csv", index=False)

print("\npairwise Wilcoxon signed-rank (Holm-corrected):")
print(pair_table.to_string(index=False))
print(f"(median_obj_diff = median of A-B across seeds; p_holm < {ALPHA} means a real difference)")

# --- plain verdict ---
best = max(het, key=lambda k: np.nanmean(obj_mat[k]))
print(f"\nhighest mean objective: {METHODS[best][0]} ({np.nanmean(obj_mat[best]):.2f})")
sig = [r for r in rows if r.get("p_holm") is not None and r["p_holm"] < ALPHA]
if not sig:
    print("but no pairwise difference is statistically significant after correction:")
    print("the heuristics are effectively tied on this instance - pick on simplicity or speed.")
print("\nsaved into: 13 Heuristics Method Comparison")