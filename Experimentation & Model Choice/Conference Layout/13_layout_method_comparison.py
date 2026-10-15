#final analysis, script 13 of the layout benchmark suite
#compares the optimisation methods (07 exact + 08 to 12 heuristics) on the
#paired-seed results each one wrote, and runs the statistical tests

#what it does in plain words:
#reads <id>_seed_results.csv for every method thats been run (any subset works), then
#builds one summary table (objective + similarity best/mean/std, time),
#computes each heuristics GAP to 07s proven optimum,
#runs a friedman omnibus test across the heuristics on the objective,
#and pairwise wilcoxon signed-rank tests with holm correction
#everything is paired by seed, so for the pairing to be valid every method must
#have been run with the SAME TIME_BUDGET and SEEDS

#the testing protocol, one rank-based omnibus test first and corrected pairwise
#tests after, is the standard way of comparing randomised algorithms over shared
#runs (demsar 2006, garcia & herrera 2008)
#the three tests themselves: friedman 1937, wilcoxon 1945, holm 1979
#rank-based tests are used since nothing guarantees the objectives are normally
#distributed, and ranks dont care about that

#install once:  pip install scipy

#outputs land in their own folder next to this script:
#13 Heuristics Method Comparison/13_comparison_table.csv
#13 Heuristics Method Comparison/13_pairwise_wilcoxon.csv

#libraries used in this script and where each one comes from:
#pathlib is python standard library, docs.python.org
#numpy does the array math, harris et al. 2020, github.com/numpy/numpy
#pandas reads and writes the csvs, mckinney 2010, github.com/pandas-dev/pandas
#scipy provides the friedman and wilcoxon tests, virtanen et al. 2020,
#github.com/scipy/scipy, stats docs at docs.scipy.org

from pathlib import Path
import numpy as np
import pandas as pd

#scipy is the one non-standard dependency here, so a failed import gets its own
#clear message instead of a confusing bare traceback
try:
    from scipy import stats
except ImportError:
    print("this script needs scipy:  pip install scipy")
    raise SystemExit(1)

#anchor everything to this scripts own location instead of wherever the
#terminal happens to be standing
BASE_DIR = Path(__file__).resolve().parent  #...\Conference Layout

#all outputs go into this sub-folder, created automatically if its missing
OUT_DIR = BASE_DIR / "13 Heuristics Method Comparison"
OUT_DIR.mkdir(exist_ok=True)

#each method wrote its csvs into its own result sub-folder, this maps the
#method id to a display name and to where its seed results file lives
METHODS = {                                          #id -> (display name, results file)
    "07": ("CP-SAT (exact)", "07 ILP Room Layout/07_seed_results.csv"),
    "08": ("GRASP",          "08 GRASP Room Layout/08_seed_results.csv"),
    "09": ("Tabu Search",    "09 Tabu Search Room Layout/09_seed_results.csv"),
    "10": ("Matheuristic",   "10 Matheuristic Room Layout/10_seed_results.csv"),
    "11": ("VNS",            "11 VNS Room Layout/11_seed_results.csv"),
    "12": ("Memetic",        "12 Memetic Room Layout/12_seed_results.csv"),
}
HEURISTICS = ["08", "09", "10", "11", "12"]          #the stochastic methods the tests compare, 07 is exact so it stays out
METRIC = "objective"                                 #the quantity the methods optimise, the tests run on this column
ALPHA = 0.05                                         #significance level, the usual 5 percent

#LOADING
#load whatever result files are present, a method that hasnt been run is
#simply skipped, so this works on any subset of 07 to 12
#set_index("seed") is what lets rows line up seed against seed across methods
data = {}
for key, (name, fname) in METHODS.items():
    p = BASE_DIR / fname
    if p.exists():
        df = pd.read_csv(p)
        if "seed" in df.columns:
            data[key] = df.set_index("seed").sort_index()

#nothing found at all means none of 07 to 12 have been run yet
if not data:
    print("no seed_results.csv files found in the method result folders. run 07-12 first.")
    raise SystemExit(1)

present = list(data.keys())
print("found results for:", ", ".join(f"{k} ({METHODS[k][0]})" for k in present))
missing = [k for k in METHODS if k not in data]
if missing:
    print("not run / skipped:", ", ".join(f"{k} ({METHODS[k][0]})" for k in missing))

#ALIGNING ON SHARED SEEDS
#the tests below are paired, seed 3 of one method against seed 3 of another,
#the blocked design demsar 2006 describes, and pairing only works when the
#methods share seeds, so everything gets restricted to the common set
#set.intersection across all the index sets finds the seeds every method has
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

#THE REFERENCE VALUE
#gaps need something to be gaps from
#if 07 proved optimality at least once, that proven value is the true optimum
#if 07 ran but never proved it, its best value is only an approximate reference
#and the note printed below says so, the gaps are then lower bounds not true gaps
#if 07 is missing entirely, the best heuristic value stands in (weakest case)
optimum = None
opt_note = ""
if "07" in data:
    df7 = data["07"]
    #the optimal column can arrive as booleans or as text depending on how
    #pandas read the csv, lowercasing and matching both forms covers it
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

#SUMMARY TABLE
#one row per method, best/mean/std of the objective, the gap to the reference
#in percent, mean similarity and mean seconds
#the nans are stripped first so a failed run doesnt poison the averages
#gap = how far below the reference in percent, 0 means the reference was matched
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

#STATISTICAL TESTS
#run on the objective, among the heuristics only, paired by seed
#07 stays out since an exact solver isnt a random method, theres nothing to test
het = [k for k in HEURISTICS if k in data]
print("\n================== STATISTICAL TESTS (objective, paired by seed) ==================")
if len(het) < 2:
    print("need at least two heuristics to compare; nothing to test.")
    raise SystemExit(0)

#M stacks the objectives into a methods x seeds grid
#np.ptp is peak to peak, the range across the methods for each single seed
#if that range is zero on every seed, all methods scored identically everywhere
obj_mat = {k: data[k][METRIC].to_numpy(float) for k in het}
M = np.vstack([obj_mat[k] for k in het])            #methods x seeds grid
per_seed_spread = np.ptp(M, axis=0)                 #range across methods, per seed
all_tied = bool(np.allclose(per_seed_spread, 0))

#identical results everywhere are a legitimate outcome, not an error
#the tests would have nothing to work with, so the script says so plainly
#and still writes the pairwise file, so the output files always exist
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

#FRIEDMAN OMNIBUS TEST (friedman 1937)
#one test across all the heuristics at once, run before any pairwise digging,
#which is the order the demsar 2006 protocol prescribes
#it ranks the methods within each seed and asks whether the average ranks differ
#more than chance would allow, and it needs at least 3 methods to make sense
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

#PAIRWISE WILCOXON SIGNED-RANK TESTS (wilcoxon 1945)
#every pair of heuristics, tested on the per-seed differences
#signed-rank uses the sizes of the differences, not just their signs, and being
#paired it compares like with like, seed 3 against seed 3
#a pair thats identical on every seed gets recorded with no p value, since the
#test needs some variation to work on
#the nan mask drops any seed missing in either method of the pair
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
        _, p = stats.wilcoxon(da, db)               #zero differences get dropped by scipys default handling
        raw[(a, b)] = p
        rows.append({"A": METHODS[a][0], "B": METHODS[b][0],
                     "median_obj_diff": round(float(np.median(diff)), 4),
                     "p_raw": p, "note": ""})
    except Exception as e:
        rows.append({"A": METHODS[a][0], "B": METHODS[b][0],
                     "median_obj_diff": round(float(np.median(diff)), 4),
                     "p_raw": None, "note": f"test failed: {e}"})

#HOLM-BONFERRONI CORRECTION (holm 1979)
#running ten tests at 5 percent each inflates the chance of a fluke finding,
#holm fixes that by sorting the p values smallest first and multiplying each
#by how many tests remain from its position, thats the (m - rank) factor
#the max(..., prev) keeps the adjusted values monotone, a later p is never
#allowed to end up smaller than an earlier one, which is part of holms procedure
#min(1.0, ...) caps everything at 1 since a probability cant go past it
holm = {}
if raw:
    ordered = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(ordered)
    prev = 0.0
    for rank, (pairkey, p) in enumerate(ordered):
        adj = max(min(1.0, (m - rank) * p), prev)
        holm[pairkey] = adj
        prev = adj

#stitch the corrected p values back onto their rows, the method ids are looked
#up from the display names since the rows only carry the names
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

#PLAIN VERDICT
#the highest mean objective gets named, and if no pairwise difference survived
#the correction the honest reading is a tie, and the tiebreak is simplicity or speed
best = max(het, key=lambda k: np.nanmean(obj_mat[k]))
print(f"\nhighest mean objective: {METHODS[best][0]} ({np.nanmean(obj_mat[best]):.2f})")
sig = [r for r in rows if r.get("p_holm") is not None and r["p_holm"] < ALPHA]
if not sig:
    print("but no pairwise difference is statistically significant after correction:")
    print("the heuristics are effectively tied on this instance - pick on simplicity or speed.")
print("\nsaved into: 13 Heuristics Method Comparison")