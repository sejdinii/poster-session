# 07_room_layout.py
# Exact assignment (OR-Tools CP-SAT), run under the SAME experimental conditions as the
# heuristics 08/09 so all three can be compared:
#   - same wall-clock budget (TIME_BUDGET seconds) per run, as a solver time limit
#   - repeated across the same SEEDS (CP-SAT's internal random seed varies per run)
#
# NOTE: CP-SAT is exact and near-deterministic. If the budget is enough to PROVE
# optimality, every seed returns the same optimum (std ~ 0) and that value is the
# gap-to-optimal reference for 08/09. If the budget is too short, it returns the best
# feasible found (status FEASIBLE) and you LOSE the optimality guarantee - the per-run
# "optimal" flag tells you which happened.
#
# Objective is recomputed with the SAME function the heuristics use, so the numbers are
# on one scale (CP-SAT internally maximises an integer-scaled version).
#
# install once:  pip install ortools
#
# inputs:  embeddings.npy        (Poster Session\Similarity Metrics)
#          05_paper_clusters.csv (Poster Session\Hierarchical Clustering)
# outputs: 07_room_assignment.csv, 07_room_summary.csv,
#          07_seed_results.csv, 07_optimization_metrics.csv

from pathlib import Path
import time

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent          # ...\Poster Session\Conference Layout

SIM_DIR     = BASE_DIR.parent / "Similarity Metrics"
CLUSTER_DIR = BASE_DIR.parent / "Hierarchical Clustering"

EMBEDDINGS_FILE = SIM_DIR / "04 MiniLM Results" / "embeddings.npy"
CLUSTERS_FILE   = CLUSTER_DIR / "05 Clustering Results" / "05_paper_clusters.csv"

# all CSV outputs go into this sub-folder, created next to this script
OUTPUT_DIR = BASE_DIR / "07 ILP Room Layout"
ID_COL          = "Number"

ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1            # 1 -> no empty rooms; higher -> fuller rooms (keep < 300)

# --- fairness controls (MUST match 08 and 09) ---
TIME_BUDGET = 5.0                  # seconds PER SEED (used as the CP-SAT time limit)
SEEDS       = list(range(20))      # the seeds to run (paired with 08/09)
# -------------------------------------------------

NUM_WORKERS     = 8                # CP-SAT search threads (set to 1 for strict reproducibility)
BASELINE_TRIALS = 5
BASELINE_SEED   = 42
# ----------------------------


# ---- quality metric: average within-room PAPER similarity (identical across 07/08/09) ----
def within_room_similarity(paper_room_idx, emb, R):
    per_room = np.full(R, np.nan)
    total_sim = 0.0
    total_pairs = 0.0
    for r in range(R):
        idx = np.where(paper_room_idx == r)[0]
        nr = len(idx)
        if nr < 2:
            continue
        E = emb[idx]
        S = E.sum(axis=0)
        sim_sum = (float(S @ S) - float((E * E).sum())) / 2.0
        pairs = nr * (nr - 1) / 2.0
        per_room[r] = sim_sum / pairs
        total_sim += sim_sum
        total_pairs += pairs
    overall = total_sim / total_pairs if total_pairs > 0 else float("nan")
    return per_room, overall


def random_baseline_similarity(room_sizes, emb, R, n_trials=5, seed=42):
    rng = np.random.default_rng(seed)
    N = len(emb)
    vals = []
    for _ in range(n_trials):
        perm = rng.permutation(N)
        assign = np.empty(N, dtype=int)
        start = 0
        for r in range(R):
            assign[perm[start:start + room_sizes[r]]] = r
            start += room_sizes[r]
        _, overall = within_room_similarity(assign, emb, R)
        vals.append(overall)
    return float(np.mean(vals))


timings = {}

for f in (EMBEDDINGS_FILE, CLUSTERS_FILE):
    if not f.exists():
        print(f"could not find: {f}")
        print("check the paths in the CONFIG section at the top of this script.")
        raise SystemExit(1)

OUTPUT_DIR.mkdir(exist_ok=True)   # make the output sub-folder if it is not there

# --- load (once) ---
t0 = time.perf_counter()
clusters = pd.read_csv(CLUSTERS_FILE)
embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
timings["load_data"] = time.perf_counter() - t0
assert len(clusters) == len(embeddings), "embeddings.npy and 05_paper_clusters.csv are misaligned"

cluster_ids = sorted(clusters["cluster"].unique())
C = len(cluster_ids)
rooms = list(ROOM_CAPACITIES.keys())
caps = list(ROOM_CAPACITIES.values())
R = len(rooms)
n_papers = len(clusters)
print(f"{n_papers} papers in {C} clusters -> {R} rooms")
print(f"experiment: {len(SEEDS)} seeds x {TIME_BUDGET}s each (CP-SAT time limit per run)")

if MIN_PAPERS_PER_ROOM > min(caps):
    print(f"\nINFEASIBLE: MIN_PAPERS_PER_ROOM ({MIN_PAPERS_PER_ROOM}) exceeds the smallest "
          f"room ({min(caps)}). Lower it.")
    raise SystemExit(1)
if MIN_PAPERS_PER_ROOM * R > n_papers:
    print(f"\nINFEASIBLE: filling all {R} rooms to {MIN_PAPERS_PER_ROOM} needs "
          f"{MIN_PAPERS_PER_ROOM * R} papers but there are only {n_papers}. Lower it.")
    raise SystemExit(1)

# --- cluster sizes, centroids, similarity (once) ---
t0 = time.perf_counter()
size = []
centroids = []
for c in cluster_ids:
    mask = (clusters["cluster"] == c).to_numpy()
    size.append(int(mask.sum()))
    cen = embeddings[mask].mean(axis=0)
    centroids.append(cen / (np.linalg.norm(cen) + 1e-12))
centroids = np.array(centroids)
size = np.array(size)
timings["cluster_stats"] = time.perf_counter() - t0

t0 = time.perf_counter()
sim = centroids @ centroids.T
W = np.maximum(sim, 0.0)                  # float weights for the comparable objective
np.fill_diagonal(W, 0.0)
W_int = np.rint(W * 1000).astype(int)     # integer weights for the CP-SAT model
timings["similarity"] = time.perf_counter() - t0

biggest = max(caps)
oversized = [(cluster_ids[i], int(size[i])) for i in range(C) if size[i] > biggest]
if oversized:
    print(f"\nPROBLEM: some clusters are bigger than the largest room ({biggest} papers):")
    for cid, s in oversized:
        print(f"   cluster {cid}: {s} papers")
    print("\nFix: raise NUM_CLUSTERS in 05, re-run 05, then this.")
    raise SystemExit(1)

cluster_id_to_pos = {cid: i for i, cid in enumerate(cluster_ids)}
clusters_pos = clusters["cluster"].map(cluster_id_to_pos).to_numpy()


def cluster_objective(assign):   # same float objective as 08/09, for comparability
    total = 0.0
    for r in range(R):
        members = np.where(assign == r)[0]
        if len(members) > 1:
            total += W[np.ix_(members, members)].sum() / 2.0
    return total

def room_loads(assign):
    loads = np.zeros(R, dtype=int)
    for i in range(C):
        loads[assign[i]] += int(size[i])
    return loads

def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall

# --- build the CP-SAT model ONCE (same for every seed) ---
t0 = time.perf_counter()
model = cp_model.CpModel()
x = [[model.NewBoolVar(f"x_{c}_{r}") for r in range(R)] for c in range(C)]
room = [model.NewIntVar(0, R - 1, f"room_{c}") for c in range(C)]
for c in range(C):
    model.Add(sum(x[c][r] for r in range(R)) == 1)
    model.Add(room[c] == sum(r * x[c][r] for r in range(R)))
for r in range(R):
    papers_in_room = sum(int(size[c]) * x[c][r] for c in range(C))
    model.Add(papers_in_room <= caps[r])
    model.Add(papers_in_room >= MIN_PAPERS_PER_ROOM)
obj_terms = []
for c in range(C):
    for d in range(c + 1, C):
        if W_int[c, d] == 0:
            continue
        s = model.NewBoolVar(f"same_{c}_{d}")
        model.Add(room[c] == room[d]).OnlyEnforceIf(s)
        model.Add(room[c] != room[d]).OnlyEnforceIf(s.Not())
        obj_terms.append(W_int[c, d] * s)
model.Maximize(sum(obj_terms))
timings["build_model"] = time.perf_counter() - t0


def extract_assignment(solver):
    assign = np.empty(C, dtype=int)
    for c in range(C):
        for r in range(R):
            if solver.Value(x[c][r]) == 1:
                assign[c] = r
                break
    return assign


# --- experiment: solve under the budget, across seeds ---
print(f"\nrunning CP-SAT across {len(SEEDS)} seeds ({TIME_BUDGET}s limit each)...")
t0 = time.perf_counter()
seed_rows = []
global_best_obj = -1.0
global_best_assign = None
for seed in SEEDS:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_BUDGET
    solver.parameters.num_search_workers = NUM_WORKERS
    solver.parameters.random_seed = int(seed)
    ts = time.perf_counter()
    status = solver.Solve(model)
    solve_s = time.perf_counter() - ts
    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        seed_rows.append({"seed": seed, "objective": float("nan"), "similarity": float("nan"),
                          "status": status_name, "optimal": False, "seconds": round(solve_s, 2)})
        print(f"  seed {seed:>2}: {status_name} (no solution in budget)")
        continue
    assign = extract_assignment(solver)
    obj = cluster_objective(assign)
    sim = paper_similarity_of(assign)
    is_opt = (status == cp_model.OPTIMAL)
    seed_rows.append({"seed": seed, "objective": round(obj, 4), "similarity": round(sim, 4),
                      "status": status_name, "optimal": is_opt, "seconds": round(solve_s, 2)})
    print(f"  seed {seed:>2}: objective {obj:8.2f}  similarity {sim:.4f}  "
          f"{status_name}{'  (proven optimal)' if is_opt else ''}  [{solve_s:.1f}s]")
    if obj > global_best_obj:
        global_best_obj = obj
        global_best_assign = assign.copy()
timings["experiment"] = time.perf_counter() - t0

if global_best_assign is None:
    print("\nno seed produced a feasible solution within the budget; raise TIME_BUDGET.")
    raise SystemExit(1)

# --- distribution across seeds ---
solved = [r for r in seed_rows if not np.isnan(r["objective"])]
objs = np.array([r["objective"] for r in solved])
sims = np.array([r["similarity"] for r in solved])
n_opt = sum(1 for r in seed_rows if r["optimal"])
print("\n=== across seeds ===")
print(f"objective   best {objs.max():8.2f}   mean {objs.mean():8.2f}   std {objs.std():6.3f}")
print(f"similarity  best {sims.max():8.4f}   mean {sims.mean():8.4f}   std {sims.std():6.4f}")
print(f"proven optimal in {n_opt} of {len(SEEDS)} runs within {TIME_BUDGET}s")
if n_opt == 0:
    print("WARNING: optimality was never proven in the budget - this is NOT the true optimum.")
    print("         raise TIME_BUDGET (or run 07 once with no limit) to get the reference value.")

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "07_seed_results.csv", index=False)

# --- save the BEST assignment with full per-room detail + baseline ---
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "07_room_assignment.csv", index=False)

paper_room_idx = global_best_assign[clusters_pos]
per_room_sim, overall_sim = within_room_similarity(paper_room_idx, embeddings, R)
room_sizes = [int((paper_room_idx == r).sum()) for r in range(R)]
baseline_sim = random_baseline_similarity(room_sizes, embeddings, R, BASELINE_TRIALS, BASELINE_SEED)

loads = room_loads(global_best_assign)
summary_rows = []
for r_i, r in enumerate(rooms):
    placed = [cluster_ids[i] for i in range(C) if global_best_assign[i] == r_i]
    summary_rows.append({
        "room": r, "capacity": caps[r_i], "papers": int(loads[r_i]),
        "fill_pct": round(100 * loads[r_i] / caps[r_i], 1),
        "n_clusters": len(placed),
        "avg_paper_similarity": round(float(per_room_sim[r_i]), 4),
        "clusters": ", ".join(map(str, placed)),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "07_room_summary.csv", index=False)
timings["save_outputs"] = time.perf_counter() - t0

print("\n=== best run, room summary ===")
print(summary[["room", "capacity", "papers", "fill_pct", "n_clusters",
               "avg_paper_similarity"]].to_string(index=False))
print(f"\nbest-run average within-room paper similarity: {overall_sim:.4f}")
print(f"  random-assignment baseline (same room sizes): {baseline_sim:.4f}")
print(f"  lift over random: {overall_sim - baseline_sim:+.4f}")

total = sum(timings.values())
print("\n=== timing summary (seconds) ===")
for stage, secs in timings.items():
    print(f"  {stage:<16} {secs:8.2f}")
print(f"  {'TOTAL':<16} {total:8.2f}")

pd.DataFrame([{
    "method": "CP-SAT (exact)",
    "time_budget_s": TIME_BUDGET, "n_seeds": len(SEEDS),
    "obj_best": round(float(objs.max()), 4), "obj_mean": round(float(objs.mean()), 4),
    "obj_std": round(float(objs.std()), 4),
    "sim_best": round(float(sims.max()), 4), "sim_mean": round(float(sims.mean()), 4),
    "sim_std": round(float(sims.std()), 4),
    "best_run_similarity": round(overall_sim, 4),
    "random_baseline_similarity": round(baseline_sim, 4),
    "proven_optimal_runs": n_opt,
    "min_papers_per_room": MIN_PAPERS_PER_ROOM,
    "n_papers": n_papers, "n_clusters": C, "n_rooms": R,
}]).to_csv(OUTPUT_DIR / "07_optimization_metrics.csv", index=False)
print("\nsaved: 07_room_assignment.csv, 07_room_summary.csv, "
      "07_seed_results.csv, 07_optimization_metrics.csv")