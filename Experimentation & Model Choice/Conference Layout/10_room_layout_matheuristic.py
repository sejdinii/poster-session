# 10_room_layout_matheuristic.py
# Matheuristic (fix-and-optimize / exact large-neighborhood search) for packing clusters
# into the rooms - a hybrid that wraps the exact CP-SAT solver inside a heuristic loop.
#
# Same problem, objective, similarity metric and experiment scaffold as 07/08/09.
# For a fair head-to-head, use the SAME TIME_BUDGET and SEEDS as the other three.
#
# Idea (Gnaegi & Baumann 2021): start from a feasible solution; repeatedly LOCK most
# clusters in their current rooms and let CP-SAT re-optimise only a small FREE subset to
# optimality (respecting capacity given the locked load), keeping improvements. Because
# the current placement is always available inside each sub-solve and we accept only
# strict improvements, the solution never gets worse. Restart on stall; keep the best.
#
# install once:  pip install ortools
#
# inputs:  embeddings.npy        (Poster Session\Similarity Metrics)
#          05_paper_clusters.csv (Poster Session\Hierarchical Clustering)
# outputs: 10_room_assignment.csv, 10_room_summary.csv,
#          10_seed_results.csv, 10_optimization_metrics.csv

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent          # ...\Poster Session\Conference Layout

SIM_DIR     = BASE_DIR.parent / "Similarity Metrics"
CLUSTER_DIR = BASE_DIR.parent / "Hierarchical Clustering"

EMBEDDINGS_FILE = SIM_DIR / "04 MiniLM Results" / "embeddings.npy"
CLUSTERS_FILE   = CLUSTER_DIR / "05 Clustering Results" / "05_paper_clusters.csv"
ID_COL          = "Number"

# all CSV outputs go into this sub-folder, created next to this script
OUTPUT_DIR = BASE_DIR / "10 Matheuristic Room Layout"

ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1            # 1 -> no empty rooms; higher -> fuller rooms (keep < 300)

# --- fairness controls (MUST match 07/08/09) ---
TIME_BUDGET = 60.0               # seconds of search PER SEED
SEEDS       = list(range(20))      # the seeds to run (paired with the others)
# ------------------------------------------------

FREE_SIZE       = 12               # clusters released for the exact re-optimisation each step
STALL_ITERS     = 30               # restart a trajectory after this many steps with no gain
SUBSOLVE_TIME   = 2.0              # safety cap (s) for each tiny exact sub-solve
RCL_SIZE        = 3                # randomness in the initial construction
BASELINE_TRIALS = 5
BASELINE_SEED   = 42
# ----------------------------


# ---- quality metric: average within-room PAPER similarity (identical across 07/08/09/10) ----
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
print(f"experiment: {len(SEEDS)} seeds x {TIME_BUDGET}s each "
      f"(~{len(SEEDS) * TIME_BUDGET:.0f}s of search total)")
print(f"freeing {FREE_SIZE} clusters per exact re-optimisation step")

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
size = np.zeros(C, dtype=int)
centroids = np.zeros((C, embeddings.shape[1]), dtype=np.float32)
for i, c in enumerate(cluster_ids):
    mask = (clusters["cluster"] == c).to_numpy()
    size[i] = int(mask.sum())
    cen = embeddings[mask].mean(axis=0)
    centroids[i] = cen / (np.linalg.norm(cen) + 1e-12)
timings["cluster_stats"] = time.perf_counter() - t0

t0 = time.perf_counter()
sim = centroids @ centroids.T
W = np.maximum(sim, 0.0)                  # float weights (objective + acceptance)
np.fill_diagonal(W, 0.0)
W_int = np.rint(W * 1000).astype(int)     # integer weights for the CP-SAT sub-model
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


# --- shared helpers ---
def cluster_objective(assign):
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

def construct():
    assign = np.full(C, -1, dtype=int)
    loads = np.zeros(R, dtype=int)
    order = sorted(range(C), key=lambda k: -size[k])
    for r in range(R):
        while loads[r] < MIN_PAPERS_PER_ROOM:
            fitting = [i for i in order if assign[i] == -1 and loads[r] + size[i] <= caps[r]]
            if not fitting:
                return None
            i = random.choice(fitting[:RCL_SIZE])
            assign[i] = r
            loads[r] += int(size[i])
    for i in order:
        if assign[i] != -1:
            continue
        feasible = [r for r in range(R) if loads[r] + size[i] <= caps[r]]
        if not feasible:
            return None
        scored = []
        for r in feasible:
            members = np.where(assign == r)[0]
            gain = W[i, members].sum() if len(members) > 0 else 0.0
            scored.append((gain, r))
        scored.sort(reverse=True)
        _, chosen = random.choice(scored[:RCL_SIZE])
        assign[i] = chosen
        loads[chosen] += int(size[i])
    return assign

def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall

def fix_and_optimize(assign, F, deadline):
    """Lock all clusters except F; let CP-SAT re-place F optimally given the locked load."""
    Fset = set(F)
    locked_load = np.zeros(R, dtype=int)
    fixed_in_room = [[] for _ in range(R)]
    for d in range(C):
        if d in Fset:
            continue
        rr = assign[d]
        locked_load[rr] += int(size[d])
        fixed_in_room[rr].append(d)

    m = cp_model.CpModel()
    y = {c: [m.NewBoolVar(f"y_{c}_{r}") for r in range(R)] for c in F}
    room_f = {c: m.NewIntVar(0, R - 1, f"rf_{c}") for c in F}
    for c in F:
        m.Add(sum(y[c][r] for r in range(R)) == 1)
        m.Add(room_f[c] == sum(r * y[c][r] for r in range(R)))
        for r in range(R):
            m.AddHint(y[c][r], 1 if assign[c] == r else 0)   # warm-start from current
    for r in range(R):
        load_r = locked_load[r] + sum(int(size[c]) * y[c][r] for c in F)
        m.Add(load_r <= caps[r])
        m.Add(load_r >= MIN_PAPERS_PER_ROOM)

    obj = []
    for c in F:                              # free <-> fixed attraction (linear)
        for r in range(R):
            if fixed_in_room[r]:
                coef = int(W_int[c, fixed_in_room[r]].sum())
                if coef:
                    obj.append(coef * y[c][r])
    Fl = list(F)                             # free <-> free (reified same-room)
    for a in range(len(Fl)):
        for b in range(a + 1, len(Fl)):
            c, d = Fl[a], Fl[b]
            if W_int[c, d] == 0:
                continue
            s = m.NewBoolVar(f"s_{c}_{d}")
            m.Add(room_f[c] == room_f[d]).OnlyEnforceIf(s)
            m.Add(room_f[c] != room_f[d]).OnlyEnforceIf(s.Not())
            obj.append(int(W_int[c, d]) * s)
    m.Maximize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(0.05, min(SUBSOLVE_TIME, deadline - time.perf_counter()))
    st = solver.Solve(m)
    new = assign.copy()
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for c in F:
            for r in range(R):
                if solver.Value(y[c][r]) == 1:
                    new[c] = r
                    break
    return new


# --- experiment: multi-start fix-and-optimize under a time budget, across seeds ---
print(f"\nrunning the matheuristic across {len(SEEDS)} seeds ({TIME_BUDGET}s budget each)...")
t0 = time.perf_counter()
seed_rows = []
global_best_obj = -1.0
global_best_assign = None
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    seed_best_obj = -1.0
    seed_best_assign = None
    iters = 0
    restarts = 0
    t_seed = time.perf_counter()
    deadline = t_seed + TIME_BUDGET
    while time.perf_counter() < deadline:
        cur = None
        for _ in range(50):
            cur = construct()
            if cur is not None:
                break
        if cur is None:
            break
        cur_obj = cluster_objective(cur)
        restarts += 1
        stall = 0
        if cur_obj > seed_best_obj:
            seed_best_obj = cur_obj
            seed_best_assign = cur.copy()
        while time.perf_counter() < deadline and stall < STALL_ITERS:
            F = random.sample(range(C), min(FREE_SIZE, C))
            new = fix_and_optimize(cur, F, deadline)
            iters += 1
            new_obj = cluster_objective(new)
            if new_obj > cur_obj + 1e-9:
                cur = new
                cur_obj = new_obj
                stall = 0
                if cur_obj > seed_best_obj:
                    seed_best_obj = cur_obj
                    seed_best_assign = cur.copy()
            else:
                stall += 1
    sim = paper_similarity_of(seed_best_assign)
    seed_rows.append({"seed": seed, "objective": round(seed_best_obj, 4),
                      "similarity": round(sim, 4), "restarts": restarts,
                      "subsolves": iters, "seconds": round(time.perf_counter() - t_seed, 2)})
    print(f"  seed {seed:>2}: objective {seed_best_obj:8.2f}  similarity {sim:.4f}  "
          f"({restarts} restarts, {iters} sub-solves)")
    if seed_best_obj > global_best_obj:
        global_best_obj = seed_best_obj
        global_best_assign = seed_best_assign.copy()
timings["experiment"] = time.perf_counter() - t0

# --- distribution across seeds ---
objs = np.array([r["objective"] for r in seed_rows])
sims = np.array([r["similarity"] for r in seed_rows])
print("\n=== across seeds ===")
print(f"objective   best {objs.max():8.2f}   mean {objs.mean():8.2f}   std {objs.std():6.3f}")
print(f"similarity  best {sims.max():8.4f}   mean {sims.mean():8.4f}   std {sims.std():6.4f}")

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "10_seed_results.csv", index=False)

# --- save the BEST assignment with full per-room detail + baseline ---
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "10_room_assignment.csv", index=False)

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
summary.to_csv(OUTPUT_DIR / "10_room_summary.csv", index=False)
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
    "method": "Matheuristic (fix-and-optimize)",
    "time_budget_s": TIME_BUDGET, "n_seeds": len(SEEDS),
    "obj_best": round(float(objs.max()), 4), "obj_mean": round(float(objs.mean()), 4),
    "obj_std": round(float(objs.std()), 4),
    "sim_best": round(float(sims.max()), 4), "sim_mean": round(float(sims.mean()), 4),
    "sim_std": round(float(sims.std()), 4),
    "best_run_similarity": round(overall_sim, 4),
    "random_baseline_similarity": round(baseline_sim, 4),
    "free_size": FREE_SIZE,
    "avg_subsolves_per_seed": round(float(np.mean([r["subsolves"] for r in seed_rows])), 1),
    "min_papers_per_room": MIN_PAPERS_PER_ROOM,
    "n_papers": n_papers, "n_clusters": C, "n_rooms": R,
}]).to_csv(OUTPUT_DIR / "10_optimization_metrics.csv", index=False)
print("\nsaved: 10_room_assignment.csv, 10_room_summary.csv, "
      "10_seed_results.csv, 10_optimization_metrics.csv")