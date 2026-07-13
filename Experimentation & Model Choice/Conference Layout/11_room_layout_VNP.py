# 11_room_layout_vns.py
# Variable Neighborhood Search (VNS) for packing clusters into the rooms - a fourth
# heuristic strategy, run under the same experimental conditions as 07/08/09/10.
#
# Same problem, objective, similarity metric and scaffold. For a fair head-to-head, use
# the SAME TIME_BUDGET and SEEDS as the other scripts.
#
# Basic VNS (Mladenovic & Hansen 1997; for CCP: Brimberg et al. 2019, Lai & Hao 2016):
#   from a local optimum, "shake" with k random feasible moves to jump away, then run
#   local search (relocate + swap) to descend again. If the result improves, accept it
#   and reset k=1; otherwise shake harder (k+1). The shake strength adapts: small jumps
#   while progress is easy, larger jumps when stuck. Keep the best over the whole run.
#
# inputs:  embeddings.npy        (Poster Session\Similarity Metrics)
#          05_paper_clusters.csv (Poster Session\Hierarchical Clustering)
# outputs: 11_room_assignment.csv, 11_room_summary.csv,
#          11_seed_results.csv, 11_optimization_metrics.csv

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent          # ...\Poster Session\Conference Layout

SIM_DIR     = BASE_DIR.parent / "Similarity Metrics"
CLUSTER_DIR = BASE_DIR.parent / "Hierarchical Clustering"

EMBEDDINGS_FILE = SIM_DIR / "04 MiniLM Results" / "embeddings.npy"
CLUSTERS_FILE   = CLUSTER_DIR / "05 Clustering Results" / "05_paper_clusters.csv"
ID_COL          = "Number"

# all CSV outputs go into this sub-folder, created next to this script
OUTPUT_DIR = BASE_DIR / "11 VNS Room Layout"

ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1            # 1 -> no empty rooms; higher -> fuller rooms (keep < 300)

# --- fairness controls (MUST match 07/08/09/10) ---
TIME_BUDGET = 60.0               # seconds of search PER SEED
SEEDS       = list(range(20))      # the seeds to run (paired with the others)
# ---------------------------------------------------

K_MAX           = 10               # largest shake strength before resetting to 1
RCL_SIZE        = 3                # randomness in the initial construction
BASELINE_TRIALS = 5
BASELINE_SEED   = 42
# ----------------------------


# ---- quality metric: average within-room PAPER similarity (identical across 07-11) ----
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
caps = np.array(list(ROOM_CAPACITIES.values()))
R = len(rooms)
n_papers = len(clusters)
print(f"{n_papers} papers in {C} clusters -> {R} rooms")
print(f"experiment: {len(SEEDS)} seeds x {TIME_BUDGET}s each "
      f"(~{len(SEEDS) * TIME_BUDGET:.0f}s of search total)")

if MIN_PAPERS_PER_ROOM > int(caps.min()):
    print(f"\nINFEASIBLE: MIN_PAPERS_PER_ROOM ({MIN_PAPERS_PER_ROOM}) exceeds the smallest "
          f"room ({int(caps.min())}). Lower it.")
    raise SystemExit(1)
if MIN_PAPERS_PER_ROOM * R > n_papers:
    print(f"\nINFEASIBLE: filling all {R} rooms to {MIN_PAPERS_PER_ROOM} needs "
          f"{MIN_PAPERS_PER_ROOM * R} papers but there are only {n_papers}. Lower it.")
    raise SystemExit(1)

# --- cluster sizes, centroids, cluster-level similarity (once) ---
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
W = np.maximum(centroids @ centroids.T, 0.0)
np.fill_diagonal(W, 0.0)
timings["similarity"] = time.perf_counter() - t0

biggest = int(caps.max())
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

def local_search(assign):
    """Relocate + swap improving moves until no improvement (the descent for VNS).
    Identical engine to 08/09 so the methods differ only in how they explore."""
    loads = room_loads(assign)
    improved = True
    while improved:
        improved = False
        for i in range(C):
            r_from = assign[i]
            if loads[r_from] - size[i] < MIN_PAPERS_PER_ROOM:
                continue
            loss = W[i, np.where(assign == r_from)[0]].sum()
            for r_to in range(R):
                if r_to == r_from or loads[r_to] + size[i] > caps[r_to]:
                    continue
                if W[i, np.where(assign == r_to)[0]].sum() - loss > 1e-9:
                    assign[i] = r_to
                    loads[r_from] -= size[i]
                    loads[r_to]   += size[i]
                    improved = True
                    break
        for i in range(C):
            for j in range(i + 1, C):
                ri, rj = assign[i], assign[j]
                if ri == rj:
                    continue
                new_ri = loads[ri] - size[i] + size[j]
                new_rj = loads[rj] - size[j] + size[i]
                if new_ri > caps[ri] or new_rj > caps[rj]:
                    continue
                if new_ri < MIN_PAPERS_PER_ROOM or new_rj < MIN_PAPERS_PER_ROOM:
                    continue
                mi = np.where(assign == ri)[0]
                mj = np.where(assign == rj)[0]
                cur = W[i, mi].sum() + W[j, mj].sum()
                new = W[i, mj[mj != j]].sum() + W[j, mi[mi != i]].sum()
                if new - cur > 1e-9:
                    assign[i], assign[j] = rj, ri
                    loads[ri] = new_ri
                    loads[rj] = new_rj
                    improved = True
    return assign

def shake(assign, k):
    """Jump to a random feasible solution k moves away (k random relocates/swaps)."""
    a = assign.copy()
    ld = room_loads(a)
    moves = 0
    attempts = 0
    while moves < k and attempts < 20 * k:
        attempts += 1
        if random.random() < 0.5:                         # random relocate
            i = random.randrange(C)
            r_from = a[i]
            if ld[r_from] - size[i] < MIN_PAPERS_PER_ROOM:
                continue
            r_to = random.randrange(R)
            if r_to == r_from or ld[r_to] + size[i] > caps[r_to]:
                continue
            a[i] = r_to
            ld[r_from] -= size[i]
            ld[r_to]   += size[i]
            moves += 1
        else:                                             # random swap
            i = random.randrange(C)
            j = random.randrange(C)
            if i == j or a[i] == a[j]:
                continue
            ri, rj = a[i], a[j]
            new_ri = ld[ri] - size[i] + size[j]
            new_rj = ld[rj] - size[j] + size[i]
            if new_ri > caps[ri] or new_rj > caps[rj]:
                continue
            if new_ri < MIN_PAPERS_PER_ROOM or new_rj < MIN_PAPERS_PER_ROOM:
                continue
            a[i], a[j] = rj, ri
            ld[ri], ld[rj] = new_ri, new_rj
            moves += 1
    return a

def vns(initial, deadline):
    x = local_search(initial.copy())
    x_obj = cluster_objective(x)
    best = x.copy()
    best_obj = x_obj
    k = 1
    iters = 0
    while time.perf_counter() < deadline:
        iters += 1
        candidate = local_search(shake(x, k))            # shake at strength k, then descend
        cand_obj = cluster_objective(candidate)
        if cand_obj > x_obj + 1e-9:                       # improved -> accept, reset shake
            x = candidate
            x_obj = cand_obj
            k = 1
            if x_obj > best_obj:
                best = x.copy()
                best_obj = x_obj
        else:                                             # no gain -> shake harder
            k += 1
            if k > K_MAX:
                k = 1
    return best, best_obj, iters

def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall


# --- experiment: VNS under a time budget, across seeds ---
print(f"\nrunning VNS across {len(SEEDS)} seeds ({TIME_BUDGET}s budget each)...")
t0 = time.perf_counter()
seed_rows = []
global_best_obj = -1.0
global_best_assign = None
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    t_seed = time.perf_counter()
    deadline = t_seed + TIME_BUDGET
    init = None
    for _ in range(50):
        init = construct()
        if init is not None:
            break
    if init is None:
        print(f"  seed {seed:>2}: could not build a feasible start")
        continue
    best_assign, best_obj, iters = vns(init, deadline)
    sim = paper_similarity_of(best_assign)
    seed_rows.append({"seed": seed, "objective": round(best_obj, 4),
                      "similarity": round(sim, 4), "iterations": iters,
                      "seconds": round(time.perf_counter() - t_seed, 2)})
    print(f"  seed {seed:>2}: objective {best_obj:8.2f}  similarity {sim:.4f}  "
          f"({iters} shake/descend cycles)")
    if best_obj > global_best_obj:
        global_best_obj = best_obj
        global_best_assign = best_assign.copy()
timings["experiment"] = time.perf_counter() - t0

if global_best_assign is None:
    print("\nno seed produced a feasible solution.")
    raise SystemExit(1)

# --- distribution across seeds ---
objs = np.array([r["objective"] for r in seed_rows])
sims = np.array([r["similarity"] for r in seed_rows])
print("\n=== across seeds ===")
print(f"objective   best {objs.max():8.2f}   mean {objs.mean():8.2f}   std {objs.std():6.3f}")
print(f"similarity  best {sims.max():8.4f}   mean {sims.mean():8.4f}   std {sims.std():6.4f}")

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "11_seed_results.csv", index=False)

# --- save the BEST assignment with full per-room detail + baseline ---
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "11_room_assignment.csv", index=False)

paper_room_idx = global_best_assign[clusters_pos]
per_room_sim, overall_sim = within_room_similarity(paper_room_idx, embeddings, R)
room_sizes = [int((paper_room_idx == r).sum()) for r in range(R)]
baseline_sim = random_baseline_similarity(room_sizes, embeddings, R, BASELINE_TRIALS, BASELINE_SEED)

loads = room_loads(global_best_assign)
summary_rows = []
for r_i, r in enumerate(rooms):
    placed = [cluster_ids[i] for i in range(C) if global_best_assign[i] == r_i]
    summary_rows.append({
        "room": r, "capacity": int(caps[r_i]), "papers": int(loads[r_i]),
        "fill_pct": round(100 * loads[r_i] / caps[r_i], 1),
        "n_clusters": len(placed),
        "avg_paper_similarity": round(float(per_room_sim[r_i]), 4),
        "clusters": ", ".join(map(str, placed)),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "11_room_summary.csv", index=False)
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
    "method": "VNS",
    "time_budget_s": TIME_BUDGET, "n_seeds": len(SEEDS),
    "obj_best": round(float(objs.max()), 4), "obj_mean": round(float(objs.mean()), 4),
    "obj_std": round(float(objs.std()), 4),
    "sim_best": round(float(sims.max()), 4), "sim_mean": round(float(sims.mean()), 4),
    "sim_std": round(float(sims.std()), 4),
    "best_run_similarity": round(overall_sim, 4),
    "random_baseline_similarity": round(baseline_sim, 4),
    "k_max": K_MAX,
    "avg_iterations_per_seed": round(float(np.mean([r["iterations"] for r in seed_rows])), 1),
    "min_papers_per_room": MIN_PAPERS_PER_ROOM,
    "n_papers": n_papers, "n_clusters": C, "n_rooms": R,
}]).to_csv(OUTPUT_DIR / "11_optimization_metrics.csv", index=False)
print("\nsaved: 11_room_assignment.csv, 11_room_summary.csv, "
      "11_seed_results.csv, 11_optimization_metrics.csv")