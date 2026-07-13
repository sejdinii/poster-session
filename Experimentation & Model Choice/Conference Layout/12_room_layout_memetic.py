# 12_room_layout_memetic.py
# Memetic (evolutionary + local search) algorithm for packing clusters into the rooms -
# the fifth heuristic strategy, run under the same experimental conditions as 07-11.
#
# Same problem, objective, similarity metric and scaffold. For a fair head-to-head, use
# the SAME TIME_BUDGET and SEEDS as the other scripts.
#
# Memetic algorithm (Moscato 1989; for CCP: Zhou et al. 2019): evolve a POPULATION of
# solutions. Each generation: pick two parents by tournament, RECOMBINE them into a
# child (capacity-respecting crossover), MUTATE it slightly, then improve it with local
# search (the "memetic" step) before it competes to replace the worst member. Keep the
# best solution seen.
#
# inputs:  embeddings.npy        (Poster Session\Similarity Metrics)
#          05_paper_clusters.csv (Poster Session\Hierarchical Clustering)
# outputs: 12_room_assignment.csv, 12_room_summary.csv,
#          12_seed_results.csv, 12_optimization_metrics.csv

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
OUTPUT_DIR = BASE_DIR / "12 Memetic Room Layout"

ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1            # 1 -> no empty rooms; higher -> fuller rooms (keep < 300)

# --- fairness controls (MUST match 07/08/09/10/11) ---
TIME_BUDGET = 60.0               # seconds of search PER SEED
SEEDS       = list(range(20))      # the seeds to run (paired with the others)
# ------------------------------------------------------

POP_SIZE        = 20               # number of solutions kept in the population
TOURNAMENT_K    = 3                # candidates compared to pick each parent
MUTATION_RATE   = 0.3              # chance a child is mutated
MUTATION_MOVES  = 2                # random moves applied when a child mutates
RCL_SIZE        = 3                # randomness in the initial construction
BASELINE_TRIALS = 5
BASELINE_SEED   = 42
# ----------------------------


# ---- quality metric: average within-room PAPER similarity (identical across 07-12) ----
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
print(f"population size {POP_SIZE}")

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

def is_feasible(assign):
    loads = room_loads(assign)
    return bool(np.all(loads <= caps) and np.all(loads >= MIN_PAPERS_PER_ROOM))

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
    """Relocate + swap improving moves until no improvement (same engine as 08/09/11)."""
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

def fix_min(child, loads):
    for r in range(R):
        guard = 0
        while loads[r] < MIN_PAPERS_PER_ROOM and guard < C:
            guard += 1
            donors = [c for c in range(C)
                      if child[c] != r
                      and loads[child[c]] - size[c] >= MIN_PAPERS_PER_ROOM
                      and loads[r] + size[c] <= caps[r]]
            if not donors:
                break
            c = min(donors, key=lambda c: size[c])
            old = child[c]
            child[c] = r
            loads[old] -= int(size[c])
            loads[r]   += int(size[c])
    return child

def crossover(pa, pb):
    """Capacity-respecting uniform crossover: each cluster inherits a parent's room when
    it fits (preserving co-locations), else falls back to a room with room to spare."""
    child = np.full(C, -1, dtype=int)
    loads = np.zeros(R, dtype=int)
    order = sorted(range(C), key=lambda k: -size[k])
    for c in order:
        options = [int(pa[c]), int(pb[c])]
        random.shuffle(options)
        placed = False
        for r in options:
            if loads[r] + size[c] <= caps[r]:
                child[c] = r
                loads[r] += int(size[c])
                placed = True
                break
        if not placed:
            feasible = [r for r in range(R) if loads[r] + size[c] <= caps[r]]
            r = max(feasible, key=lambda r: caps[r] - loads[r]) if feasible \
                else int(np.argmax(caps - loads))
            child[c] = r
            loads[r] += int(size[c])
    fix_min(child, loads)
    return child

def make_child(pa, pb):
    child = crossover(pa, pb)
    if not is_feasible(child):                 # safety net (does not trigger at MIN=1)
        child = (pa if cluster_objective(pa) >= cluster_objective(pb) else pb).copy()
    return child

def perturb(assign, n_moves):
    a = assign.copy()
    ld = room_loads(a)
    moves = 0
    attempts = 0
    while moves < n_moves and attempts < 20 * n_moves:
        attempts += 1
        if random.random() < 0.5:
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
        else:
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

def tournament(pop):
    idxs = random.sample(range(len(pop)), min(TOURNAMENT_K, len(pop)))
    best_i = max(idxs, key=lambda i: pop[i][0])
    return pop[best_i][1]

def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall


# --- experiment: memetic algorithm under a time budget, across seeds ---
print(f"\nrunning the memetic algorithm across {len(SEEDS)} seeds ({TIME_BUDGET}s budget each)...")
t0 = time.perf_counter()
seed_rows = []
global_best_obj = -1.0
global_best_assign = None
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    t_seed = time.perf_counter()
    deadline = t_seed + TIME_BUDGET

    # initial population (each member improved by local search = the memetic step)
    pop = []
    for _ in range(POP_SIZE):
        if time.perf_counter() >= deadline:
            break
        a = None
        for _ in range(50):
            a = construct()
            if a is not None:
                break
        if a is None:
            continue
        a = local_search(a)
        pop.append([cluster_objective(a), a])
    if not pop:
        print(f"  seed {seed:>2}: could not build a population")
        continue

    seed_best_idx = max(range(len(pop)), key=lambda i: pop[i][0])
    seed_best_obj = pop[seed_best_idx][0]
    seed_best_assign = pop[seed_best_idx][1].copy()

    generations = 0
    while time.perf_counter() < deadline and len(pop) >= 2:
        generations += 1
        pa = tournament(pop)
        pb = tournament(pop)
        child = make_child(pa, pb)
        if random.random() < MUTATION_RATE:
            child = perturb(child, MUTATION_MOVES)
        child = local_search(child)
        obj = cluster_objective(child)
        worst_i = min(range(len(pop)), key=lambda i: pop[i][0])
        if obj > pop[worst_i][0]:
            pop[worst_i] = [obj, child]
        if obj > seed_best_obj:
            seed_best_obj = obj
            seed_best_assign = child.copy()

    sim = paper_similarity_of(seed_best_assign)
    seed_rows.append({"seed": seed, "objective": round(seed_best_obj, 4),
                      "similarity": round(sim, 4), "generations": generations,
                      "seconds": round(time.perf_counter() - t_seed, 2)})
    print(f"  seed {seed:>2}: objective {seed_best_obj:8.2f}  similarity {sim:.4f}  "
          f"({generations} generations)")
    if seed_best_obj > global_best_obj:
        global_best_obj = seed_best_obj
        global_best_assign = seed_best_assign.copy()
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

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "12_seed_results.csv", index=False)

# --- save the BEST assignment with full per-room detail + baseline ---
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "12_room_assignment.csv", index=False)

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
summary.to_csv(OUTPUT_DIR / "12_room_summary.csv", index=False)
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
    "method": "Memetic",
    "time_budget_s": TIME_BUDGET, "n_seeds": len(SEEDS),
    "obj_best": round(float(objs.max()), 4), "obj_mean": round(float(objs.mean()), 4),
    "obj_std": round(float(objs.std()), 4),
    "sim_best": round(float(sims.max()), 4), "sim_mean": round(float(sims.mean()), 4),
    "sim_std": round(float(sims.std()), 4),
    "best_run_similarity": round(overall_sim, 4),
    "random_baseline_similarity": round(baseline_sim, 4),
    "pop_size": POP_SIZE,
    "avg_generations_per_seed": round(float(np.mean([r["generations"] for r in seed_rows])), 1),
    "min_papers_per_room": MIN_PAPERS_PER_ROOM,
    "n_papers": n_papers, "n_clusters": C, "n_rooms": R,
}]).to_csv(OUTPUT_DIR / "12_optimization_metrics.csv", index=False)
print("\nsaved: 12_room_assignment.csv, 12_room_summary.csv, "
      "12_seed_results.csv, 12_optimization_metrics.csv")