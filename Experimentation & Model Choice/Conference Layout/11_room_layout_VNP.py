#variable neighborhood search (vns) for packing clusters into the rooms,
#script 11 of the layout benchmark suite, a fourth heuristic strategy
#run under the same experimental conditions as 07, 08, 09 and 10

#same problem, objective, similarity metric and scaffold
#for a fair head-to-head the TIME_BUDGET and the SEEDS must match the other scripts

#the method, basic vns (mladenovic & hansen 1997, hansen & mladenovic 2001,
#and the overview in hansen et al. 2017):
#from a local optimum, "shake" with k random legal moves to jump away, then run
#local search (relocate + swap) to descend again
#if the result improves, accept it and reset k=1, otherwise shake harder (k+1)
#so the shake strength adapts, small jumps while progress is easy, larger jumps
#when stuck, and the best layout over the whole run is kept

#the one idea behind all of it: a plain local search stops at the first dead end,
#and a shake is what throws the layout far enough away that descending again
#can land somewhere new, k is how far away

#vns applied to the capacitated clustering problem, the family this problem
#belongs to: brimberg, mladenovic, todosijevic & urosevic 2019 and lai & hao 2016

#the construction that builds each start is the randomized greedy of grasp
#(feo & resende 1995), placing big clusters first, the classic decreasing-order
#idea from bin packing (johnson et al. 1974)

#one thing worth knowing about heuristics like this: they have no natural
#finishing point, the search would just go forever, so the stop is the clock
#thats why TIME_BUDGET exists

#inputs: embeddings.npy from the Similarity Metrics folder (made by 04)
#05_paper_clusters.csv from the Hierarchical Clustering folder (made by 05)
#outputs: 11_room_assignment.csv, 11_room_summary.csv,
#11_seed_results.csv, 11_optimization_metrics.csv

#libraries used in this script and where each one comes from:
#pathlib, time and random are python standard library, docs.python.org
#numpy does the vector math, harris et al. 2020, github.com/numpy/numpy
#pandas reads and writes the csvs, mckinney 2010, github.com/pandas-dev/pandas

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd

#CONFIG
#everything adjustable lives up here so nothing needs hunting deeper in the file

#anchor everything to this scripts own location instead of wherever the
#terminal happens to be standing, __file__ is this scripts own file,
#resolve() makes it a full path and parent is the folder its sitting in
BASE_DIR = Path(__file__).resolve().parent  #...\Conference Layout

#the two input folders are siblings of this scripts folder
SIM_DIR     = BASE_DIR.parent / "Similarity Metrics"
CLUSTER_DIR = BASE_DIR.parent / "Hierarchical Clustering"

EMBEDDINGS_FILE = SIM_DIR / "04 MiniLM Results" / "embeddings.npy"
CLUSTERS_FILE   = CLUSTER_DIR / "05 Clustering Results" / "05_paper_clusters.csv"
ID_COL          = "Number"  #the column that identifies a paper in the clusters csv

#all csv outputs go into this sub-folder, created next to this script
OUTPUT_DIR = BASE_DIR / "11 VNS Room Layout"

#the twelve real rooms of the conference
#the capacities add up to exactly 5286, one seat per paper, zero slack
ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1  #1 means no empty rooms, higher forces fuller rooms (keep it below the smallest room, 300)

#fairness controls, these MUST match scripts 07 to 10 and the rest of the suite
#every method gets the same budget and the same seeds so the statistical
#tests in 13 can pair attempt against attempt
#60 seconds is the budget the real results were produced with
TIME_BUDGET = 60.0  #seconds of search PER SEED
SEEDS       = list(range(20))  #the seeds to run (paired with the others)

#K_MAX is how far the shake is allowed to escalate before dropping back to 1,
#the neighborhood ladder that gives vns its name (mladenovic & hansen 1997)
K_MAX           = 10  #largest shake strength before resetting to 1
RCL_SIZE        = 3  #randomness in the initial construction, the grasp restricted candidate list (feo & resende 1995)
BASELINE_TRIALS = 5  #how many random shuffles the baseline averages
BASELINE_SEED   = 42  #fixed so the baseline itself is reproducible


#QUALITY METRIC
#average within-room PAPER similarity, measured on the papers themselves
#identical across scripts 07 to 12 so every method is judged by the same ruler
#the objective the search climbs is cluster-level, this one reads the actual papers
def within_room_similarity(paper_room_idx, emb, R):
    per_room = np.full(R, np.nan)  #one score per room, nan until computed
    total_sim = 0.0
    total_pairs = 0.0
    for r in range(R):
        idx = np.where(paper_room_idx == r)[0]  #rows of the papers standing in room r
        nr = len(idx)
        if nr < 2:
            continue  #a room with 0 or 1 papers has no pairs to score
        E = emb[idx]
        #sum of all pairwise dot products without a double loop, plain algebra:
        #S @ S counts every pair twice plus each vector with itself,
        #so i subtract the self terms and halve whats left
        S = E.sum(axis=0)
        sim_sum = (float(S @ S) - float((E * E).sum())) / 2.0
        pairs = nr * (nr - 1) / 2.0  #number of unordered pairs in the room
        per_room[r] = sim_sum / pairs
        total_sim += sim_sum
        total_pairs += pairs
    #overall averages over every pair in every room together, not room by room
    overall = total_sim / total_pairs if total_pairs > 0 else float("nan")
    return per_room, overall


#RANDOM BASELINE
#shuffles the exact same papers into the exact same room sizes with no method
#at all, measures that the same way, and averages a few trials
#a similarity number only means something next to this baseline,
#the printed lift at the end is method against chance
def random_baseline_similarity(room_sizes, emb, R, n_trials=5, seed=42):
    rng = np.random.default_rng(seed)  #seeded so the baseline is reproducible
    N = len(emb)
    vals = []
    for _ in range(n_trials):
        perm = rng.permutation(N)  #one random shuffle of all papers
        assign = np.empty(N, dtype=int)
        #deal the shuffled papers into the rooms in order, filling each size exactly
        start = 0
        for r in range(R):
            assign[perm[start:start + room_sizes[r]]] = r
            start += room_sizes[r]
        _, overall = within_room_similarity(assign, emb, R)
        vals.append(overall)
    return float(np.mean(vals))


timings = {}  #stage timings collect in here for the summary at the end

#stop early with a clear message if an input file isnt where i expect it,
#way better than a confusing crash halfway through the search
for f in (EMBEDDINGS_FILE, CLUSTERS_FILE):
    if not f.exists():
        print(f"could not find: {f}")
        print("check the paths in the CONFIG section at the top of this script.")
        raise SystemExit(1)

OUTPUT_DIR.mkdir(exist_ok=True)  #make the output sub-folder, exist_ok stops the crash if its already there

#LOADING
#read the clusters csv and the embeddings once, both made by earlier scripts
#astype float32 halves the memory and the precision is plenty for similarity
t0 = time.perf_counter()
clusters = pd.read_csv(CLUSTERS_FILE)
embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
timings["load_data"] = time.perf_counter() - t0
#the assert catches the classic mistake of mixing files from different runs,
#row i of the embeddings must be the same paper as row i of the clusters csv
assert len(clusters) == len(embeddings), "embeddings.npy and 05_paper_clusters.csv are misaligned"

#basic counts of the problem, C clusters going into R rooms
cluster_ids = sorted(clusters["cluster"].unique())  #distinct cluster numbers, sorted for a stable order
C = len(cluster_ids)
rooms = list(ROOM_CAPACITIES.keys())
caps = np.array(list(ROOM_CAPACITIES.values()))
R = len(rooms)
n_papers = len(clusters)
print(f"{n_papers} papers in {C} clusters -> {R} rooms")
print(f"experiment: {len(SEEDS)} seeds x {TIME_BUDGET}s each "
      f"(~{len(SEEDS) * TIME_BUDGET:.0f}s of search total)")

#feasibility checks before any searching starts
#the first catches a minimum bigger than the smallest room,
#the second catches asking for more filled rooms than there are papers
if MIN_PAPERS_PER_ROOM > int(caps.min()):
    print(f"\nINFEASIBLE: MIN_PAPERS_PER_ROOM ({MIN_PAPERS_PER_ROOM}) exceeds the smallest "
          f"room ({int(caps.min())}). Lower it.")
    raise SystemExit(1)
if MIN_PAPERS_PER_ROOM * R > n_papers:
    print(f"\nINFEASIBLE: filling all {R} rooms to {MIN_PAPERS_PER_ROOM} needs "
          f"{MIN_PAPERS_PER_ROOM * R} papers but there are only {n_papers}. Lower it.")
    raise SystemExit(1)

#CLUSTER STATS
#one size and one centroid per cluster, computed once
#the centroid is the average of the clusters paper vectors, normalised back
#to unit length, the +1e-12 stops a divide by zero on a degenerate centroid
#these centroids are what the search actually works with
t0 = time.perf_counter()
size = np.zeros(C, dtype=int)
centroids = np.zeros((C, embeddings.shape[1]), dtype=np.float32)
for i, c in enumerate(cluster_ids):
    mask = (clusters["cluster"] == c).to_numpy()
    size[i] = int(mask.sum())
    cen = embeddings[mask].mean(axis=0)
    centroids[i] = cen / (np.linalg.norm(cen) + 1e-12)
timings["cluster_stats"] = time.perf_counter() - t0

#SIMILARITY TABLE
#W holds the similarity between every pair of centroids
#negatives get clipped to zero and the diagonal set to zero, since only real
#closeness between two different clusters should earn points
t0 = time.perf_counter()
W = np.maximum(centroids @ centroids.T, 0.0)
np.fill_diagonal(W, 0.0)
timings["similarity"] = time.perf_counter() - t0

#a cluster bigger than the biggest room fits nowhere, so no legal layout
#can exist, better to stop now and say exactly how to fix it
biggest = int(caps.max())
oversized = [(cluster_ids[i], int(size[i])) for i in range(C) if size[i] > biggest]
if oversized:
    print(f"\nPROBLEM: some clusters are bigger than the largest room ({biggest} papers):")
    for cid, s in oversized:
        print(f"   cluster {cid}: {s} papers")
    print("\nFix: raise NUM_CLUSTERS in 05, re-run 05, then this.")
    raise SystemExit(1)

cluster_id_to_pos = {cid: i for i, cid in enumerate(cluster_ids)}  #cluster number -> position in the arrays
clusters_pos = clusters["cluster"].map(cluster_id_to_pos).to_numpy()  #per paper, the position of its cluster


#SHARED HELPERS
#the score of a layout, the sum of similarities of every cluster pair sharing a room
#the sliced submatrix counts each pair twice (its symmetric, diagonal zero)
#so the sum gets halved
def cluster_objective(assign):
    total = 0.0
    for r in range(R):
        members = np.where(assign == r)[0]
        if len(members) > 1:
            total += W[np.ix_(members, members)].sum() / 2.0
    return total

#how many papers each room holds under a layout, cluster sizes added up
def room_loads(assign):
    loads = np.zeros(R, dtype=int)
    for i in range(C):
        loads[assign[i]] += int(size[i])
    return loads

#THE BUILD
#constructs one complete layout from nothing, the same randomized greedy
#construction grasp uses (feo & resende 1995), identical to 08, 09 and 10 on
#purpose so every method starts from the same kind of layout
#clusters are placed biggest first, the classic decreasing-order idea from
#bin packing (johnson et al. 1974), since the big ones are the hardest to fit
#stage one seeds every room up to its minimum so no room can end up empty
#stage two places everything left, each cluster gets every room that still has
#space scored by how similar the cluster is to whats already sitting there,
#and then one of the top RCL_SIZE rooms is picked at random
#a dead end (some cluster fits nowhere) returns None and the caller starts fresh
def construct():
    assign = np.full(C, -1, dtype=int)  #-1 means not placed yet
    loads = np.zeros(R, dtype=int)
    order = sorted(range(C), key=lambda k: -size[k])  #biggest cluster first
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
            continue  #already placed during the seeding stage
        feasible = [r for r in range(R) if loads[r] + size[i] <= caps[r]]
        if not feasible:
            return None
        #score every room that fits by the similarity gained from joining it
        scored = []
        for r in feasible:
            members = np.where(assign == r)[0]
            gain = W[i, members].sum() if len(members) > 0 else 0.0
            scored.append((gain, r))
        scored.sort(reverse=True)  #best gain first
        _, chosen = random.choice(scored[:RCL_SIZE])
        assign[i] = chosen
        loads[chosen] += int(size[i])
    return assign

#THE DESCENT
#the local search half of vns, two kinds of small change tried over and over:
#moving one cluster to another room, and swapping two clusters between rooms
#a change is kept only if it raises the score and leaves every room legal,
#meaning inside its capacity and not below the minimum
#the 1e-9 guards just ignore float noise pretending to be an improvement
#the loop runs until one full pass finds nothing left to improve
def local_search(assign):
    """Relocate + swap improving moves until no improvement (the descent for VNS).
    Identical engine to 08/09 so the methods differ only in how they explore."""
    loads = room_loads(assign)
    improved = True
    while improved:
        improved = False
        #relocate moves, one cluster to a different room
        for i in range(C):
            r_from = assign[i]
            if loads[r_from] - size[i] < MIN_PAPERS_PER_ROOM:
                continue  #taking it out would leave its room under the minimum
            loss = W[i, np.where(assign == r_from)[0]].sum()  #what leaving costs
            for r_to in range(R):
                if r_to == r_from or loads[r_to] + size[i] > caps[r_to]:
                    continue
                #keep the move if what the new room offers beats what leaving costs
                if W[i, np.where(assign == r_to)[0]].sum() - loss > 1e-9:
                    assign[i] = r_to
                    loads[r_from] -= size[i]
                    loads[r_to]   += size[i]
                    improved = True
                    break
        #swap moves, two clusters trade rooms
        for i in range(C):
            for j in range(i + 1, C):
                ri, rj = assign[i], assign[j]
                if ri == rj:
                    continue  #same room, nothing to trade
                #the room loads both would have after trading
                new_ri = loads[ri] - size[i] + size[j]
                new_rj = loads[rj] - size[j] + size[i]
                if new_ri > caps[ri] or new_rj > caps[rj]:
                    continue
                if new_ri < MIN_PAPERS_PER_ROOM or new_rj < MIN_PAPERS_PER_ROOM:
                    continue
                #each cluster is scored against the other room WITHOUT its swap
                #partner, since the partner is leaving that room in the trade
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

#THE SHAKE
#the other half of vns, the deliberate jump (mladenovic & hansen 1997)
#it makes k random moves without ever checking whether they help, which is the
#whole point, the descent afterwards is what turns the mess back into a layout
#every move still has to keep the layout legal, so the jump lands somewhere
#usable rather than somewhere broken
#the attempts counter is a safety net, random picks often land on illegal moves
#and without a cap this could spin forever when the rooms are tight
def shake(assign, k):
    """Jump to a random feasible solution k moves away (k random relocates/swaps)."""
    a = assign.copy()  #work on a copy, the caller keeps its own layout
    ld = room_loads(a)
    moves = 0  #how many jumps actually landed
    attempts = 0  #how many were tried, legal or not
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

#THE VNS LOOP
#this is the method itself, shake then descend, over and over
#x is where the search currently stands, best is the finest layout ever seen
#k grows every time a cycle fails and snaps back to 1 the moment one succeeds,
#which is the adaptive ladder of mladenovic & hansen 1997
#when k passes K_MAX it also resets to 1 rather than growing forever
def vns(initial, deadline):
    x = local_search(initial.copy())  #descend once before any shaking starts
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

#turns a cluster layout into a paper layout and measures it from outside,
#the paper-level check that rides along with every seed
def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall


#THE EXPERIMENT
#one vns run per seed, one long walk rather than many restarts,
#since the shake is what carries this method to new regions instead
#both random generators get the seed so one number fixes every choice a run makes
print(f"\nrunning VNS across {len(SEEDS)} seeds ({TIME_BUDGET}s budget each)...")
t0 = time.perf_counter()
seed_rows = []
global_best_obj = -1.0
global_best_assign = None
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    t_seed = time.perf_counter()
    deadline = t_seed + TIME_BUDGET  #the wall clock stop, the only stopping rule
    #the build is random and can dead end, so it gets up to 50 attempts
    init = None
    for _ in range(50):
        init = construct()
        if init is not None:
            break
    if init is None:
        print(f"  seed {seed:>2}: could not build a feasible start")
        continue  #this seed contributes no row, the others carry on
    best_assign, best_obj, iters = vns(init, deadline)
    sim = paper_similarity_of(best_assign)
    seed_rows.append({"seed": seed, "objective": round(best_obj, 4),
                      "similarity": round(sim, 4), "iterations": iters,
                      "seconds": round(time.perf_counter() - t_seed, 2)})
    print(f"  seed {seed:>2}: objective {best_obj:8.2f}  similarity {sim:.4f}  "
          f"({iters} shake/descend cycles)")
    #keep the best layout seen across all the seeds
    if best_obj > global_best_obj:
        global_best_obj = best_obj
        global_best_assign = best_assign.copy()
timings["experiment"] = time.perf_counter() - t0

#if every single seed failed to build a start there is nothing to save
if global_best_assign is None:
    print("\nno seed produced a feasible solution.")
    raise SystemExit(1)

#ACROSS SEEDS
#the distribution over runs that the statistical tests in 13 will read
objs = np.array([r["objective"] for r in seed_rows])
sims = np.array([r["similarity"] for r in seed_rows])
print("\n=== across seeds ===")
print(f"objective   best {objs.max():8.2f}   mean {objs.mean():8.2f}   std {objs.std():6.3f}")
print(f"similarity  best {sims.max():8.4f}   mean {sims.mean():8.4f}   std {sims.std():6.4f}")

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "11_seed_results.csv", index=False)

#SAVING THE BEST LAYOUT
#three detail files plus the metrics row script 13 reads
#the baseline shuffles the same papers into the exact same room sizes,
#so the lift printed below is method against chance
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}  #cluster number -> room name
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "11_room_assignment.csv", index=False)

paper_room_idx = global_best_assign[clusters_pos]
per_room_sim, overall_sim = within_room_similarity(paper_room_idx, embeddings, R)
room_sizes = [int((paper_room_idx == r).sum()) for r in range(R)]
baseline_sim = random_baseline_similarity(room_sizes, embeddings, R, BASELINE_TRIALS, BASELINE_SEED)

#one row per room, how full it came out, how many clusters it holds,
#how similar its papers are, and which clusters landed there
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

#TIMING SUMMARY
#where the seconds went, stage by stage
total = sum(timings.values())
print("\n=== timing summary (seconds) ===")
for stage, secs in timings.items():
    print(f"  {stage:<16} {secs:8.2f}")
print(f"  {'TOTAL':<16} {total:8.2f}")

#one metrics row with the same columns every method writes,
#so the comparison table in 13 assembles itself
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