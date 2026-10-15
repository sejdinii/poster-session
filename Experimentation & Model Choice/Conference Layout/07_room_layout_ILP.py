#exact assignment with cp-sat, script 07 of the layout benchmark suite
#this is the exact solver the heuristics 08 to 12 get compared against,
#run under the same experimental conditions so every method shares one scale:
#same wall-clock budget (TIME_BUDGET seconds) per run, used as the solver time limit,
#and the same SEEDS list (cp-sats internal random seed changes per run)
#the fairness protocol is the one set in section 7.4 of the dissertation

#NOTE on what exact means here: cp-sat is exact and near-deterministic
#if the budget is enough to PROVE optimality, every seed returns the same
#optimum (std around 0) and that value is the gap reference for 08 to 12
#if the budget is too short it returns the best feasible layout found
#(status FEASIBLE) and the optimality guarantee is LOST
#the per-run optimal flag in the csv tells me which of the two happened
#thats the best-found vs proven-best distinction section 7.1 draws
#the full-scale four hour attempt with this exact model is reported in section 7.3

#the objective is recomputed with the same float function the heuristics use,
#so the numbers sit on one scale (cp-sat internally maximises an integer-scaled version)

#the problem and the model:
#this is the exact integer model stated in section 7.2 of the dissertation,
#table 7.1 walks every line of it in plain words
#the problem family is clique partitioning (grotschel & wakabayashi 1989),
#shown to be np-hard (grotschel & wakabayashi 1990)
#modern solvers applied to exactly this problem: du et al. 2022

#libraries used in this script and where each one comes from:
#pathlib and time are python standard library, docs.python.org
#numpy does the vector math, harris et al. 2020, github.com/numpy/numpy
#pandas reads and writes the csvs, mckinney 2010, github.com/pandas-dev/pandas
#ortools gives the cp-sat solver, perron & didier 2025, github.com/google/or-tools
#cp-sat docs: developers.google.com/optimization/cp/cp_solver

#install once:  pip install ortools

#inputs: embeddings.npy from the Similarity Metrics folder (made by 04)
#05_paper_clusters.csv from the Hierarchical Clustering folder (made by 05)
#outputs: 07_room_assignment.csv, 07_room_summary.csv,
#07_seed_results.csv, 07_optimization_metrics.csv

from pathlib import Path
import time

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

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

#all csv outputs go into this sub-folder, created next to this script
OUTPUT_DIR = BASE_DIR / "07 ILP Room Layout"
ID_COL          = "Number"  #the column that identifies a paper in the clusters csv

#the twelve real rooms used across chapters 7 to 9 of the dissertation
#the capacities add up to exactly 5286, one seat per paper, zero slack
ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1  #1 means no empty rooms, higher forces fuller rooms (keep it below the smallest room, 300)

#fairness controls, these MUST match scripts 08 to 12
#every method gets the same budget and the same seeds so the statistical
#tests in 13 can pair attempt against attempt, the section 7.4 protocol
#5.0 is my quick smoke-test setting, the dissertation numbers in tables
#7.2 to 7.8 come from 60 second runs, so set 60.0 here to reproduce them
TIME_BUDGET = 5.0  #seconds PER SEED (used as the cp-sat time limit)
SEEDS       = list(range(20))  #the seeds to run (paired with 08 to 12)

NUM_WORKERS     = 8  #cp-sat search threads (set to 1 for strict reproducibility)
BASELINE_TRIALS = 5  #how many random shuffles the baseline averages
BASELINE_SEED   = 42  #fixed so the baseline itself is reproducible


#QUALITY METRIC
#average within-room PAPER similarity, the outside measure from section 3.5
#identical across scripts 07 to 12 so every method is judged by the same ruler
#the objective the solver climbs is cluster-level, this one reads the actual papers
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
#this is the random baseline of section 3.5, a similarity number only means
#something next to it, the printed lift at the end is method against chance
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
#way better than a confusing crash halfway through the solve
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
caps = list(ROOM_CAPACITIES.values())
R = len(rooms)
n_papers = len(clusters)
print(f"{n_papers} papers in {C} clusters -> {R} rooms")
print(f"experiment: {len(SEEDS)} seeds x {TIME_BUDGET}s each (CP-SAT time limit per run)")

#feasibility checks before any solving starts
#the first catches a minimum bigger than the smallest room,
#the second catches asking for more filled rooms than there are papers
if MIN_PAPERS_PER_ROOM > min(caps):
    print(f"\nINFEASIBLE: MIN_PAPERS_PER_ROOM ({MIN_PAPERS_PER_ROOM}) exceeds the smallest "
          f"room ({min(caps)}). Lower it.")
    raise SystemExit(1)
if MIN_PAPERS_PER_ROOM * R > n_papers:
    print(f"\nINFEASIBLE: filling all {R} rooms to {MIN_PAPERS_PER_ROOM} needs "
          f"{MIN_PAPERS_PER_ROOM * R} papers but there are only {n_papers}. Lower it.")
    raise SystemExit(1)

#CLUSTER STATS
#one size and one centroid per cluster, computed once
#the centroid is the average of the clusters paper vectors, normalised back
#to unit length, the +1e-12 stops a divide by zero on a degenerate centroid
#these centroids are what the optimisation actually works with (section 3.4)
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

#SIMILARITY TABLE
#W holds the similarity between every pair of centroids, the W table of section 3.4
#negatives get clipped to zero and the diagonal set to zero, since only real
#closeness between two different clusters should earn points
#W_int is the same table scaled by 1000 and rounded, cp-sat can only maximise
#an integer objective (or-tools docs), the float W stays for the comparable score
t0 = time.perf_counter()
sim = centroids @ centroids.T
W = np.maximum(sim, 0.0)  #float weights for the comparable objective
np.fill_diagonal(W, 0.0)
W_int = np.rint(W * 1000).astype(int)  #integer weights for the cp-sat model
timings["similarity"] = time.perf_counter() - t0

#a cluster bigger than the biggest room fits nowhere, so no legal layout
#can exist, better to stop now and say exactly how to fix it
biggest = max(caps)
oversized = [(cluster_ids[i], int(size[i])) for i in range(C) if size[i] > biggest]
if oversized:
    print(f"\nPROBLEM: some clusters are bigger than the largest room ({biggest} papers):")
    for cid, s in oversized:
        print(f"   cluster {cid}: {s} papers")
    print("\nFix: raise NUM_CLUSTERS in 05, re-run 05, then this.")
    raise SystemExit(1)

cluster_id_to_pos = {cid: i for i, cid in enumerate(cluster_ids)}  #cluster number -> position in the arrays
clusters_pos = clusters["cluster"].map(cluster_id_to_pos).to_numpy()  #per paper, the position of its cluster


#the score of a layout, the sum of similarities of every cluster pair sharing
#a room, this is f(a) from section 7.2
#the sliced submatrix counts each pair twice (its symmetric, diagonal zero)
#so the sum gets halved
def cluster_objective(assign):  #same float objective as 08 to 12, for comparability
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

#turns a cluster layout into a paper layout and measures it from outside,
#the paper-level check that rides along with every solve (section 3.5)
def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall

#BUILDING THE CP-SAT MODEL
#built ONCE and reused for every seed, only the solvers random seed changes
#this is the integer program written out in section 7.2, table 7.1 explains
#it line by line in plain words
t0 = time.perf_counter()
model = cp_model.CpModel()
#x[c][r] is a yes/no variable saying cluster c sits in room r
#room[c] holds the same information as one plain integer, handy for the pair rules
x = [[model.NewBoolVar(f"x_{c}_{r}") for r in range(R)] for c in range(C)]
room = [model.NewIntVar(0, R - 1, f"room_{c}") for c in range(C)]
#every cluster sits in exactly one room, and room[c] is tied to whichever x lit up
for c in range(C):
    model.Add(sum(x[c][r] for r in range(R)) == 1)
    model.Add(room[c] == sum(r * x[c][r] for r in range(R)))
#no room over its seat count, no room under the minimum occupancy
for r in range(R):
    papers_in_room = sum(int(size[c]) * x[c][r] for c in range(C))
    model.Add(papers_in_room <= caps[r])
    model.Add(papers_in_room >= MIN_PAPERS_PER_ROOM)
#the objective, one yes/no variable s per cluster pair
#OnlyEnforceIf is cp-sats way of saying a rule only applies when s is true,
#the two lines together force s to be true exactly when both clusters share
#a room, and then that pairs weight counts towards the score
#pairs with weight zero are skipped since they cant change anything
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


#read the finished layout back out of the solver,
#for each cluster find which x variable came out as 1
def extract_assignment(solver):
    assign = np.empty(C, dtype=int)
    for c in range(C):
        for r in range(R):
            if solver.Value(x[c][r]) == 1:
                assign[c] = r
                break
    return assign


#THE EXPERIMENT
#solve once per seed, same model, same budget, only the seed changes
#cp-sat is near deterministic so the spread should be tiny when it has enough time
print(f"\nrunning CP-SAT across {len(SEEDS)} seeds ({TIME_BUDGET}s limit each)...")
t0 = time.perf_counter()
seed_rows = []
global_best_obj = -1.0
global_best_assign = None
for seed in SEEDS:
    solver = cp_model.CpSolver()  #a fresh solver per seed
    #max_time_in_seconds is the wall clock budget, the fairness control,
    #and random_seed is what makes the 20 runs pairable with 08 to 12
    solver.parameters.max_time_in_seconds = TIME_BUDGET
    solver.parameters.num_search_workers = NUM_WORKERS
    solver.parameters.random_seed = int(seed)
    ts = time.perf_counter()
    status = solver.Solve(model)
    solve_s = time.perf_counter() - ts
    status_name = solver.StatusName(status)
    #OPTIMAL means proven best, FEASIBLE means best found before time ran out,
    #anything else means no layout at all in the budget, logged as nan
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
    #keep the best layout seen across all the seeds
    if obj > global_best_obj:
        global_best_obj = obj
        global_best_assign = assign.copy()
timings["experiment"] = time.perf_counter() - t0

if global_best_assign is None:
    print("\nno seed produced a feasible solution within the budget; raise TIME_BUDGET.")
    raise SystemExit(1)

#ACROSS SEEDS
#the distribution over runs that the statistical tests in 13 will read
solved = [r for r in seed_rows if not np.isnan(r["objective"])]  #only runs that produced a layout
objs = np.array([r["objective"] for r in solved])
sims = np.array([r["similarity"] for r in solved])
n_opt = sum(1 for r in seed_rows if r["optimal"])  #how many runs ended with a proof
print("\n=== across seeds ===")
print(f"objective   best {objs.max():8.2f}   mean {objs.mean():8.2f}   std {objs.std():6.3f}")
print(f"similarity  best {sims.max():8.4f}   mean {sims.mean():8.4f}   std {sims.std():6.4f}")
print(f"proven optimal in {n_opt} of {len(SEEDS)} runs within {TIME_BUDGET}s")
#this warning matters, without a single proven run the best value above is
#best-found, not the true optimum, the distinction section 7.1 explains
if n_opt == 0:
    print("WARNING: optimality was never proven in the budget - this is NOT the true optimum.")
    print("         raise TIME_BUDGET (or run 07 once with no limit) to get the reference value.")

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "07_seed_results.csv", index=False)

#SAVING THE BEST LAYOUT
#three detail files plus the metrics row script 13 reads
#the baseline shuffles the same papers into the exact same room sizes,
#so the lift printed below is method against chance (section 3.5)
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}  #cluster number -> room name
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "07_room_assignment.csv", index=False)

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