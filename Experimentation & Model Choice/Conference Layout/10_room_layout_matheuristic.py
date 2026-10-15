#matheuristic (fix-and-optimize / exact large-neighborhood search) for packing clusters
#into the rooms, script 10 of the layout benchmark suite
#a hybrid that wraps the exact cp-sat solver inside a heuristic loop

#same problem, objective, similarity metric and experiment scaffold as 07, 08 and 09
#for a fair head-to-head the TIME_BUDGET and the SEEDS must match the other scripts

#the method:
#matheuristics in general, meaning heuristics with a maths solver inside them,
#are collected in maniezzo, stutzle & voss 2010
#the specific idea here is fix-and-optimize, first used for lot sizing by
#helber & sahling 2010, and it belongs to the wider family of large neighborhood
#search (shaw 1998, pisinger & ropke 2010, and the survey by ahuja et al. 2002)
#applying this to capacitated clustering, the family this problem belongs to,
#is gnagi & baumann 2021

#the idea in plain words (gnagi & baumann 2021): start from a legal layout,
#then repeatedly LOCK most clusters where they are and let cp-sat re-place only
#a small FREE subset perfectly, respecting the seats the locked clusters take up,
#keeping the answer only when it improves
#the current placement is always available inside each sub-solve (it goes in as a
#hint) and only strict improvements are accepted, so the layout never gets worse
#restart on stall, keep the best
#the bet is that many small perfect decisions add up to one strong layout

#the construction that builds each start is the randomized greedy of grasp
#(feo & resende 1995), placing big clusters first, the classic decreasing-order
#idea from bin packing (johnson et al. 1974)

#install once:  pip install ortools

#inputs: embeddings.npy from the Similarity Metrics folder (made by 04)
#05_paper_clusters.csv from the Hierarchical Clustering folder (made by 05)
#outputs: 10_room_assignment.csv, 10_room_summary.csv,
#10_seed_results.csv, 10_optimization_metrics.csv

#libraries used in this script and where each one comes from:
#pathlib, time and random are python standard library, docs.python.org
#numpy does the vector math, harris et al. 2020, github.com/numpy/numpy
#pandas reads and writes the csvs, mckinney 2010, github.com/pandas-dev/pandas
#ortools gives the cp-sat solver, perron & didier 2025, github.com/google/or-tools
#cp-sat docs: developers.google.com/optimization/cp/cp_solver

from pathlib import Path
import time
import random

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
ID_COL          = "Number"  #the column that identifies a paper in the clusters csv

#all csv outputs go into this sub-folder, created next to this script
OUTPUT_DIR = BASE_DIR / "10 Matheuristic Room Layout"

#the twelve real rooms of the conference
#the capacities add up to exactly 5286, one seat per paper, zero slack
ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1  #1 means no empty rooms, higher forces fuller rooms (keep it below the smallest room, 300)

#fairness controls, these MUST match scripts 07, 08, 09 and the rest of the suite
#every method gets the same budget and the same seeds so the statistical
#tests in 13 can pair attempt against attempt
#60 seconds is the budget the real results were produced with
TIME_BUDGET = 60.0  #seconds of search PER SEED
SEEDS       = list(range(20))  #the seeds to run (paired with the others)

#FREE_SIZE is the size of the neighborhood handed back to the exact solver
#it rests on one fact: cp-sat chokes on all 529 clusters at once, but solves
#twelve of them perfectly in a fraction of a second, so it gets twelve at a time
#bigger means better sub-answers but far slower solves, this is the main dial
#of a fix-and-optimize method (helber & sahling 2010)
FREE_SIZE       = 12  #clusters released for the exact re-optimisation each step
STALL_ITERS     = 30  #restart a trajectory after this many steps with no gain
SUBSOLVE_TIME   = 2.0  #safety cap in seconds for each tiny exact sub-solve
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
caps = list(ROOM_CAPACITIES.values())
R = len(rooms)
n_papers = len(clusters)
print(f"{n_papers} papers in {C} clusters -> {R} rooms")
print(f"experiment: {len(SEEDS)} seeds x {TIME_BUDGET}s each "
      f"(~{len(SEEDS) * TIME_BUDGET:.0f}s of search total)")
print(f"freeing {FREE_SIZE} clusters per exact re-optimisation step")

#feasibility checks before any searching starts
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
#W_int is the same table scaled by 1000 and rounded, cp-sat can only maximise
#an integer objective (or-tools docs), the float W stays for scoring and acceptance
t0 = time.perf_counter()
sim = centroids @ centroids.T
W = np.maximum(sim, 0.0)  #float weights (objective + acceptance)
np.fill_diagonal(W, 0.0)
W_int = np.rint(W * 1000).astype(int)  #integer weights for the cp-sat sub-model
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
#construction grasp uses (feo & resende 1995), identical to 08 and 09 on purpose
#so every method starts from the same kind of layout and only the search differs
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

#turns a cluster layout into a paper layout and measures it from outside,
#the paper-level check that rides along with every seed
def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall

#THE FIX-AND-OPTIMIZE STEP
#this is the whole method in one function (helber & sahling 2010, gnagi & baumann 2021)
#F is the small set of clusters that get freed, everything else stays put
#a brand new tiny cp-sat model is built each call, holding only the free clusters,
#so the solver sees twelve decisions instead of 529 and can finish in a blink
def fix_and_optimize(assign, F, deadline):
    """Lock all clusters except F; let CP-SAT re-place F optimally given the locked load."""
    Fset = set(F)  #a set so the membership test below is instant
    locked_load = np.zeros(R, dtype=int)  #seats already taken by the locked clusters
    fixed_in_room = [[] for _ in range(R)]  #which locked clusters sit in each room
    for d in range(C):
        if d in Fset:
            continue
        rr = assign[d]
        locked_load[rr] += int(size[d])
        fixed_in_room[rr].append(d)

    #the sub-model, one yes/no variable per free cluster per room,
    #plus room_f holding the same choice as a plain integer for the pair rules
    m = cp_model.CpModel()
    y = {c: [m.NewBoolVar(f"y_{c}_{r}") for r in range(R)] for c in F}
    room_f = {c: m.NewIntVar(0, R - 1, f"rf_{c}") for c in F}
    for c in F:
        m.Add(sum(y[c][r] for r in range(R)) == 1)  #each free cluster lands in exactly one room
        m.Add(room_f[c] == sum(r * y[c][r] for r in range(R)))
        for r in range(R):
            m.AddHint(y[c][r], 1 if assign[c] == r else 0)   #warm-start from current
    #the capacity rule counts the locked seats plus whatever the free ones add,
    #so the sub-answer is always legal in the full layout
    for r in range(R):
        load_r = locked_load[r] + sum(int(size[c]) * y[c][r] for c in F)
        m.Add(load_r <= caps[r])
        m.Add(load_r >= MIN_PAPERS_PER_ROOM)

    #the objective has two halves
    #half one, a free cluster joining a room earns the similarity to everything
    #locked in that room, which is a plain number known up front, so its linear
    obj = []
    for c in F:                              # free <-> fixed attraction (linear)
        for r in range(R):
            if fixed_in_room[r]:
                coef = int(W_int[c, fixed_in_room[r]].sum())
                if coef:
                    obj.append(coef * y[c][r])
    #half two, two free clusters landing together, which isnt known up front,
    #so it needs a yes/no variable s tied to them sharing a room
    #OnlyEnforceIf is cp-sats way of applying a rule only when s is true
    Fl = list(F)                             # free <-> free (reified same-room)
    for a in range(len(Fl)):
        for b in range(a + 1, len(Fl)):
            c, d = Fl[a], Fl[b]
            if W_int[c, d] == 0:
                continue  #zero weight pairs cant change the score, skip them
            s = m.NewBoolVar(f"s_{c}_{d}")
            m.Add(room_f[c] == room_f[d]).OnlyEnforceIf(s)
            m.Add(room_f[c] != room_f[d]).OnlyEnforceIf(s.Not())
            obj.append(int(W_int[c, d]) * s)
    m.Maximize(sum(obj))

    #one thread keeps the tiny solve cheap and repeatable
    #the time cap is whichever is smaller, SUBSOLVE_TIME or whats left of the
    #whole budget, and never below 0.05s so the call cant be handed zero time
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(0.05, min(SUBSOLVE_TIME, deadline - time.perf_counter()))
    st = solver.Solve(m)
    new = assign.copy()
    #if the solve produced anything, read the free clusters new rooms out of it,
    #if it produced nothing the copy is returned unchanged and nothing is lost
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for c in F:
            for r in range(R):
                if solver.Value(y[c][r]) == 1:
                    new[c] = r
                    break
    return new


#THE EXPERIMENT
#one multi-start fix-and-optimize run per seed
#build a start, then free a random dozen clusters over and over and let the
#solver re-place them, accepting only improvements, restart when it stalls
#both random generators get the seed so one number fixes every choice a run makes
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
    deadline = t_seed + TIME_BUDGET  #the wall clock stop, the only stopping rule
    while time.perf_counter() < deadline:
        #the build is random and can dead end, so it gets up to 50 attempts
        #before giving up on this seed entirely
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
        #the improvement loop, each pass frees a random FREE_SIZE clusters
        #and hands them to the exact solver
        while time.perf_counter() < deadline and stall < STALL_ITERS:
            F = random.sample(range(C), min(FREE_SIZE, C))
            new = fix_and_optimize(cur, F, deadline)
            iters += 1
            new_obj = cluster_objective(new)
            #strict improvement only, which is why the layout can never get worse
            #the 1e-9 guard ignores float noise pretending to be an improvement
            if new_obj > cur_obj + 1e-9:
                cur = new
                cur_obj = new_obj
                stall = 0
                if cur_obj > seed_best_obj:
                    seed_best_obj = cur_obj
                    seed_best_assign = cur.copy()
            else:
                stall += 1  #that random dozen had nothing to give, try another
    sim = paper_similarity_of(seed_best_assign)
    seed_rows.append({"seed": seed, "objective": round(seed_best_obj, 4),
                      "similarity": round(sim, 4), "restarts": restarts,
                      "subsolves": iters, "seconds": round(time.perf_counter() - t_seed, 2)})
    print(f"  seed {seed:>2}: objective {seed_best_obj:8.2f}  similarity {sim:.4f}  "
          f"({restarts} restarts, {iters} sub-solves)")
    #keep the best layout seen across all the seeds
    if seed_best_obj > global_best_obj:
        global_best_obj = seed_best_obj
        global_best_assign = seed_best_assign.copy()
timings["experiment"] = time.perf_counter() - t0

#ACROSS SEEDS
#the distribution over runs that the statistical tests in 13 will read
objs = np.array([r["objective"] for r in seed_rows])
sims = np.array([r["similarity"] for r in seed_rows])
print("\n=== across seeds ===")
print(f"objective   best {objs.max():8.2f}   mean {objs.mean():8.2f}   std {objs.std():6.3f}")
print(f"similarity  best {sims.max():8.4f}   mean {sims.mean():8.4f}   std {sims.std():6.4f}")

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "10_seed_results.csv", index=False)

#SAVING THE BEST LAYOUT
#three detail files plus the metrics row script 13 reads
#the baseline shuffles the same papers into the exact same room sizes,
#so the lift printed below is method against chance
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}  #cluster number -> room name
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "10_room_assignment.csv", index=False)

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
summary.to_csv(OUTPUT_DIR / "10_room_summary.csv", index=False)
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