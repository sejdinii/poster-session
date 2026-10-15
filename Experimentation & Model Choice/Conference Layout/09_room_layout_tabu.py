#tabu search heuristic for packing clusters into the rooms, script 09 of the layout benchmark suite

#run as a FAIR experiment:
#matched wall-clock budget (TIME_BUDGET seconds) instead of a fixed step count,
#repeated across a set of SEEDS, reporting best / mean / std for objective and similarity
#for a fair head-to-head with grasp (08) the TIME_BUDGET and the SEEDS must be the same,
#same objective and same similarity metric, so the methods are directly comparable

#the method, tabu search (glover 1986, glover 1989, glover & laguna 1997):
#a plain local search stops at the first dead end where nothing improves
#tabu search keeps walking anyway, taking the best move available even when
#it makes the score worse, and to stop itself from instantly undoing that move
#and circling forever it forbids each just-moved cluster for a number of steps
#that forbidden list is where the name comes from, its the short term memory
#the escape valve is the aspiration criterion, a tabu move is still allowed
#if it would produce the best layout seen so far (glover & laguna 1997)

#tabu search has been applied to clique partitioning, the family this problem
#belongs to, by palubeckis, ostreika & tomkevicius 2014, and to capacitated
#clustering by martinez-gavara et al. 2015

#within each seeds budget this runs multi-start tabu search: build a start,
#run a tabu trajectory until it converges or time runs out, then restart,
#keeping the best layout seen anywhere
#the construction that builds each start is the randomized greedy of grasp
#(feo & resende 1995), and it places big clusters first, the classic
#decreasing-order idea from bin packing (johnson et al. 1974)

#one thing worth knowing about heuristics like this: they have no natural
#finishing point, the search would just go forever, so the stop is the clock
#thats why TIME_BUDGET exists

#inputs: embeddings.npy from the Similarity Metrics folder (made by 04)
#05_paper_clusters.csv from the Hierarchical Clustering folder (made by 05)
#outputs: 09_room_assignment.csv, 09_room_summary.csv,
#09_seed_results.csv, 09_optimization_metrics.csv

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
OUTPUT_DIR = BASE_DIR / "09 Tabu Search Room Layout"

#the twelve real rooms of the conference
#the capacities add up to exactly 5286, one seat per paper, zero slack
ROOM_CAPACITIES = {
    "Room 1": 700, "Room 2": 600, "Room 3": 550,
    "Room 4": 500, "Room 5": 450, "Room 6": 425,
    "Room 7": 400, "Room 8": 380, "Room 9": 350,
    "Room 10": 320, "Room 11": 300, "Room 12": 311,
}

MIN_PAPERS_PER_ROOM = 1  #1 means no empty rooms, higher forces fuller rooms (keep it below the smallest room, 300)

#fairness controls, these MUST match script 08 and the rest of the suite
#every method gets the same budget and the same seeds so the statistical
#tests in 13 can pair attempt against attempt
#60 seconds is the budget the real results were produced with
TIME_BUDGET = 60.0  #seconds of search PER SEED
SEEDS       = list(range(20))  #the seeds to run (paired with 08)

#tenure is how long a just-moved cluster stays forbidden, the core tabu
#parameter (glover & laguna 1997), too short and the search circles,
#too long and it cant reach good layouts at all
TABU_TENURE     = 12  #how many steps a just-moved cluster stays tabu
STALL_LIMIT     = 250  #restart a trajectory after this many steps with no gain
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
        loads[assign[i]] += size[i]
    return loads

#THE BUILD
#constructs one complete layout from nothing, the same randomized greedy
#construction grasp uses (feo & resende 1995), identical to script 08 on purpose
#so both methods start from the same kind of layout and only the search differs
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
            loads[r] += size[i]
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
        loads[chosen] += size[i]
    return assign

#THE TABU WALK
#one trajectory, the actual tabu search (glover & laguna 1997)
#every step looks at ALL possible relocate and swap moves, picks the single
#best one that is allowed, and takes it even if it lowers the score
#tabu_until[i] holds the step number until which cluster i is forbidden,
#and the aspiration rule lets a forbidden move through anyway when it would
#beat the best layout seen so far
#the walk ends on the deadline, on a dead end with no allowed move, or after
#STALL_LIMIT steps without a new best
#the 1e-9 guards just ignore float noise pretending to be an improvement
def tabu_search(assign, deadline):
    loads = room_loads(assign)
    cur_obj = cluster_objective(assign)
    best_assign = assign.copy()  #the best layout seen anywhere on this walk
    best_obj = cur_obj
    tabu_until = np.zeros(C, dtype=int)  #per cluster, the step it becomes free again
    stall = 0
    step = 0
    while time.perf_counter() < deadline:
        step += 1
        members = [np.where(assign == r)[0] for r in range(R)]  #who sits in each room right now
        best_move = None
        best_delta = -np.inf  #starts at minus infinity so even a worsening move gets taken

        #relocate moves, one cluster to a different room
        for i in range(C):
            r_from = assign[i]
            if loads[r_from] - size[i] < MIN_PAPERS_PER_ROOM:
                continue  #taking it out would leave its room under the minimum
            loss = W[i, members[r_from]].sum()  #what leaving costs
            for r_to in range(R):
                if r_to == r_from or loads[r_to] + size[i] > caps[r_to]:
                    continue
                delta = W[i, members[r_to]].sum() - loss  #change in score, can be negative
                is_tabu = step < tabu_until[i]
                #the aspiration criterion, a tabu move is allowed after all
                #if it would give the best layout seen so far
                if is_tabu and not (cur_obj + delta > best_obj + 1e-9):
                    continue
                if delta > best_delta:
                    best_delta = delta
                    best_move = ("relocate", i, r_from, r_to)

        #swap moves, two clusters trade rooms
        for i in range(C):
            ri = assign[i]
            for j in range(i + 1, C):
                rj = assign[j]
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
                mi, mj = members[ri], members[rj]
                cur = W[i, mi].sum() + W[j, mj].sum()
                new = W[i, mj[mj != j]].sum() + W[j, mi[mi != i]].sum()
                delta = new - cur
                #a swap is tabu if either of the two clusters is still forbidden
                is_tabu = (step < tabu_until[i]) or (step < tabu_until[j])
                if is_tabu and not (cur_obj + delta > best_obj + 1e-9):
                    continue
                if delta > best_delta:
                    best_delta = delta
                    best_move = ("swap", i, j, ri, rj)

        if best_move is None:
            break  #nothing allowed anywhere, the walk is stuck

        #carry out the winning move and mark the clusters involved as tabu
        if best_move[0] == "relocate":
            _, i, r_from, r_to = best_move
            assign[i] = r_to
            loads[r_from] -= size[i]
            loads[r_to]   += size[i]
            tabu_until[i] = step + TABU_TENURE
        else:
            _, i, j, ri, rj = best_move
            assign[i], assign[j] = rj, ri
            loads[ri] = loads[ri] - size[i] + size[j]
            loads[rj] = loads[rj] - size[j] + size[i]
            tabu_until[i] = step + TABU_TENURE
            tabu_until[j] = step + TABU_TENURE

        #the score is updated by the delta instead of recomputed from scratch,
        #which is what makes a step cheap enough to do thousands of times
        cur_obj += best_delta
        if cur_obj > best_obj + 1e-9:
            best_obj = cur_obj
            best_assign = assign.copy()
            stall = 0
        else:
            stall += 1
            if stall >= STALL_LIMIT:
                break  #wandering with no gain for too long, cut it and restart
    #the best layout seen is returned, not wherever the walk happened to end
    return best_assign, step

#turns a cluster layout into a paper layout and measures it from outside,
#the paper-level check that rides along with every seed
def paper_similarity_of(assign):
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall, paper_room_idx


#THE EXPERIMENT
#one multi-start tabu run per seed, build a start, walk it, restart,
#until the clock runs out, keeping the best layout seen
#both random generators get the seed so one number fixes every choice a run makes
print(f"\nrunning Tabu Search across {len(SEEDS)} seeds ({TIME_BUDGET}s budget each)...")
t0 = time.perf_counter()
seed_rows = []
global_best_obj = -1.0
global_best_assign = None
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    seed_best_obj = -1.0
    seed_best_assign = None
    trajectories = 0
    steps_total = 0
    t_seed = time.perf_counter()
    deadline = t_seed + TIME_BUDGET  #the wall clock stop, the only stopping rule
    while time.perf_counter() < deadline:
        #the build is random and can dead end, so it gets up to 50 attempts
        #before giving up on this seed entirely
        init = None
        for _ in range(50):
            init = construct()
            if init is not None:
                break
        if init is None:
            break
        assign, steps = tabu_search(init, deadline)
        obj = cluster_objective(assign)
        trajectories += 1
        steps_total += steps
        if obj > seed_best_obj:
            seed_best_obj = obj
            seed_best_assign = assign.copy()
    sim, _ = paper_similarity_of(seed_best_assign)
    seed_rows.append({"seed": seed, "objective": round(seed_best_obj, 4),
                      "similarity": round(sim, 4), "trajectories": trajectories,
                      "steps": steps_total, "seconds": round(time.perf_counter() - t_seed, 2)})
    print(f"  seed {seed:>2}: objective {seed_best_obj:8.2f}  similarity {sim:.4f}  "
          f"({trajectories} trajectories, {steps_total} steps)")
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

pd.DataFrame(seed_rows).to_csv(OUTPUT_DIR / "09_seed_results.csv", index=False)

#SAVING THE BEST LAYOUT
#three detail files plus the metrics row script 13 reads
#the baseline shuffles the same papers into the exact same room sizes,
#so the lift printed below is method against chance
t0 = time.perf_counter()
cluster_to_room = {cluster_ids[i]: rooms[global_best_assign[i]] for i in range(C)}  #cluster number -> room name
clusters["room"] = clusters["cluster"].map(cluster_to_room)
clusters[[ID_COL, "cluster", "room"]].to_csv(OUTPUT_DIR / "09_room_assignment.csv", index=False)

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
summary.to_csv(OUTPUT_DIR / "09_room_summary.csv", index=False)
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
    "method": "Tabu Search",
    "time_budget_s": TIME_BUDGET, "n_seeds": len(SEEDS),
    "obj_best": round(float(objs.max()), 4), "obj_mean": round(float(objs.mean()), 4),
    "obj_std": round(float(objs.std()), 4),
    "sim_best": round(float(sims.max()), 4), "sim_mean": round(float(sims.mean()), 4),
    "sim_std": round(float(sims.std()), 4),
    "best_run_similarity": round(overall_sim, 4),
    "random_baseline_similarity": round(baseline_sim, 4),
    "avg_trajectories_per_seed": round(float(np.mean([r["trajectories"] for r in seed_rows])), 1),
    "min_papers_per_room": MIN_PAPERS_PER_ROOM, "tabu_tenure": TABU_TENURE,
    "n_papers": n_papers, "n_clusters": C, "n_rooms": R,
}]).to_csv(OUTPUT_DIR / "09_optimization_metrics.csv", index=False)
print("\nsaved: 09_room_assignment.csv, 09_room_summary.csv, "
      "09_seed_results.csv, 09_optimization_metrics.csv")