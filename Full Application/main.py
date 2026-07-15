# main.py - the entry point of the pipeline. this is the only file you run.
#
# what it does, start to finish:
#   1. takes the users papers file and room sizes from the command line
#   2. checks the input makes sense (friendly errors, no crashes)
#   3. embedding turns the abstracts into vectors
#   4. clustering groups similar papers into small topic clusters
#   5. grasp packs whole clusters into the rooms
#   6. quality measures how good the layout really is
#   7. saves the layout + a room summary + a small quality report
#
# how to run it:
#   python main.py --papers my_papers.csv --rooms 700,600,550
#
# optional flags:
#   --abstract-col   name of the abstract column (default: Abstract)
#   --id-col         name of the paper id column (default: row numbers 1,2,3...)
#   --out            output folder (default: a new "Production N" folder next
#                    to main.py - Production 1, Production 2, ... one per run)
#
# this is the only file that knows about files, paths and users. the other four
# files (embedding, clustering, grasp, quality) never touch a file or a path -
# they get data handed in and give data back.

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# the worker files live in subfolders, and one folder name ("Layout evaluation")
# contains a space - normal import syntax cant handle that. so we add each
# subfolder to the list of places python searches for files, and then import
# every worker by its plain file name. folder names can be anything this way
BASE_DIR = Path(__file__).resolve().parent
for sub in ("Embeddings", "Clustering", "Heuristics", "Layout evaluation"):
    sys.path.insert(0, str(BASE_DIR / sub))

from embeddings_miniLM import embed
from hierarchical_clustering import build_tree, cut_tree, cluster_info, similarity_table
from GRASP import grasp, room_loads
from layout_evaluation import (within_room_similarity, random_baseline_similarity,
                               paper_similarity_of, room_summary)


# ---- the settings of the tool, all in one place ----

# how fine to cluster: this is the same rule as the thesis (clusters = papers / 10),
# just read from the other end - papers/10 clusters IS ~10 papers per cluster.
# and it scales by itself: 2000 papers -> 200 clusters -> still ~10 each,
# 5286 papers -> 529 clusters -> still ~10 each. the 10 is an AVERAGE, ward
# cuts where topics really break, so real sizes spread around it (e.g. 2-44)
PAPERS_PER_CLUSTER = 10

# every room must hold at least this many papers. 1 = no empty rooms
MIN_PER_ROOM = 1

# how many independent tries grasp gets. we keep the best one.
# a few long tries beats many short ones when all we want is one good layout
N_SEEDS = 5


def auto_time_budget(n_papers):
    """seconds of grasp search per try, scaled to the input size.
    rough rule: 1 second per 100 papers, never below 10s, never above 300s.
    at the thesis scale (~5300 papers) this gives ~53s, close to the 60s that
    produced a healthy benchmark. a starting rule, can be tuned later"""
    return float(min(300.0, max(10.0, n_papers / 100.0)))


def main():
    # ---- step 1: read what the user gave us ----

    # the flags the user can pass on the command line
    ap = argparse.ArgumentParser(
        description="assign papers to rooms so each room is topically coherent")
    ap.add_argument("--papers", default=None, help="csv file with the papers")
    ap.add_argument("--rooms", default=None,
                    help="room sizes, comma separated, e.g. 700,600,550")
    ap.add_argument("--abstract-col", default="Abstract",
                    help="name of the abstract column (default: Abstract)")
    ap.add_argument("--id-col", default=None,
                    help="name of the paper id column (default: row numbers)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: a new Production N folder next to main.py)")
    args = ap.parse_args()

    # interactive fallback: if the program was started without the flags (for
    # example with the editors play button), it asks for them instead of dying.
    # you can drag the csv file straight onto the terminal to answer the first
    # question - the terminal types the path for you
    if args.papers is None:
        raw = input("where is your papers csv? (drag the file here, then press enter): ").strip()
        # drag and drop pastes the path differently per system: windows wraps it
        # in quotes, mac puts a backslash before every space. clean both so any
        # style works
        args.papers = raw.strip('"').strip("'").replace("\\ ", " ")
    if args.rooms is None:
        args.rooms = input("room sizes, comma separated (e.g. 700,600,550): ").strip()

    # stop early with a clear message if the file isnt there
    papers_path = Path(args.papers)
    if not papers_path.exists():
        print(f"could not find the papers file: {papers_path}")
        return 1

    # where the results will go. every run gets its own fresh numbered folder
    # next to main.py (Production 1, Production 2, ...) so no run ever
    # overwrites an earlier one. --out still overrides this if given
    if args.out:
        out_dir = Path(args.out)
    else:
        run_number = 1
        while (BASE_DIR / f"Production {run_number}").exists():
            run_number += 1
        out_dir = BASE_DIR / f"Production {run_number}"
    out_dir.mkdir(exist_ok=True)

    # room sizes come in as one string like "700,600,550" - turn it into numbers
    try:
        caps_list = [int(x) for x in args.rooms.split(",") if x.strip()]
    except ValueError:
        print(f"could not read the room sizes '{args.rooms}' - "
              "give whole numbers separated by commas, e.g. 700,600,550")
        return 1

    # ---- step 2: check the input makes sense before doing any heavy work ----

    # open the users csv into a table
    df = pd.read_csv(papers_path)

    if args.abstract_col not in df.columns:
        print(f"could not find an abstract column called '{args.abstract_col}'. "
              f"the file has these columns: {list(df.columns)}")
        return 1

    # papers without an abstract cant be embedded, so they get dropped with a note
    before = len(df)
    df = df.dropna(subset=[args.abstract_col])
    df = df[df[args.abstract_col].astype(str).str.strip().astype(bool)]
    if len(df) < before:
        print(f"note: dropped {before - len(df)} rows with an empty abstract")
    if len(df) == 0:
        print("no papers with abstracts were found in the file.")
        return 1

    if args.id_col is not None and args.id_col not in df.columns:
        print(f"no id column called '{args.id_col}'. "
              f"the file has: {list(df.columns)}")
        return 1

    n = len(df)
    caps = np.array(caps_list)
    R = len(caps)
    room_names = [f"Room {i + 1}" for i in range(R)]

    # room sizes like 0 or -5 make no sense
    if R == 0 or np.any(caps <= 0):
        print(f"room sizes must be positive whole numbers, got: {caps_list}")
        return 1

    # the check that matters most: every paper needs a seat somewhere
    if caps.sum() < n:
        print(f"not enough seats: {n} papers but only {caps.sum()} seats across "
              f"{R} rooms. add capacity or remove papers.")
        return 1
    if MIN_PER_ROOM * R > n:
        print(f"cannot put at least {MIN_PER_ROOM} paper(s) into each of {R} rooms "
              f"with only {n} papers.")
        return 1

    print(f"{n} papers -> {R} rooms ({caps.sum()} seats, {caps.sum() - n} spare)")

    # ---- step 3: decide the settings from the data itself ----

    # cluster count follows the papers (about 10 per cluster), but never fewer
    # clusters than rooms - every room needs at least one cluster to hold
    n_clusters = max(R, min(n, round(n / PAPERS_PER_CLUSTER)))
    budget = auto_time_budget(n)
    print(f"clustering into {n_clusters} groups; "
          f"grasp gets {N_SEEDS} tries x {budget:.0f}s each")

    # ---- step 4: abstracts -> vectors ----

    # the heavy step: every abstract becomes one vector of 384 numbers
    t0 = time.perf_counter()
    emb = embed(df[args.abstract_col].astype(str).tolist())
    print(f"embedded in {time.perf_counter() - t0:.1f}s")

    # ---- step 5: vectors -> clusters ----

    # the tree is built once. if a cluster turns out bigger than the largest
    # room, no layout can exist, so we re-cut the same tree a bit finer until
    # every cluster fits. re-cutting is nearly free, rebuilding is not
    tree = build_tree(emb)
    labels = cut_tree(tree, n_clusters)
    while True:
        cluster_ids, sizes, centroids, clusters_pos = cluster_info(emb, labels)
        if sizes.max() <= caps.max() or n_clusters >= n:
            break
        n_clusters = min(n, n_clusters + max(1, n_clusters // 5))
        print(f"  a cluster was bigger than the largest room - re-cutting into {n_clusters}")
        labels = cut_tree(tree, n_clusters)
    W = similarity_table(centroids)
    print(f"{len(cluster_ids)} clusters (sizes {sizes.min()}-{sizes.max()})")

    # ---- step 6: clusters -> rooms, with grasp ----

    # grasp doesnt compute paper similarity itself. we hand it this small
    # function from quality so every try gets a similarity number in the
    # results - for reporting only, grasp never optimizes on it
    sim_fn = lambda a: paper_similarity_of(a, clusters_pos, emb, R)
    print(f"packing clusters into rooms...")
    best, seed_rows = grasp(sizes, caps, W, MIN_PER_ROOM, budget,
                            list(range(N_SEEDS)), similarity_fn=sim_fn, verbose=True)
    if best is None:
        print("could not build a valid layout - these room sizes cannot hold the "
              "paper groups. try adding capacity.")
        return 1

    # ---- step 7: measure, save, report ----

    # best says where each CLUSTER goes. clusters_pos translates that into
    # where each PAPER goes
    paper_room_idx = best[clusters_pos]

    # the honest quality measure, computed on the papers themselves
    per_room_sim, overall = within_room_similarity(paper_room_idx, emb, R)

    # what a random shuffle into these same room sizes would score - the
    # comparison point that makes the similarity number readable
    room_sizes = [int((paper_room_idx == r).sum()) for r in range(R)]
    baseline = random_baseline_similarity(room_sizes, emb, R)

    # the deliverable: one row per paper, which room it goes to
    ids = df[args.id_col] if args.id_col else pd.Series(range(1, n + 1), name="paper")
    layout = pd.DataFrame({
        (args.id_col or "paper"): ids.values,
        "room": [room_names[r] for r in paper_room_idx],
    })
    layout.to_csv(out_dir / "layout.csv", index=False)

    # the per-room table: capacity, papers, fill, how coherent each room is
    summary = room_summary(best, cluster_ids, sizes, caps, room_names, per_room_sim)
    summary.to_csv(out_dir / "room_summary.csv", index=False)

    # a short text report so the user can judge the layout at a glance
    report = (
        f"source: {papers_path.name}\n"
        f"papers: {n}   rooms: {R}   seats: {int(caps.sum())}\n"
        f"clusters: {len(cluster_ids)} (~{PAPERS_PER_CLUSTER} papers each)\n"
        f"method: grasp, {N_SEEDS} tries x {budget:.0f}s, best kept\n\n"
        f"average within-room paper similarity: {overall:.4f}\n"
        f"random-assignment baseline:           {baseline:.4f}\n"
        f"lift over random:                     {overall - baseline:+.4f}\n"
    )
    (out_dir / "quality_report.txt").write_text(report)

    print("\n=== room summary ===")
    print(summary[["room", "capacity", "papers", "fill_pct", "n_clusters",
                   "avg_paper_similarity"]].to_string(index=False))
    print("\n" + report)
    print(f"saved to {out_dir}: layout.csv, room_summary.csv, quality_report.txt")
    return 0


# run main() when this file is executed, and hand its result back to the
# system: 0 means everything went fine, 1 means an input problem was found
if __name__ == "__main__":
    raise SystemExit(main())