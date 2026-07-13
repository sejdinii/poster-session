# layout_evaluation.py - step 5 of the pipeline: judges how good a layout really is
#
# grasp optimizes a score built on cluster centroids, which is a shortcut.
# this file measures the real thing: how similar are the ACTUAL papers that
# ended up sharing a room. it also answers "compared to what?" by scoring a
# random shuffle of the same papers into the same rooms - without that
# comparison point, a similarity number on its own means nothing.
#
# strict rule: this file knows nothing about csvs, users, rooms or paths.
# main.py hands it layouts and vectors, it hands back numbers and a table.

import numpy as np
import pandas as pd


def within_room_similarity(paper_room_idx, emb, R):
    """for each room: the average similarity between every pair of papers
    inside it. returns (one number per room, one overall number)."""

    per_room = np.full(R, np.nan)
    total_sim = 0.0
    total_pairs = 0.0

    for r in range(R):
        # which papers sit in room r
        idx = np.where(paper_room_idx == r)[0]
        nr = len(idx)

        # a room with 0 or 1 papers has no pairs to measure
        if nr < 2:
            continue

        # comparing every pair one by one would be millions of operations for
        # a big room. theres a shortcut: the sum of all pairwise dot products
        # can be computed from just the sum of the vectors -
        #   sum over pairs = (length of the summed vector squared
        #                     - sum of each vectors squared length) / 2
        # this only equals cosine similarity because our vectors are length 1,
        # so dont feed this anything unnormalized
        E = emb[idx]
        S = E.sum(axis=0)
        sim_sum = (float(S @ S) - float((E * E).sum())) / 2.0

        # number of pairs in a room of nr papers
        pairs = nr * (nr - 1) / 2.0

        per_room[r] = sim_sum / pairs
        total_sim += sim_sum
        total_pairs += pairs

    # the overall number weighs big rooms more, since they hold more pairs
    overall = total_sim / total_pairs if total_pairs > 0 else float("nan")
    return per_room, overall


def random_baseline_similarity(room_sizes, emb, R, n_trials=5, seed=42):
    """what a random shuffle of the same papers into the same room sizes
    scores. this is the comparison point that makes the real number readable:
    0.38 alone says little, 0.38 vs 0.11 random says the grouping works."""

    # fixed seed so the baseline is the same every run - repeatable reports
    rng = np.random.default_rng(seed)
    N = len(emb)
    vals = []

    for _ in range(n_trials):
        # deal the papers randomly into rooms of exactly the same sizes
        perm = rng.permutation(N)
        assign = np.empty(N, dtype=int)
        start = 0
        for r in range(R):
            assign[perm[start:start + room_sizes[r]]] = r
            start += room_sizes[r]

        _, overall = within_room_similarity(assign, emb, R)
        vals.append(overall)

    # average over a few shuffles so one lucky/unlucky deal doesnt mislead
    return float(np.mean(vals))


def paper_similarity_of(assign, clusters_pos, embeddings, R):
    """convenience: takes a cluster->room layout, translates it to paper->room
    using clusters_pos, and returns the overall paper similarity. this is the
    small function main hands to grasp for per-try reporting."""
    paper_room_idx = assign[clusters_pos]
    _, overall = within_room_similarity(paper_room_idx, embeddings, R)
    return overall


def _room_loads(assign, sizes, R):
    # tiny local seat counter. grasp has its own copy of this - kept separate
    # on purpose so the step files never depend on each other
    loads = np.zeros(R, dtype=int)
    for i in range(len(assign)):
        loads[assign[i]] += int(sizes[i])
    return loads


def room_summary(assign, cluster_ids, sizes, caps, room_names, per_room_sim):
    """the human-readable table: one row per room with capacity, papers placed,
    fill percent, how many clusters, that rooms similarity, and which clusters."""

    R = len(caps)
    loads = _room_loads(assign, sizes, R)
    rows = []

    for r_i, r in enumerate(room_names):
        # which clusters ended up in this room
        placed = [cluster_ids[i] for i in range(len(cluster_ids)) if assign[i] == r_i]
        rows.append({
            "room": r,
            "capacity": int(caps[r_i]),
            "papers": int(loads[r_i]),
            "fill_pct": round(100 * loads[r_i] / caps[r_i], 1),
            "n_clusters": len(placed),
            "avg_paper_similarity": round(float(per_room_sim[r_i]), 4),
            "clusters": ", ".join(map(str, placed)),
        })

    return pd.DataFrame(rows)