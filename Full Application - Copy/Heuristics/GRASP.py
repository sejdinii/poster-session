# grasp.py - step 4 of the pipeline: clusters go in, a room layout comes out
#
# the job: put whole clusters into rooms so that similar clusters end up
# together, without ever going over a rooms seat limit. this is the method
# that won my benchmark against four other heuristics, so its the one the
# tool uses.
#
# how grasp works in one breath: build a decent layout with a bit of
# randomness, polish it with small moves until it stops improving, write down
# the score. then throw it away and start fresh. repeat until the time budget
# runs out, keep the best attempt seen.
#
# strict rule: this file knows nothing about csvs, users or paths. it receives
# cluster sizes, room capacities and the similarity table W, and returns which
# cluster goes into which room. data in, data out.

import time
import random

import numpy as np


# ---------------------------------------------------------------
# piece 1: the two small helpers
# ---------------------------------------------------------------

def cluster_objective(assign, W):
    """the score of a layout: total similarity of all cluster pairs that share a room."""

    # assign is one room number per cluster, e.g. assign[4] = 2 means
    # cluster 4 sits in room 2
    R = int(assign.max()) + 1 if len(assign) else 0

    total = 0.0
    for r in range(R):
        # who lives in room r
        members = np.where(assign == r)[0]

        # add up the similarity of every pair inside the room. the /2 is there
        # because the table W counts each pair twice (a-with-b and b-with-a)
        if len(members) > 1:
            total += W[np.ix_(members, members)].sum() / 2.0
    return total

    # note: the score only rewards putting similar clusters together. a room
    # gets zero points for simply being full - fullness is handled by the
    # seat limits, not by the score


def room_loads(assign, sizes, R):
    """counts how many papers currently sit in each room."""
    loads = np.zeros(R, dtype=int)
    for i in range(len(assign)):
        loads[assign[i]] += int(sizes[i])
    return loads


# ---------------------------------------------------------------
# piece 2: construct - build one complete legal layout from scratch
# ---------------------------------------------------------------

def construct(sizes, caps, W, min_per_room, rcl_size=3):
    """builds one full layout, greedy but with a little randomness.
    returns the layout, or None if it painted itself into a corner
    (the caller just tries again - thats normal, not an error)."""

    C = len(sizes)
    R = len(caps)

    # -1 means "not placed yet"
    assign = np.full(C, -1, dtype=int)
    loads = np.zeros(R, dtype=int)

    # we place big clusters first - big ones are the hardest to fit, so give
    # them first pick while rooms are still empty
    order = sorted(range(C), key=lambda k: -sizes[k])

    # phase 1: make every room legal. each room must reach min_per_room papers
    # (with min_per_room = 1 this just means: no empty rooms)
    for r in range(R):
        while loads[r] < min_per_room:
            # which unplaced clusters would fit in this room
            fitting = [i for i in order if assign[i] == -1 and loads[r] + sizes[i] <= caps[r]]
            if not fitting:
                return None
            # pick randomly among the top few candidates instead of always the
            # single best. this is the randomness that makes every attempt
            # different - without it, all attempts would be identical
            i = random.choice(fitting[:rcl_size])
            assign[i] = r
            loads[r] += int(sizes[i])

    # phase 2: place everything that is still unplaced, biggest first.
    # for each cluster, score every room by how similar the cluster is to the
    # clusters already inside, then pick randomly among the top few rooms
    for i in order:
        if assign[i] != -1:
            continue
        feasible = [r for r in range(R) if loads[r] + sizes[i] <= caps[r]]
        if not feasible:
            return None
        scored = []
        for r in feasible:
            members = np.where(assign == r)[0]
            gain = W[i, members].sum() if len(members) > 0 else 0.0
            scored.append((gain, r))
        scored.sort(reverse=True)
        _, chosen = random.choice(scored[:rcl_size])
        assign[i] = chosen
        loads[chosen] += int(sizes[i])

    return assign


# ---------------------------------------------------------------
# piece 3: improve - polish a finished layout with small moves
# ---------------------------------------------------------------

def local_search(assign, sizes, caps, W, min_per_room):
    """takes a complete layout and keeps making small improving moves until
    nothing helps anymore. two kinds of moves: relocate one cluster to another
    room, or swap two clusters between rooms."""

    C = len(sizes)
    R = len(caps)
    loads = room_loads(assign, sizes, R)

    improved = True
    while improved:
        improved = False

        # move 1: relocate. try taking each cluster out of its room and
        # putting it somewhere it fits better
        for i in range(C):
            r_from = assign[i]

            # dont empty a room below the minimum, this check keeps every
            # move legal - do not remove it
            if loads[r_from] - sizes[i] < min_per_room:
                continue

            # what the cluster is worth where it is now
            loss = W[i, np.where(assign == r_from)[0]].sum()

            for r_to in range(R):
                if r_to == r_from or loads[r_to] + sizes[i] > caps[r_to]:
                    continue
                # move only if the new room is a strictly better fit
                if W[i, np.where(assign == r_to)[0]].sum() - loss > 1e-9:
                    assign[i] = r_to
                    loads[r_from] -= sizes[i]
                    loads[r_to]   += sizes[i]
                    improved = True
                    break

        # move 2: swap. try exchanging two clusters that sit in different
        # rooms - useful when neither could relocate alone (no free seats)
        # but trading places helps both
        for i in range(C):
            for j in range(i + 1, C):
                ri, rj = assign[i], assign[j]
                if ri == rj:
                    continue

                # seat counts after the trade - both rooms must stay legal,
                # do not remove these checks
                new_ri = loads[ri] - sizes[i] + sizes[j]
                new_rj = loads[rj] - sizes[j] + sizes[i]
                if new_ri > caps[ri] or new_rj > caps[rj]:
                    continue
                if new_ri < min_per_room or new_rj < min_per_room:
                    continue

                # score before vs after the trade (mj without j and mi
                # without i, since each is leaving that room)
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


# ---------------------------------------------------------------
# piece 4: the grasp loop - construct, improve, repeat, keep the best
# ---------------------------------------------------------------

def grasp(sizes, caps, W, min_per_room, time_budget, seeds,
          rcl_size=3, verbose=True, similarity_fn=None):
    """runs the whole method. for every seed: build + polish layouts over and
    over until time_budget seconds are up, keep the best one seen.
    returns (best_layout, seed_rows) where seed_rows is one result dict per seed.

    similarity_fn is optional: a function that scores a layout by paper-level
    similarity, used only for reporting - grasp itself never looks at it."""

    seed_rows = []
    global_best_obj = -1.0
    global_best = None

    for seed in seeds:
        # same seed = same sequence of random choices = same result on the
        # same machine. this is what makes a run repeatable
        random.seed(seed)
        np.random.seed(seed)

        seed_best_obj = -1.0
        seed_best = None
        restarts = 0
        t_seed = time.perf_counter()

        # the alarm clock: keep attempting until the budget is used up.
        # the check happens BETWEEN attempts, so a running attempt always
        # finishes first - the run can overshoot the budget by a moment
        while time.perf_counter() - t_seed < time_budget:
            a = construct(sizes, caps, W, min_per_room, rcl_size)
            if a is None:
                continue     # corner case, just try again
            a = local_search(a, sizes, caps, W, min_per_room)
            obj = cluster_objective(a, W)
            restarts += 1
            if obj > seed_best_obj:
                seed_best_obj = obj
                seed_best = a.copy()

        # one result row per seed, for the report at the end
        row = {"seed": seed, "objective": round(seed_best_obj, 4),
               "restarts": restarts,
               "seconds": round(time.perf_counter() - t_seed, 2)}
        if similarity_fn is not None and seed_best is not None:
            row["similarity"] = round(float(similarity_fn(seed_best)), 4)
        seed_rows.append(row)

        if verbose:
            sim_txt = f"  similarity {row['similarity']:.4f}" if "similarity" in row else ""
            print(f"  seed {seed:>2}: objective {seed_best_obj:8.2f}{sim_txt}  "
                  f"({restarts} restarts)")

        if seed_best_obj > global_best_obj:
            global_best_obj = seed_best_obj
            global_best = seed_best.copy()

    return global_best, seed_rows