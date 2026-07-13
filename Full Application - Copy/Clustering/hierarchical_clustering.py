# clustering.py - step 3 of the pipeline: vectors go in, clusters come out
#
# ward clustering builds a family tree of all papers, bottom up: the two most
# similar groups merge first, then the next, until everything is one big group.
# we then cut that tree at a chosen height and get n_clusters small topic boxes.
#
# this file also computes what grasp needs to know about the clusters:
# how many papers each one holds, its centroid (the average direction of its
# papers), and the cluster-vs-cluster similarity table W. that way grasp only
# has to decide which cluster goes into which room, nothing else.
#
# strict rule: this file knows nothing about csvs, users, rooms or paths.
# main.py hands it the vectors, it hands back clusters. data in, data out.

import numpy as np


def build_tree(embeddings):
    """builds the ward merge tree once. cutting it later is nearly free."""

    # scipy is heavy and slow to load, so we import it only when this function
    # is actually called. never called = never loaded. the slow loading only
    # happens on the first call - after that its already in memory
    from scipy.cluster.hierarchy import linkage

    # ward wants the raw vectors, not a similarity matrix. and because our
    # vectors are length 1, euclidean distance and cosine similarity agree -
    # so this really is grouping by topic. dont feed it anything unnormalized
    return linkage(embeddings, method="ward")


def cut_tree(tree, n_clusters):
    """cuts the finished tree into n_clusters groups."""
    from scipy.cluster.hierarchy import fcluster

    # returns one label per paper, in the same order as the papers went in
    # (row 7 of the embeddings = label 7 here). labels start at 1, not 0
    return fcluster(tree, t=n_clusters, criterion="maxclust")


def cluster(embeddings, n_clusters):
    """convenience: build the tree and cut it in one call."""
    return cut_tree(build_tree(embeddings), n_clusters)


def cluster_info(embeddings, labels):
    """describes the clusters for grasp: sizes, centroids, and which cluster
    each paper sits in. returns (cluster_ids, sizes, centroids, clusters_pos)."""

    labels = np.asarray(labels)

    # the distinct cluster labels, sorted (1, 2, 3, ...)
    cluster_ids = np.unique(labels)
    C = len(cluster_ids)

    sizes = np.zeros(C, dtype=int)
    centroids = np.zeros((C, embeddings.shape[1]), dtype=np.float32)

    for i, c in enumerate(cluster_ids):
        mask = labels == c

        # size = simply how many papers carry this label
        sizes[i] = mask.sum()

        # centroid = the average of the clusters vectors, then scaled back to
        # length 1 (averaging shortens vectors a bit, and all our similarity
        # math assumes length 1). the +1e-12 just makes sure we never divide
        # by zero, do not remove it
        cen = embeddings[mask].mean(axis=0)
        centroids[i] = cen / (np.linalg.norm(cen) + 1e-12)

    # clusters_pos: for every paper, the row number of its cluster in
    # sizes/centroids. this is the link that lets us translate a cluster->room
    # answer back into a paper->room answer at the end
    id_to_pos = {c: i for i, c in enumerate(cluster_ids)}
    clusters_pos = np.array([id_to_pos[c] for c in labels])

    return cluster_ids, sizes, centroids, clusters_pos


def similarity_table(centroids):
    """the table W that grasp optimizes on: W[a][b] = how alike clusters a and b are."""

    # centroids are length 1, so a plain matrix multiply gives all the cosine
    # similarities at once. negatives get clipped to 0 (rare and tiny for
    # abstracts, and grasp only rewards positive togetherness anyway)
    W = np.maximum(centroids @ centroids.T, 0.0)

    # zero the diagonal so a cluster cant score points for being with itself
    np.fill_diagonal(W, 0.0)
    return W