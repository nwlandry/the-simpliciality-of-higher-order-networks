import numpy as np

from ..trie import Trie
from .utilities import powerset


def simplicial_fraction(H, min_size=2, exclude_min_size=True):
    try:
        ns = count_simplices(H, min_size, exclude_min_size)
        ps = potential_simplices(H, min_size, exclude_min_size)
        return ns / ps
    except ZeroDivisionError:
        return np.nan


def potential_simplices(H, min_size=2, exclude_min_size=True):
    # record total number of hyperedges that are potential simplices
    return len(H.edges.filterby("size", min_size + exclude_min_size, "geq"))


def count_simplices(H, min_size=2, exclude_min_size=True):
    # build trie data structure
    t = Trie()
    all_edges = H.edges.members()
    t.build_trie(all_edges)

    edges = H.edges.filterby("size", min_size + exclude_min_size, "geq").members()

    # for each hyperedge, determine if it's a simplex
    count = 0
    # The following loop is embarassingly parallel, so parallelize to increase speed would be good
    for e in edges:
        if is_simplex(t, e, min_size):
            count += 1
    return count


def is_simplex(t, edge, min_size=2):
    for e in powerset(edge, min_size):
        if not t.search(e):
            return False
    return True
