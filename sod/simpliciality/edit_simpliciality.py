import numpy as np
import xgi

from ..trie import Trie
from .utilities import count_subfaces, max_number_of_subfaces, missing_subfaces


def edit_simpliciality(H, min_size=2, exclude_min_size=True):
    """Computes the simplicial edit distance.

    The number of edges needed to be added
    to a hypergraph to make it a simplicial complex.

    Parameters
    ----------
    H : xgi.Hypergraph
        The hypergraph of interest
    min_size: int, default: 1
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces.

    Returns
    -------
    int
        The edit simpliciality
    """
    edges = H.edges.filterby("size", min_size, "geq").members()

    t = Trie()
    t.build_trie(edges)

    maxH = xgi.Hypergraph(
        H.edges.maximal()
        .filterby("size", min_size + exclude_min_size, "geq")
        .members(dtype=dict)
    )
    ms = 0
    for id1, e in maxH.edges.members(dtype=dict).items():
        redundant_missing_faces = set()
        for id2 in maxH.edges.neighbors(id1):
            if id2 < id1:
                c = maxH._edge[id2].intersection(e)
                if len(c) >= min_size:
                    redundant_missing_faces.update(missing_subfaces(t, c, min_size))

                    # we don't have to worry about the intersection being a max face
                    # because a) there are no multiedges and b) these are all maximal
                    # faces so no inclusions.
                    if not t.search(c):
                        redundant_missing_faces.add(frozenset(c))

        nm = max_number_of_subfaces(min_size, len(e))
        nf = count_subfaces(t, e, min_size)
        rmf = len(redundant_missing_faces)
        ms += nm - nf - rmf

    try:
        s = len(edges)
        return s / (ms + s)
    except ZeroDivisionError:
        return np.nan


def edit_simpliciality_full_construction(H, min_size=2, exclude_min_size=True):
    """Computes the simplicial edit distance.

    The number of edges needed to be added
    to a hypergraph to make it a simplicial complex.

    Parameters
    ----------
    H : xgi.Hypergraph
        The hypergraph of interest
    min_size: int, default: 1
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces.

    Returns
    -------
    int
        The edit simpliciality
    """
    edges = H.edges.filterby("size", min_size, "geq").members()
    max_edges = (
        H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
    )

    t = Trie()
    t.build_trie(edges)

    ms = set()
    for e in max_edges:
        ms.update(missing_subfaces(t, e, min_size=min_size))

    try:
        s = len(edges)
        m = len(ms)
        return s / (m + s)
    except ZeroDivisionError:
        return np.nan
