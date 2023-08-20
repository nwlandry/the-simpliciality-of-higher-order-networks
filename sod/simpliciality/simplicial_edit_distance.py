import numpy as np

from ..trie import Trie
from .utilities import missing_subfaces


def simplicial_edit_distance(H, min_size=2, exclude_min_size=True, normalize=True):
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

    edges = (
        H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
    )

    t = Trie()
    t.build_trie(H.edges.members())

    ms = set()
    for e in edges:
        ms.update(missing_subfaces(H, e, min_size=min_size))
    try:
        s = H.num_edges
        m = len(ms)

        if normalize:
            return m / (m + s)
        else:
            return m
    except ZeroDivisionError:
        return np.nan
