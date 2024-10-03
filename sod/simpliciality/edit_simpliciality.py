import numpy as np

from ..trie import Trie
from .simplicial_edit_distance import simplicial_edit_distance
from .utilities import missing_subfaces


def edit_simpliciality(H, min_size=2, exclude_min_size=True):
    """Computes the edit simpliciality.

    The number of edges needed to be added
    to a hypergraph to make it a simplicial complex.

    Parameters
    ----------
    H : xgi.Hypergraph
        The hypergraph of interest
    min_size: int, default: 2
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces.
    exclude_min_size : bool, optional
        Whether to include minimal simplices when counting simplices, by default True

    Returns
    -------
    float
        The edit simpliciality

    See Also
    --------
    simplicial_edit_distance

    References
    ----------
    "The simpliciality of higher-order order networks"
    by Nicholas Landry, Jean-Gabriel Young, and Nicole Eikmeier,
    *EPJ Data Science* **13**, 17 (2024).
    """
    return 1 - simplicial_edit_distance(
        H, min_size=min_size, exclude_min_size=exclude_min_size
    )


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

    s = len(edges)
    mf = len(max_edges)
    m = len(ms)
    if m + s - mf > 0:
        return (s - mf) / (m + s - mf)
    else:
        return np.nan
