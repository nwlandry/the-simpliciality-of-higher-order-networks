import numpy as np

from ..trie import Trie
from .utilities import powerset


def simplicial_fraction(H, min_size=2, exclude_min_size=True):
    """Computing the simplicial fraction for a hypergraph.

    What fraction of the hyperedges are simplices?

    Parameters
    ----------
    H : Hypergraph
        The hypergraph of interest
    min_size: int, optional
        The minimum hyperedge size to include when
        calculating whether a hyperedge is a simplex
        by counting subfaces. For more details, see
        the Notes below. By default, 2.
    exclude_min_size : bool, optional
        Whether to exclude minimal simplices when counting simplices.
        For more detailed information, see the Notes below. By default, True.

    Returns
    -------
    float
        The simplicial fraction

    Notes
    -----
    1. The formal definition of a simplicial complex can be unnecessarily
    strict when used to represent perfect inclusion structures.
    By definition, a simplex always contains singletons
    (edges comprising a single node) and the empty set.
    Several datasets will not include such interactions by construction.
    To circumvent this issue, we use a relaxed definition of
    downward closure that excludes edges of a certain size or smaller
    wherever it makes sense. By default, we set the minimum size
    to be 2 since some datasets do not contain singletons.

    2. Hyperedges we call “minimal faces” may significantly skew the
    simpliciality of a dataset. The minimal faces of a hypergraph :math:`H`
    are the edges of the minimal size, i.e., :math:`|e| = min(K)`, where :math:`K`
    is the set of sizes that we consider based on note 1.
    (In a traditional simplicial complex, the minimal faces are singletons).
    With the size restrictions in place, the minimal faces of a hypergraph
    are always simplices because, by definition, there are no smaller edges
    for these edges to include. When measuring the simpliciality of a dataset,
    it is most meaningful to focus on the faces for which inclusion is possible,
    and so, by default, we exclude these minimal faces when counting potential
    simplices.

    References
    ----------
    "The simpliciality of higher-order order networks"
    by Nicholas Landry, Jean-Gabriel Young, and Nicole Eikmeier,
    *EPJ Data Science* **13**, 17 (2024).
    """
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
