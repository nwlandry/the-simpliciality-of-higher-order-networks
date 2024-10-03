import numpy as np
import xgi

from ..trie import Trie
from .utilities import count_missing_subfaces, missing_subfaces


def simplicial_edit_distance(H, min_size=2, exclude_min_size=True, normalize=True):
    """Computes the simplicial edit distance.

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
    normalize : bool, optional
        Whether to normalize by the total number of edges

    Returns
    -------
    float
        The edit simpliciality

    See Also
    --------
    edit_simpliciality

    References
    ----------
    "The simpliciality of higher-order order networks"
    by Nicholas Landry, Jean-Gabriel Young, and Nicole Eikmeier,
    *EPJ Data Science* **13**, 17 (2024).
    """
    edges = H.edges.filterby("size", min_size, "geq").members()

    t = Trie()
    t.build_trie(edges)

    maxH = xgi.Hypergraph(
        H.edges.maximal()
        .filterby("size", min_size + exclude_min_size, "geq")
        .members(dtype=dict)
    )
    if not maxH.edges:
        return np.nan

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

        mf = count_missing_subfaces(t, e, min_size)
        rmf = len(redundant_missing_faces)
        ms += mf - rmf

    if normalize:
        s = len(edges)
        mf = maxH.num_edges
        if s - mf + ms > 0:
            return ms / (s - mf + ms)
        else:
            return np.nan
    else:
        return ms
