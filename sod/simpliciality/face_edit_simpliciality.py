from .mean_face_edit_distance import mean_face_edit_distance


def face_edit_simpliciality(H, min_size=2, exclude_min_size=True):
    """Computes the face edit simpliciality.

    The average fraction of sub-edges contained in a hyperedge
    relative to a simplex.

    Parameters
    ----------
    H : xgi.Hypergraph
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
        The face edit simpliciality

    See Also
    --------
    mean_face_edit_distance

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
    return 1 - mean_face_edit_distance(
        H, min_size=min_size, exclude_min_size=exclude_min_size
    )
