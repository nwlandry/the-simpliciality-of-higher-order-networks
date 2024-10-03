from .mean_face_edit_distance import mean_face_edit_distance


def face_edit_simpliciality(H, min_size=2, exclude_min_size=True):
    """Computes the face edit simpliciality.

    The average number of edges needed to be added
    to a hyperedge to make it a simplex.

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
        The face edit simpliciality

    See Also
    --------
    mean_face_edit_distance

    References
    ----------
    "The simpliciality of higher-order order networks"
    by Nicholas Landry, Jean-Gabriel Young, and Nicole Eikmeier,
    *EPJ Data Science* **13**, 17 (2024).
    """
    return 1 - mean_face_edit_distance(
        H, min_size=min_size, exclude_min_size=exclude_min_size
    )
