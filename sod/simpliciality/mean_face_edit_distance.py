from ..trie import Trie
from .utilities import count_missing_subfaces, max_number_of_subfaces


def mean_face_edit_distance(H, min_size=1, exclude_min_size=True, normalize=True):
    """Computes the mean face edit distance

    Parameters
    ----------
    H : Hypergraph
        The hypergraph of interest
    min_size : int, optional
        The minimum size to be considered a simplex, by default 2
    exclude_min_size : bool, optional
        Whether to include minimal simplices when counting simplices, by default True
    normalize : bool, optional
        Whether to normalize the face edit distance, by default True

    Returns
    -------
    float
        The mean face edit distance

    See Also
    --------
    face_edit_simpliciality

    References
    ----------
    "The simpliciality of higher-order order networks"
    by Nicholas Landry, Jean-Gabriel Young, and Nicole Eikmeier,
    *EPJ Data Science* **13**, 17 (2024).
    """
    t = Trie()
    t.build_trie(H.edges.filterby("size", min_size, "geq").members())

    max_faces = (
        H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
    )
    avg_d = 0
    for e in max_faces:
        if len(e) >= min_size:
            d = count_missing_subfaces(t, e, min_size=min_size)  # missing subfaces
            m = max_number_of_subfaces(min_size, len(e))
            if normalize and m != 0:
                d *= 1.0 / m
            avg_d += d / len(max_faces)
    return avg_d
