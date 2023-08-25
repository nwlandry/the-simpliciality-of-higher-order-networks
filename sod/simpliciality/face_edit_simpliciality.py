import numpy as np

from ..trie import Trie
from .utilities import count_subfaces, max_number_of_subfaces


def face_edit_simpliciality(H, min_size=2, exclude_min_size=True):
    edges = (
        H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
    )
    t = Trie()
    t.build_trie(H.edges.filterby("size", min_size, "geq").members())

    if not edges:
        return np.nan

    fes = 0
    for e in edges:
        n = count_subfaces(t, e, min_size=min_size)
        d = max_number_of_subfaces(min_size, len(e))
        # happens when you include the minimal faces when counting simplices
        try:
            fes += float(n / (d * len(edges)))
        except ZeroDivisionError:
            fes += 1.0 / len(edges)
    return fes
