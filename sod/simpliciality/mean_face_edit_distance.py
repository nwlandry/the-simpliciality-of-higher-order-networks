from ..trie import Trie
from .utilities import count_subfaces, max_number_of_subfaces


def mean_face_edit_distance(H, min_size=1, exclude_min_size=True, normalize=True):
    t = Trie()
    t.build_trie(H.edges.filterby("size", min_size, "geq").members())

    max_faces = (
        H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
    )
    avg_d = 0
    for e in max_faces:
        if len(e) >= min_size:
            s = count_subfaces(t, e, min_size=min_size)
            m = max_number_of_subfaces(min_size, len(e))
            d = m - s
            if normalize:
                d *= 1.0 / m
            avg_d += d / len(max_faces)
    return avg_d
