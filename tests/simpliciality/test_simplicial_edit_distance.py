import numpy as np

from sod import *


def test_simplicial_edit_distance(
    sc1_with_singletons,
    h_missing_one_singleton,
    h_missing_one_link,
    h_links_and_triangles2,
    h1,
):
    # simplicial complex
    sed = simplicial_edit_distance(sc1_with_singletons)
    assert sed == 0.0

    sed = simplicial_edit_distance(sc1_with_singletons, min_size=1)
    assert sed == 0.0

    sed = simplicial_edit_distance(
        sc1_with_singletons, min_size=1, exclude_min_size=False
    )
    assert sed == 0.0

    # h1
    sed = simplicial_edit_distance(h_missing_one_singleton)
    assert sed == 0.0

    sed = simplicial_edit_distance(h_missing_one_singleton, min_size=1)
    assert np.allclose(sed, 1 / 6)

    sed = simplicial_edit_distance(
        h_missing_one_singleton, min_size=1, exclude_min_size=False
    )
    assert np.allclose(sed, 1 / 6)

    # h2
    sed = simplicial_edit_distance(h_missing_one_link)
    assert np.allclose(sed, 1 / 3)

    sed = simplicial_edit_distance(h_missing_one_link, min_size=1)
    assert np.allclose(sed, 1 / 6)

    # links and triangles 2
    sed = simplicial_edit_distance(h_links_and_triangles2)
    assert np.allclose(sed, 1 / 3)

    sed = simplicial_edit_distance(h_links_and_triangles2, min_size=1)
    assert np.allclose(sed, 2 / 3)

    sed = simplicial_edit_distance(h_links_and_triangles2, exclude_min_size=False)
    assert np.allclose(sed, 2 / 5)

    # test h1
    sed = simplicial_edit_distance(h1)
    s = 4
    m = 4 + 10
    mf = 3
    assert np.allclose(sed, m / (m + s - mf))

    sed = simplicial_edit_distance(h1, min_size=1)
    s = 4
    m = 4 + 10 + 7
    mf = 3
    assert np.allclose(sed, m / (m + s - mf))

    sed = simplicial_edit_distance(h1, exclude_min_size=False)
    s = 4
    m = 4 + 10
    mf = 3
    assert np.allclose(sed, m / (m + s - mf))

    sed = simplicial_edit_distance(h1, exclude_min_size=False, normalize=False)
    assert np.allclose(sed, m)
