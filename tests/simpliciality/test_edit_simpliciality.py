from sod import *


def test_edit_simpliciality(
    sc1_with_singletons,
    h_missing_one_singleton,
    h_missing_one_link,
    h_links_and_triangles2,
    h1,
):
    # simplicial complex
    es = edit_simpliciality(sc1_with_singletons)
    assert es == 1.0

    es = edit_simpliciality(sc1_with_singletons, min_size=1)
    assert es == 1.0

    es = edit_simpliciality(sc1_with_singletons, min_size=1, exclude_min_size=False)
    assert es == 1.0

    # h1
    es = edit_simpliciality(h_missing_one_singleton)
    assert es == 1.0

    es = edit_simpliciality(h_missing_one_singleton, min_size=1)
    assert np.allclose(es, 5 / 6)

    es = edit_simpliciality(
        h_missing_one_singleton, min_size=1, exclude_min_size=False
    )
    assert np.allclose(es, 5 / 6)

    # h2
    es = edit_simpliciality(h_missing_one_link)
    assert np.allclose(es, 2 / 3)

    es = edit_simpliciality(h_missing_one_link, min_size=1)
    assert np.allclose(es, 5 / 6)

    # links and triangles 2
    es = edit_simpliciality(h_links_and_triangles2)
    assert np.allclose(es, 2 / 3)

    es = edit_simpliciality(h_links_and_triangles2, min_size=1)
    assert np.allclose(es, 1 / 3)

    es = edit_simpliciality(h_links_and_triangles2, exclude_min_size=False)
    assert np.allclose(es, 3 / 5)

    # test h1
    es = edit_simpliciality(h1)
    s = 4
    m = 4 + 10
    mf = 3
    assert np.allclose(es, (s - mf) / (m + s - mf))

    es = edit_simpliciality(h1, min_size=1)
    s = 4
    m = 4 + 10 + 7
    mf = 3
    assert np.allclose(es, (s - mf) / (m + s - mf))

    es = edit_simpliciality(h1, exclude_min_size=False)
    s = 4
    m = 4 + 10
    mf = 3
    assert np.allclose(es, (s - mf) / (m + s - mf))


def test_edit_simpliciality_full_construction(
    sc1_with_singletons,
    h_missing_one_singleton,
    h_missing_one_link,
    h_links_and_triangles2,
    h1,
):
    # simplicial complex
    es = edit_simpliciality_full_construction(sc1_with_singletons)
    assert es == 1.0

    es = edit_simpliciality_full_construction(sc1_with_singletons, min_size=1)
    assert es == 1.0

    es = edit_simpliciality_full_construction(
        sc1_with_singletons, min_size=1, exclude_min_size=False
    )
    assert es == 1.0

    # h1
    es = edit_simpliciality_full_construction(h_missing_one_singleton)
    assert es == 1.0

    es = edit_simpliciality_full_construction(h_missing_one_singleton, min_size=1)
    assert np.allclose(es, 5 / 6)

    es = edit_simpliciality_full_construction(
        h_missing_one_singleton, min_size=1, exclude_min_size=False
    )
    assert np.allclose(es, 5 / 6)

    # h2
    es = edit_simpliciality_full_construction(h_missing_one_link)
    assert np.allclose(es, 2 / 3)

    es = edit_simpliciality_full_construction(h_missing_one_link, min_size=1)
    assert np.allclose(es, 5 / 6)

    # links and triangles 2
    es = edit_simpliciality_full_construction(h_links_and_triangles2)
    assert np.allclose(es, 2 / 3)

    es = edit_simpliciality_full_construction(h_links_and_triangles2, min_size=1)
    assert np.allclose(es, 1 / 3)

    es = edit_simpliciality_full_construction(
        h_links_and_triangles2, exclude_min_size=False
    )
    assert np.allclose(es, 3/ 5)

    # test h1
    es = edit_simpliciality_full_construction(h1)
    s = 4
    m = 4 + 10
    mf = 3
    assert np.allclose(es, (s - mf) / (s - mf + m))

    es = edit_simpliciality_full_construction(h1, min_size=1)
    s = 4
    m = 4 + 10 + 7
    mf = 3
    assert np.allclose(es, (s - mf) / (s - mf + m))

    es = edit_simpliciality_full_construction(h1, exclude_min_size=False)
    s = 4
    m = 4 + 10
    mf = 3
    assert np.allclose(es, (s - mf) / (s - mf + m))
