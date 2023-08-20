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
    assert np.allclose(es, 6 / 7)

    es = edit_simpliciality(h_missing_one_singleton, min_size=1, exclude_min_size=False)
    assert np.allclose(es, 6 / 7)

    # h2
    es = edit_simpliciality(h_missing_one_link)
    assert np.allclose(es, 3 / 4)

    es = edit_simpliciality(h_missing_one_link, min_size=1)
    assert np.allclose(es, 6 / 7)

    # links and triangles 2
    es = edit_simpliciality(h_links_and_triangles2)
    assert np.allclose(es, 3 / 4)

    es = edit_simpliciality(h_links_and_triangles2, min_size=1)
    assert np.allclose(es, 1 / 2)

    es = edit_simpliciality(h_links_and_triangles2, exclude_min_size=False)
    assert np.allclose(es, 3 / 4)

    # test h1
    es = edit_simpliciality(h1)
    s = 4
    m = 4 + 10
    assert np.allclose(es, s / (m + s))

    es = edit_simpliciality(h1, min_size=1)
    s = 4
    m = 4 + 10 + 7
    assert np.allclose(es, s / (m + s))

    es = edit_simpliciality(h1, exclude_min_size=False)
    s = 4
    m = 4 + 10
    assert np.allclose(es, s / (m + s))


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
    assert np.allclose(es, 6 / 7)

    es = edit_simpliciality_full_construction(
        h_missing_one_singleton, min_size=1, exclude_min_size=False
    )
    assert np.allclose(es, 6 / 7)

    # h2
    es = edit_simpliciality_full_construction(h_missing_one_link)
    assert np.allclose(es, 3 / 4)

    es = edit_simpliciality_full_construction(h_missing_one_link, min_size=1)
    assert np.allclose(es, 6 / 7)

    # links and triangles 2
    es = edit_simpliciality_full_construction(h_links_and_triangles2)
    assert np.allclose(es, 3 / 4)

    es = edit_simpliciality_full_construction(h_links_and_triangles2, min_size=1)
    assert np.allclose(es, 1 / 2)

    es = edit_simpliciality_full_construction(
        h_links_and_triangles2, exclude_min_size=False
    )
    assert np.allclose(es, 3 / 4)

    # test h1
    es = edit_simpliciality_full_construction(h1)
    s = 4
    m = 4 + 10
    assert np.allclose(es, s / (m + s))

    es = edit_simpliciality_full_construction(h1, min_size=1)
    s = 4
    m = 4 + 10 + 7
    assert np.allclose(es, s / (m + s))

    es = edit_simpliciality_full_construction(h1, exclude_min_size=False)
    s = 4
    m = 4 + 10
    assert np.allclose(es, s / (m + s))
