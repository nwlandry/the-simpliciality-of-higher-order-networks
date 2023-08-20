from sod import *


def test_face_edit_simpliciality(
    sc1_with_singletons,
    h_missing_one_singleton,
    h_missing_one_link,
    h_links_and_triangles2,
):
    # simplicial complex
    fes = face_edit_simpliciality(sc1_with_singletons)
    assert fes == 1.0

    fes = face_edit_simpliciality(sc1_with_singletons, min_size=1)
    assert fes == 1.0

    fes = face_edit_simpliciality(
        sc1_with_singletons, min_size=1, exclude_min_size=False
    )
    assert fes == 1.0

    # h1
    fes = face_edit_simpliciality(h_missing_one_singleton)
    assert fes == 1.0

    fes = face_edit_simpliciality(h_missing_one_singleton, min_size=1)
    assert np.allclose(fes, 5 / 6)

    fes = face_edit_simpliciality(
        h_missing_one_singleton, min_size=1, exclude_min_size=False
    )
    assert np.allclose(fes, 5 / 6)

    # h2
    fes = face_edit_simpliciality(h_missing_one_link)
    assert np.allclose(fes, 2 / 3)

    fes = face_edit_simpliciality(h_missing_one_link, min_size=1)
    assert np.allclose(fes, 5 / 6)

    # links and triangles 2
    fes = face_edit_simpliciality(h_links_and_triangles2)
    assert np.allclose(fes, 2 / 3)

    fes = face_edit_simpliciality(h_links_and_triangles2, min_size=1)
    assert np.allclose(fes, 2 / 9)

    fes = face_edit_simpliciality(h_links_and_triangles2, exclude_min_size=False)
    assert np.allclose(fes, 7 / 9)
