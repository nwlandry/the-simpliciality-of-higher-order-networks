from sod import *


def test_is_simplex(sc1_with_singletons, h_missing_one_singleton):
    t = Trie()
    edges = sc1_with_singletons.edges.members()
    t.build_trie(edges)

    assert is_simplex(t, {1, 2, 3}, min_size=2)
    assert is_simplex(t, {1, 2, 3}, min_size=1)
    assert is_simplex(t, {1, 2}, min_size=1)

    t = Trie()
    edges = h_missing_one_singleton.edges.members()
    t.build_trie(edges)

    assert is_simplex(t, {1, 2, 3})
    assert not is_simplex(t, {1, 2, 3}, min_size=1)
    assert not is_simplex(t, {2, 3}, min_size=1)
    assert is_simplex(t, {1, 2}, min_size=1)


def test_count_simplices(sc1_with_singletons, h_missing_one_singleton):
    ns = count_simplices(sc1_with_singletons)
    assert ns == 1

    ns = count_simplices(sc1_with_singletons, min_size=1)
    assert ns == 4

    ns = count_simplices(sc1_with_singletons, min_size=1, exclude_min_size=False)
    assert ns == 7

    ns = count_simplices(h_missing_one_singleton, min_size=1)
    assert ns == 1


def test_potential_simplices(sc1_with_singletons, h_missing_one_link):
    ps = potential_simplices(sc1_with_singletons)
    assert ps == 1

    ps = potential_simplices(sc1_with_singletons, min_size=1)
    assert ps == 4

    ps = potential_simplices(sc1_with_singletons, min_size=1, exclude_min_size=False)
    assert ps == 7

    ps = potential_simplices(h_missing_one_link, min_size=1)
    assert ps == 3


def test_simplicial_fraction(
    sc1_with_singletons, h_missing_one_singleton, h_missing_one_link
):
    # simplicial complex
    sf = simplicial_fraction(sc1_with_singletons)
    assert sf == 1.0

    sf = simplicial_fraction(sc1_with_singletons, min_size=1)
    assert sf == 1.0

    sf = simplicial_fraction(sc1_with_singletons, min_size=1, exclude_min_size=False)
    assert sf == 1.0

    # h1
    sf = simplicial_fraction(h_missing_one_singleton)
    assert sf == 1.0

    sf = simplicial_fraction(h_missing_one_singleton, min_size=1)
    assert sf == 1 / 4

    sf = simplicial_fraction(
        h_missing_one_singleton, min_size=1, exclude_min_size=False
    )
    assert sf == 0.5

    # h2
    sf = simplicial_fraction(h_missing_one_link)
    assert sf == 0

    sf = simplicial_fraction(h_missing_one_link, min_size=1)
    assert sf == 2 / 3
