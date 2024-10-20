from sod import Trie, count_missing_subfaces, max_number_of_subfaces, powerset


def test_powerset():
    a = {1, 2, 3}

    # test default behavior
    subsets = {frozenset(s) for s in powerset(a, min_size=1)}
    assert subsets == {
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
        frozenset({1, 2, 3}),
    }

    subsets = {frozenset(s) for s in powerset(a, min_size=2)}
    assert subsets == {
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
        frozenset({1, 2, 3}),
    }

    subsets = {frozenset(s) for s in powerset(a, max_size=2)}
    assert subsets == {
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
    }


def test_count_missing_subfaces(h_missing_one_link):
    t = Trie()
    t.build_trie(h_missing_one_link.edges.members())
    assert count_missing_subfaces(t, {1}, min_size=2) == 0
    assert count_missing_subfaces(t, {2, 3}, min_size=2) == 0
    assert count_missing_subfaces(t, {2, 3}) == 0
    assert count_missing_subfaces(t, {1, 2, 3}) == 1
    assert count_missing_subfaces(t, {1, 2, 3}, min_size=2) == 1


def test_max_number_of_subfaces():
    assert max_number_of_subfaces(1, 3) == 6
    assert max_number_of_subfaces(2, 3) == 3
    assert max_number_of_subfaces(1, 4) == 14
    assert max_number_of_subfaces(2, 4) == 10
