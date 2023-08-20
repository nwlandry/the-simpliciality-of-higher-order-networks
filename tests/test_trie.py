from sod import Trie


def test_trie(h_links_and_triangles2):
    t = Trie()
    edges = h_links_and_triangles2.edges.members()
    t.build_trie(edges)

    assert t.search({1, 3})
    assert t.search({2, 3})
    assert t.search({1, 2, 3})
    assert t.search({1, 4})
    assert t.search({2, 3, 4})
    assert t.search({2, 4})
    assert not t.search({1})
    assert t.search((1, 3))
    assert t.search((3, 1))
    assert t.search((3, 2, 1))
