import pytest
import xgi


@pytest.fixture
def sc1_with_singletons():
    return xgi.Hypergraph([{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}])


@pytest.fixture
def h_missing_one_singleton():
    return xgi.Hypergraph([{1}, {2}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}])


@pytest.fixture
def h_missing_one_link():
    return xgi.Hypergraph([{1}, {2}, {3}, {1, 3}, {2, 3}, {1, 2, 3}])


@pytest.fixture
def h_links_and_triangles():
    return xgi.Hypergraph([{1, 3}, {2, 3}, {1, 2, 3}])


@pytest.fixture
def h_links_and_triangles2():
    return xgi.Hypergraph([{1, 3}, {2, 3}, {1, 2, 3}, {1, 4}, {2, 3, 4}, {2, 4}])


@pytest.fixture
def h1():
    return xgi.Hypergraph([{1, 2, 3}, {2, 3, 4, 5}, {5, 6, 7}, {5, 6}])
