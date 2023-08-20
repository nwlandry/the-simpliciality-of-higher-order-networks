import logging
import random
from collections import defaultdict

import numpy as np
from xgi import Hypergraph
from xgi.exception import XGIError


def configuration_model(d, s=None, num_swaps=None):
    """Generate hypergraph configuration null model

    Parameters
    ----------
    d : dict or Hypergraph
        If a dict, this is the degree sequence, and if it is
        a Hypergraph object, it is the starting dataset.
    s : dict, optional
        The size sequence if k is the degree sequence.
        Otherwise it's ignored. By default, None.
    num_swaps : int, optional
        The number of proposed edge swaps, by default 1000

    Returns
    -------
    Hypergraph
        The reshuffled hypergraph
    """
    if isinstance(d, Hypergraph):
        H_CM = d.copy()
    elif isinstance(d, dict) and isinstance(s, dict):
        H_CM = _initialize_hypergraph(d, s)
    else:
        raise XGIError("Invalid input!")

    if num_swaps is None:
        num_swaps = 10 * H_CM.num_edges

    swaps = 0
    while swaps < num_swaps:
        k, l = random.sample(list(H_CM.edges), 2)
        i = random.choice(list(H_CM.edges.members(k)))
        j = random.choice(list(H_CM.edges.members(l)))
        try:
            H_CM.double_edge_swap(i, j, k, l)
        except:
            pass
        swaps += 1

    return H_CM


def _initialize_hypergraph(d, s):
    # A Principled, Flexible and Efficient Framework for Hypergraph Benchmarking https://arxiv.org/abs/2212.08593
    H = Hypergraph()

    nwd = defaultdict(set)
    eos = defaultdict(set)

    for v, deg in d.items():
        nwd[deg].add(v)

    for e, size in s.items():
        eos[size].add(e)

    nodes_with_degree = {deg: nwd[deg] for deg in sorted(nwd)[::-1]}
    edges_of_size = {size: eos[size] for size in sorted(eos)[::-1]}

    for size in edges_of_size:
        eos = edges_of_size[size]
        for id in eos:
            e = _extract_hyperedge(size, nodes_with_degree)
            if len(e) > 0:
                H.add_edge(e, id=id)
    return H


def _extract_hyperedge(size, nodes_with_degree):
    if size < 1:
        raise ValueError(f"Invalid size: {size}")

    nodes_chosen = dict()
    n_nodes_sampled = 0
    degrees = iter(
        sorted((deg for deg in nodes_with_degree.keys() if deg > 0), reverse=True)
    )
    while n_nodes_sampled < size:
        try:
            deg = next(degrees)
        except StopIteration:
            logging.info(
                "There aren't any more nodes available to choose."
                "Breaking the condition on either the dimension or degree sequence."
            )

            nodes_chosen[0] = set(
                np.random.choice(
                    list(nodes_with_degree[0]),
                    size=size - n_nodes_sampled,
                    replace=False,
                )
            )
            n_nodes_sampled = size
            continue

        n_nodes_to_sample = min(len(nodes_with_degree[deg]), size - n_nodes_sampled)

        nodes_chosen[deg] = set(
            np.random.choice(
                list(nodes_with_degree[deg]),
                size=n_nodes_to_sample,
                replace=False,
            )
        )
        n_nodes_sampled += n_nodes_to_sample

    for deg, node_set in nodes_chosen.items():
        if deg > 0:
            nodes_with_degree[deg] = nodes_with_degree[deg] - node_set
            nodes_with_degree[deg - 1] = (
                nodes_with_degree.get(deg - 1, set()) | node_set
            )

    nodes_chosen = set.union(*(node_set for node_set in nodes_chosen.values()))
    return nodes_chosen
