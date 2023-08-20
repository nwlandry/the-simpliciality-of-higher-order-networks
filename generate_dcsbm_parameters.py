import json
import os
import time

import biSBM as bm
import numpy as np
import xgi


def to_bipartite_edgelist(H):
    data = []
    n = H.num_nodes
    for id1, members in H._node.items():
        for id2 in members:
            data.append([id1, id2 + n])
    return np.array(data, dtype=np.int_)


max_order = 10
dataset = "coauthor-mag-history"

H = xgi.load_xgi_data(dataset, max_order=max_order)
H.cleanup()
edgelist = to_bipartite_edgelist(H)


mcmc = bm.engines.MCMC(
    f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",
    n_sweeps=1,
    is_parallel=False,
    n_cores=1,
)
types = mcmc.gen_types(H.num_nodes, H.num_edges)

sampler = bm.OptimalKs(mcmc, edgelist, types, default_args=True, random_init_k=False)

start = time.time()
sampler.minimize_bisbm_dl()
print(time.time() - start)

labels = sampler.bm_state["mb"]

n_a = sampler.bm_state["n_a"]
n_b = sampler.bm_state["n_b"]

k_a = sampler.bm_state["ka"]
k_b = sampler.bm_state["kb"]

omega = sampler.bm_state["e_rs"][:k_a, k_a:]

k = H.nodes.degree.asdict()
s = H.edges.size.asdict()

g1 = {i: int(labels[i]) for i in range(n_a)}
g2 = {i: int(labels[i + n_a] - k_a) for i in range(n_b)}

data = dict()
data["omega"] = omega.tolist()
data["d"] = k
data["s"] = s
data["g1"] = g1
data["g2"] = g2

with open(f"Data/DCSBM_parameters_{dataset}.json", "w") as file:
    datastring = json.dumps(data, indent=1)
    file.write(datastring)
