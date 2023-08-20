import json
import multiprocessing as mp
import os
import sys

import numpy as np
import xgi

from sod import *


def cl_in_parallel(k, s, min_size):
    H_CL = xgi.chung_lu_hypergraph(k, s)

    sf = simplicial_fraction(H_CL, min_size=min_size)
    es = edit_simpliciality(H_CL, min_size=min_size)
    fes = face_edit_simpliciality(H_CL, min_size=min_size)

    print("CL completed", flush=True)
    return sf, es, fes


def cm_in_parallel(H, min_size):
    H_CM = configuration_model(H)

    sf = simplicial_fraction(H_CM, min_size=min_size)
    es = edit_simpliciality(H_CM, min_size=min_size)
    fes = face_edit_simpliciality(H_CM, min_size=min_size)

    print("CM completed", flush=True)
    return sf, es, fes


def dcsbm_in_parallel(d, s, g1, g2, omega, min_size):
    H_DCSBM = xgi.dcsbm_hypergraph(d, s, g1, g2, omega)

    sf = simplicial_fraction(H_DCSBM, min_size=min_size)
    es = edit_simpliciality(H_DCSBM, min_size=min_size)
    fes = face_edit_simpliciality(H_DCSBM, min_size=min_size)

    print("DCSBM completed", flush=True)
    return sf, es, fes


# args
dataset = sys.argv[1]
realizations = int(sys.argv[2])
num_processes = len(os.sched_getaffinity(0))

max_order = 10
min_size = 2


if not os.path.exists("Data"):
    os.mkdir("Data")
if not os.path.exists("Figures"):
    os.mkdir("Figures")


try:
    with open(f"Data/model_simpliciality_{dataset}.json", "r") as file:
        data = json.loads(file.read())
except:
    H = xgi.load_xgi_data(dataset, max_order=max_order)
    H.cleanup(singletons=False)

    k = H.nodes.degree.asdict()
    s = H.edges.size.asdict()
    n = H.num_nodes
    m = H.num_edges

    data = dict()

    # Initialize configuration model data
    data["CM"] = dict()

    # Initialize Chung-Lu model data
    data["CL"] = dict()

    # Initialize DCSBM data
    data["DCSBM"] = dict()

    arglist = []

    # configuration model
    for i in range(realizations):
        arglist.append((H, min_size))

    with mp.Pool(processes=num_processes) as pool:
        cm_data = pool.starmap(cm_in_parallel, arglist)

    data["CM"]["sf"] = [d[0] for d in cm_data]
    data["CM"]["es"] = [d[1] for d in cm_data]
    data["CM"]["fes"] = [d[2] for d in cm_data]

    arglist = []
    # chung-lu model
    for i in range(realizations):
        arglist.append((k, s, min_size))

    with mp.Pool(processes=num_processes) as pool:
        cl_data = pool.starmap(cl_in_parallel, arglist)

    data["CL"]["sf"] = [d[0] for d in cl_data]
    data["CL"]["es"] = [d[1] for d in cl_data]
    data["CL"]["fes"] = [d[2] for d in cl_data]

    # DCSBM
    with open(f"Data/DCSBM_parameters_{dataset}.json", "r") as file:
        j = json.loads(file.read())

    # convert everything to int just in case
    d = {int(i): int(deg) for i, deg in j["d"].items()}
    s = {int(i): int(size) for i, size in j["s"].items()}
    g1 = {int(i): int(g) for i, g in j["g1"].items()}
    g2 = {int(i): int(g) for i, g in j["g2"].items()}
    omega = np.array(j["omega"])

    arglist = []
    for i in range(realizations):
        arglist.append((d, s, g1, g2, omega, min_size))

    with mp.Pool(processes=num_processes) as pool:
        dcsbm_data = pool.starmap(dcsbm_in_parallel, arglist)

    data["DCSBM"]["sf"] = [d[0] for d in dcsbm_data]
    data["DCSBM"]["es"] = [d[1] for d in dcsbm_data]
    data["DCSBM"]["fes"] = [d[2] for d in dcsbm_data]

    with open(f"Data/model_simpliciality_{dataset}.json", "w") as file:
        datastring = json.dumps(data, indent=2)
        file.write(datastring)
