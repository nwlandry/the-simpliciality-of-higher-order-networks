import json
import os
from sys import platform

import numpy as np
import xgi
from joblib import Parallel, delayed

from sod import *


def cm_in_parallel(H, dataset_name, num_swaps, min_size):
    H_CM = configuration_model(H, num_swaps=num_swaps)

    sf = simplicial_fraction(H_CM, min_size=min_size)
    es = edit_simpliciality(H_CM, min_size=min_size)
    fes = face_edit_simpliciality(H_CM, min_size=min_size)

    print("CM completed", flush=True)
    return dataset_name, num_swaps, sf, es, fes


# args
datasets = [
    "email-enron",
    "contact-high-school",
]  # , "ndc-substances", "tags-ask-ubuntu"]
realizations = 1

if platform == "linux" or platform == "linux2":
    num_processes = len(os.sched_getaffinity(0))
elif platform == "darwin" or platform == "win32":
    num_processes = os.cpu_count()


max_order = 10
min_size = 2
num_num_swaps = 10

if not os.path.exists("Data"):
    os.mkdir("Data")
if not os.path.exists("Figures"):
    os.mkdir("Figures")


try:
    with open("Data/cm_convergence.json", "r") as file:
        data = json.loads(file.read())
except:
    arglist = []
    for dataset in datasets:
        H = xgi.load_xgi_data(dataset, max_order=max_order)
        H.cleanup(singletons=False)

        n = H.num_nodes
        m = H.num_edges
        # Initialize configuration model data

        max_log = np.log10(10 * m)

        num_swaps = np.logspace(1, max_log, num_num_swaps)

        # configuration model
        for nswaps in num_swaps:
            arglist.append((H, dataset, nswaps, min_size))
    cm_data = Parallel(n_jobs=num_processes)(
        delayed(cm_in_parallel)(*arg) for arg in arglist
    )

    data = {d: defaultdict(list) for d in datasets}
    for name, nswaps, sf, es, fes in cm_data:
        data[name]["num-swaps"].append(nswaps)
        data[name]["sf"].append(sf)
        data[name]["es"].append(es)
        data[name]["fes"].append(fes)

    with open("Data/cm_convergence.json", "w") as file:
        datastring = json.dumps(data, indent=2)
        file.write(datastring)
