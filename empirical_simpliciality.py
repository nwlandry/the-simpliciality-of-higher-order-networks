import json
import os

import xgi

from sod import *

if not os.path.exists("Data"):
    os.mkdir("Data")
if not os.path.exists("Figures"):
    os.mkdir("Figures")

datasets = [
    "contact-primary-school",
    "contact-high-school",
    "hospital-lyon",
    "email-enron",
    "email-eu",
    "ndc-substances",
    "diseasome",
    "disgenenet",
    "congress-bills",
    "tags-ask-ubuntu",
]

min_size = 2

try:
    with open(f"Data/empirical_simpliciality.json", "r") as file:
        data = json.loads(file.read())
except:
    data = dict()

    # This argument is very important to the computational feasibility of this algorithm
    max_order = 10

    for d in datasets:
        data[d] = dict()
        H = xgi.load_xgi_data(d, max_order=max_order)
        H.cleanup(singletons=True)

        data[d]["es"] = edit_simpliciality(H, min_size=min_size)
        data[d]["fes"] = face_edit_simpliciality(H, min_size=min_size)
        data[d]["sf"] = simplicial_fraction(H, min_size=min_size)

        print("Just finished ", d, flush=True)

    with open(f"Data/empirical_simpliciality.json", "w") as file:
        datastring = json.dumps(data, indent=2)
        file.write(datastring)
