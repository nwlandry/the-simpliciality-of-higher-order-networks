import numpy as np
import xgi
from xgi import nodestat_func

from .edit_simpliciality import edit_simpliciality
from .face_edit_simpliciality import face_edit_simpliciality
from .simplicial_fraction import simplicial_fraction


@nodestat_func
def local_simplicial_fraction(net, bunch, min_size=2, exclude_min_size=True):
    s = dict()
    for n in bunch:
        nbrs = net.nodes.neighbors(n)
        if len(nbrs) == 0:
            s[n] = np.nan
        else:
            nbrs.add(n)
            sh = xgi.subhypergraph(net, nodes=nbrs)
            s[n] = simplicial_fraction(sh, min_size, exclude_min_size)
    return s


@nodestat_func
def local_edit_simpliciality(net, bunch, min_size=2, exclude_min_size=True):
    s = dict()
    for n in bunch:
        nbrs = net.nodes.neighbors(n)
        if len(nbrs) == 0:
            s[n] = np.nan
        else:
            nbrs.add(n)
            sh = xgi.subhypergraph(net, nodes=nbrs)
            s[n] = edit_simpliciality(sh, min_size, exclude_min_size)
    return s


@nodestat_func
def local_face_edit_simpliciality(net, bunch, min_size=2, exclude_min_size=True):
    s = dict()
    for n in bunch:
        nbrs = net.nodes.neighbors(n)
        if len(nbrs) == 0:
            s[n] = np.nan
        else:
            nbrs.add(n)
            sh = xgi.subhypergraph(net, nodes=nbrs)
            s[n] = face_edit_simpliciality(sh, min_size, exclude_min_size)
    return s
