"""Draw hypergraphs and simplicial complexes with matplotlib."""

from collections import defaultdict
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xgi
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from numpy import ndarray
from xgi.exception import XGIError
from xgi.stats import IDStat


def _scalar_arg_to_dict(scalar_arg, ids, min_val, max_val):
    """Map different types of arguments for drawing style to a dict with scalar values.

    Parameters
    ----------
    scalar_arg : int, float, dict, iterable, or NodeStat/EdgeStat
        Attributes for drawing parameter.
    ids : NodeView or EdgeView
        This is the node or edge IDs that attributes get mapped to.
    min_val : int or float
        The minimum value of the drawing parameter.
    max_val : int or float
        The maximum value of the drawing parameter.

    Returns
    -------
    dict
        An ID: scalar dictionary.

    Raises
    ------
    TypeError
        If a int, float, list, dict, or NodeStat/EdgeStat is not passed.
    """
    if isinstance(scalar_arg, str):
        raise TypeError(
            "Argument must be int, float, dict, iterable, "
            f"or NodeStat/EdgeStat. Received {type(scalar_arg)}"
        )

    # Single argument
    if isinstance(scalar_arg, (int, float)):
        return {id: scalar_arg for id in ids}

    # IDStat
    if isinstance(scalar_arg, IDStat):
        vals = np.interp(
            scalar_arg.asnumpy(),
            [scalar_arg.min(), scalar_arg.max()],
            [min_val, max_val],
        )
        return dict(zip(ids, vals))

    # Iterables of floats or ints
    if isinstance(scalar_arg, Iterable):
        if isinstance(scalar_arg, dict):
            try:
                return {id: float(scalar_arg[id]) for id in scalar_arg if id in ids}
            except ValueError as e:
                raise TypeError(
                    "The input dict must have values that can be cast to floats."
                )

        elif isinstance(scalar_arg, (list, ndarray)):
            try:
                return {id: float(scalar_arg[idx]) for idx, id in enumerate(ids)}
            except ValueError as e:
                raise TypeError(
                    "The input list or array must have values that can be cast to floats."
                )
        else:
            raise TypeError(
                "Argument must be an dict, list, or numpy array of floats or ints."
            )

    raise TypeError(
        "Argument must be int, float, dict, iterable, "
        f"or NodeStat/EdgeStat. Received {type(scalar_arg)}"
    )


def _color_arg_to_dict(color_arg, ids, cmap):
    """Map different types of arguments for drawing style to a dict with color values.

    Parameters
    ----------
    color_arg : Several formats are accepted:

        Single color values

        * str
        * 3- or 4-tuple

        Iterable of colors (each color specified as above)

        * numpy array
        * list
        * dict {id: color} pairs

        Iterable of numerical values (floats or ints)

        * list
        * dict
        * numpy array

        Stats

        * NodeStat
        * EdgeStat

        Attributes for drawing parameter.
    ids : NodeView or EdgeView
        This is the node or edge IDs that attributes get mapped to.
    cmap : ListedColormap or LinearSegmentedColormap
        colormap to use for NodeStat/EdgeStat.

    Returns
    -------
    dict
        An ID: color dictionary.

    Raises
    ------
    TypeError
        If a string, dict, iterable, or NodeStat/EdgeStat is not passed.

    Notes
    -----
    For the iterable of values, we do not accept tuples,
    because there is the potential for ambiguity.
    """

    # single argument. Must be a string or a tuple of floats
    if isinstance(color_arg, str) or (
        isinstance(color_arg, tuple) and isinstance(color_arg[0], float)
    ):
        return {id: color_arg for id in ids}

    # Iterables of colors. The values of these iterables must strings or tuples. As of now,
    # there is not a check to verify that the tuples contain floats.
    if isinstance(color_arg, Iterable):
        if isinstance(color_arg, dict) and isinstance(
            next(iter(color_arg.values())), (str, tuple, ndarray)
        ):
            return {id: color_arg[id] for id in color_arg if id in ids}
        if isinstance(color_arg, (list, ndarray)) and isinstance(
            color_arg[0], (str, tuple, ndarray)
        ):
            return {id: color_arg[idx] for idx, id in enumerate(ids)}

    # Stats or iterable of values
    if isinstance(color_arg, (Iterable, IDStat)):
        # set max and min of interpolation based on color map
        if isinstance(cmap, ListedColormap):
            minval = 0
            maxval = cmap.N
        elif isinstance(cmap, LinearSegmentedColormap):
            minval = 0.2
            maxval = 0.8
        else:
            raise XGIError("Invalid colormap!")

        # handle the case of IDStat vs iterables
        if isinstance(color_arg, IDStat):
            vals = np.interp(
                color_arg.asnumpy(),
                [color_arg.min(), color_arg.max()],
                [minval, maxval],
            )
            return {
                id: np.array(cmap(vals[i])).reshape(1, -1) for i, id in enumerate(ids)
            }

        elif isinstance(color_arg, Iterable):
            if isinstance(color_arg, dict) and isinstance(
                next(iter(color_arg.values())), (int, float)
            ):
                v = list(color_arg.values())
                vals = np.interp(v, [np.min(v), np.max(v)], [minval, maxval])
                # because we have ids, we can't just assume that the keys of arg correspond to
                # the ids.
                return {
                    id: np.array(cmap(v)).reshape(1, -1)
                    for v, id in zip(vals, color_arg.keys())
                    if id in ids
                }

            if isinstance(color_arg, (list, ndarray)) and isinstance(
                color_arg[0], (int, float)
            ):
                vals = np.interp(
                    color_arg, [np.min(color_arg), np.max(color_arg)], [minval, maxval]
                )
                return {
                    id: np.array(cmap(vals[i])).reshape(1, -1)
                    for i, id in enumerate(ids)
                }
            else:
                raise TypeError(
                    "Argument must be an dict, list, or numpy array of floats."
                )

    raise TypeError(
        "Argument must be str, dict, iterable, or "
        f"NodeStat/EdgeStat. Received {type(color_arg)}"
    )


def _CCW_sort(p):
    """
    Sort the input 2D points counterclockwise.
    """
    p = np.array(p)
    mean = np.mean(p, axis=0)
    d = p - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def draw_node_labels(
    H,
    pos,
    node_labels=False,
    font_size_nodes=10,
    font_color_nodes="black",
    font_family_nodes="sans-serif",
    font_weight_nodes="normal",
    alpha_nodes=None,
    bbox_nodes=None,
    horizontalalignment_nodes="center",
    verticalalignment_nodes="center",
    ax_nodes=None,
    clip_on_nodes=True,
):
    """Draw node labels on the hypergraph or simplicial complex.

    Parameters
    ----------
    H : Hypergraph or SimplicialComplex.
    pos : dict
        Dictionary of positions node_id:(x,y).
    node_labels : bool or dict, optional
        If True, draw ids on the nodes. If a dict, must contain (node_id: label) pairs.
        By default, False.
    font_size_nodes : int, optional
        Font size for text labels, by default 10.
    font_color_nodes : str, optional
        Font color string, by default "black".
    font_family_nodes : str, optional
        Font family, by default "sans-serif".
    font_weight_nodes : str (default='normal')
        Font weight.
    alpha_nodes : float, optional
        The text transparency, by default None.
    bbox_nodes : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for node labels.
        When it is None (default), use Matplotlib's ax.text default
    horizontalalignment_nodes : str, optional
        Horizontal alignment {'center', 'right', 'left'}.
        By default, "center".
    verticalalignment_nodes : str, optional
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}.
        By default, "center".
    ax_nodes : matplotlib.pyplot.axes, optional
        Draw the graph in the specified Matplotlib axes.
        By default, None.
    clip_on_nodes : bool, optional
        Turn on clipping of node labels at axis boundaries.
        By default, True.

    Returns
    -------
    dict
        `dict` of labels keyed by node id.

    See Also
    --------
    draw
    draw_nodes
    draw_hyperedges
    draw_simplices
    draw_hyperedge_labels
    """
    if ax_nodes is None:
        ax = plt.gca()
    else:
        ax = ax_nodes

    if node_labels is True:
        node_labels = {id: id for id in H.nodes}

    # Plot the labels in the last layer
    zorder = xgi.max_edge_order(H) + 1

    text_items = {}
    for idx, label in node_labels.items():
        (x, y) = pos[idx]

        if not isinstance(label, str):
            label = str(label)

        t = ax.text(
            x,
            y,
            label,
            size=font_size_nodes,
            color=font_color_nodes,
            family=font_family_nodes,
            weight=font_weight_nodes,
            alpha=alpha_nodes,
            horizontalalignment=horizontalalignment_nodes,
            verticalalignment=verticalalignment_nodes,
            transform=ax.transData,
            bbox=bbox_nodes,
            clip_on=clip_on_nodes,
            zorder=zorder,
        )
        text_items[idx] = t

    return text_items


def draw_hyperedge_labels(
    H,
    pos,
    hyperedge_labels=False,
    font_size_edges=10,
    font_color_edges="black",
    font_family_edges="sans-serif",
    font_weight_edges="normal",
    alpha_edges=None,
    bbox_edges=None,
    horizontalalignment_edges="center",
    verticalalignment_edges="center",
    ax_edges=None,
    rotate_edges=False,
    clip_on_edges=True,
):
    """Draw hyperedge labels on the hypegraph or simplicial complex.

    Parameters
    ----------
    H : Hypergraph.
    pos : dict
        Dictionary of positions node_id:(x,y).
    hyperedge_labels : bool or dict, optional
        If True, draw ids on the hyperedges. If a dict, must contain (edge_id: label)
        pairs.  By default, False.
    font_size_edges : int, optional
        Font size for text labels, by default 10.
    font_color_edges : str, optional
        Font color string, by default "black".
    font_family_edges : str (default='sans-serif')
        Font family.
    font_weight_edges : str (default='normal')
        Font weight.
    alpha_edges : float, optional
        The text transparency, by default None.
    bbox_edges : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        By default, {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}
    horizontalalignment_edges : str, optional
        Horizontal alignment {'center', 'right', 'left'}.
        By default, "center".
    verticalalignment_edges: str, optional
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}.
        By default, "center".
    ax_edges : matplotlib.pyplot.axes, optional
        Draw the graph in the specified Matplotlib axes. By default, None.
    rotate_edges : bool, optional
        Rotate edge labels for dyadic links to lie parallel to edges, by default False.
    clip_on_edges: bool, optional
        Turn on clipping of hyperedge labels at axis boundaries, by default True.

    Returns
    -------
    dict
        `dict` of labels keyed by hyperedge id.

    See Also
    --------
    draw
    draw_nodes
    draw_hyperedges
    draw_simplices
    draw_node_labels

    """
    if ax_edges is None:
        ax = plt.gca()
    else:
        ax = ax_edges

    if hyperedge_labels is True:
        hyperedge_labels = {id: id for id in H.edges}

    text_items = {}
    for id, label in hyperedge_labels.items():
        he = H.edges.members(id)
        coordinates = [[pos[n][0], pos[n][1]] for n in he]
        x, y = np.mean(coordinates, axis=0)

        if len(he) == 2:
            # Rotate edge labels for dyadic links to lie parallel to edges
            if rotate_edges:
                x_diff, y_diff = np.subtract(coordinates[1], coordinates[0])
                angle = np.arctan2(y_diff, x_diff) / (2.0 * np.pi) * 360
                # Make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                # Transform data coordinate angle to screen coordinate angle
                xy = np.array((x, y))
                trans_angle = ax.transData.transform_angles(
                    np.array((angle,)), xy.reshape((1, 2))
                )[0]
            else:
                trans_angle = 0.0
        else:
            trans_angle = 0.0

        # Use default box of white with white border
        if bbox_edges is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        else:
            bbox = bbox_edges

        if not isinstance(label, str):
            label = str(label)

        t = ax.text(
            x,
            y,
            label,
            size=font_size_edges,
            color=font_color_edges,
            family=font_family_edges,
            weight=font_weight_edges,
            alpha=alpha_edges,
            horizontalalignment=horizontalalignment_edges,
            verticalalignment=verticalalignment_edges,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            clip_on=clip_on_edges,
        )
        text_items[id] = t

    return text_items


def draw_multilayer(
    H,
    pos=None,
    ax=None,
    dyad_color="black",
    dyad_lw=0.5,
    edge_fc=None,
    node_fc="white",
    node_ec="black",
    node_lw=0.5,
    node_size=5,
    max_order=None,
    conn_lines=True,
    conn_lines_style="dotted",
    width=5,
    height=5,
    h_angle=10,
    v_angle=20,
    sep=1,
    **kwargs,
):
    """Draw a hypergraph or simplicial complex visualized in 3D
    showing hyperedges/simplices of different orders on superimposed layers.

    Parameters
    ----------
    H : Hypergraph or SimplicialComplex.
        Higher-order network to plot.
    pos : dict or None, optional
        The positions of the nodes in the multilayer network. If None, a default layout will be computed using xgi.barycenter_spring_layout(). Default is None.
    ax : matplotlib Axes3DSubplot or None, optional
        The subplot to draw the visualization on. If None, a new subplot will be created. Default is None.
    dyad_color : str, dict, iterable, or EdgeStat, optional
        Color of the dyadic links.  If str, use the same color for all edges. If a dict,
        must contain (edge_id: color_str) pairs.  If iterable, assume the colors are
        specified in the same order as the edges are found in H.edges. If EdgeStat, use
        a colormap (specified with dyad_color_cmap) associated to it. By default,
        "black".
    dyad_lw : int, float, dict, iterable, or EdgeStat, optional
        Line width of edges of order 1 (dyadic links).  If int or float, use the same
        width for all edges.  If a dict, must contain (edge_id: width) pairs.  If
        iterable, assume the widths are specified in the same order as the edges are
        found in H.edges. If EdgeStat, use a monotonic linear interpolation defined
        between min_dyad_lw and max_dyad_lw. By default, 1.5.
    edge_fc : str, dict, iterable, or EdgeStat, optional
        Color of the hyperedges.  If str, use the same color for all nodes.  If a dict,
        must contain (edge_id: color_str) pairs.  If other iterable, assume the colors
        are specified in the same order as the hyperedges are found in H.edges. If
        EdgeStat, use the colormap specified with edge_fc_cmap. If None (default), use
        the H.edges.size.
    node_fc : str, dict, iterable, or NodeStat, optional
        Color of the nodes.  If str, use the same color for all nodes.  If a dict, must
        contain (node_id: color_str) pairs.  If other iterable, assume the colors are
        specified in the same order as the nodes are found in H.nodes. If NodeStat, use
        the colormap specified with node_fc_cmap. By default, "white".
    node_ec : str, dict, iterable, or NodeStat, optional
        Color of node borders.  If str, use the same color for all nodes.  If a dict,
        must contain (node_id: color_str) pairs.  If other iterable, assume the colors
        are specified in the same order as the nodes are found in H.nodes. If NodeStat,
        use the colormap specified with node_ec_cmap. By default, "black".
    node_lw : int, float, dict, iterable, or NodeStat, optional
        Line width of the node borders in pixels.  If int or float, use the same width
        for all node borders.  If a dict, must contain (node_id: width) pairs.  If
        iterable, assume the widths are specified in the same order as the nodes are
        found in H.nodes. If NodeStat, use a monotonic linear interpolation defined
        between min_node_lw and max_node_lw. By default, 1.
    node_size : int, float, dict, iterable, or NodeStat, optional
        Radius of the nodes in pixels.  If int or float, use the same radius for all
        nodes.  If a dict, must contain (node_id: radius) pairs.  If iterable, assume
        the radiuses are specified in the same order as the nodes are found in
        H.nodes. If NodeStat, use a monotonic linear interpolation defined between
        min_node_size and max_node_size. By default, 15.
    max_order : int, optional
        Maximum of hyperedges to plot. If None (default), plots all orders.
    conn_lines : bool, optional
        Whether to draw connections between layers. Default is True.
    conn_lines_style : str, optional
        The linestyle of the connections between layers. Default is 'dotted'.
    width : float, optional
        The width of the figure in inches. Default is 5.
    height : float, optional
        The height of the figure in inches. Default is 5.
    h_angle : float, optional
        The rotation angle around the horizontal axis in degrees. Default is 10.
    v_angle : float, optional
        The rotation angle around the vertical axis in degrees. Default is 0.
    sep : float, optional
        The separation between layers. Default is 1.
    **kwargs : optional args
        Alternate default values. Values that can be overwritten are the following:
        * min_node_size
        * max_node_size
        * min_node_lw
        * max_node_lw
        * min_dyad_lw
        * max_dyad_lw
        * node_fc_cmap
        * node_ec_cmap
        * dyad_color_cmap
        * edge_fc_cmap

    Returns
    -------
    ax : matplotlib Axes3DSubplot
        The subplot with the multilayer network visualization.
    """
    settings = {
        "min_node_size": 10.0,
        "max_node_size": 30.0,
        "min_dyad_lw": 2.0,
        "max_dyad_lw": 10.0,
        "min_node_lw": 1.0,
        "max_node_lw": 5.0,
        "node_fc_cmap": cm.Reds,
        "node_ec_cmap": cm.Greys,
        "edge_fc_cmap": cm.Blues,
        "dyad_color_cmap": cm.Greys,
    }

    settings.update(kwargs)

    if edge_fc is None:
        edge_fc = H.edges.size

    if pos is None:
        pos = xgi.barycenter_spring_layout(H)

    if ax is None:
        _, ax = plt.subplots(
            1, 1, figsize=(width, height), dpi=600, subplot_kw={"projection": "3d"}
        )

    s = xgi.unique_edge_sizes(H)
    if max_order is None:
        max_order = max(s) - 1
    else:
        max_order = min(max_order, max(s) - 1)
    min_order = min(s) - 1

    xs, ys = zip(*pos.values())

    dyad_color = _color_arg_to_dict(dyad_color, H.edges, settings["dyad_color_cmap"])
    dyad_lw = _scalar_arg_to_dict(
        dyad_lw, H.edges, settings["min_dyad_lw"], settings["max_dyad_lw"]
    )

    edge_fc = _color_arg_to_dict(edge_fc, H.edges, settings["edge_fc_cmap"])

    node_fc = _color_arg_to_dict(node_fc, H.nodes, settings["node_fc_cmap"])
    node_ec = _color_arg_to_dict(node_ec, H.nodes, settings["node_ec_cmap"])
    node_lw = _scalar_arg_to_dict(
        node_lw,
        H.nodes,
        settings["min_node_lw"],
        settings["max_node_lw"],
    )
    node_size = _scalar_arg_to_dict(
        node_size, H.nodes, settings["min_node_size"], settings["max_node_size"]
    )

    for id, he in H.edges.members(dtype=dict).items():
        d = len(he) - 1
        zs = d * sep

        # dyads
        if d > max_order:
            continue

        if d == 1:
            he = list(he)
            x1 = [pos[he[0]][0], pos[he[0]][1], zs]
            x2 = [pos[he[1]][0], pos[he[1]][1], zs]
            l = Line3DCollection(
                [(x1, x2)],
                color=dyad_color[id],
                linewidth=dyad_lw[id],
            )
            ax.add_collection3d(l)
        # higher-orders
        else:
            poly = []
            vertices = np.array([[pos[i][0], pos[i][1], zs] for i in he])
            vertices = _CCW_sort(vertices)
            poly.append(vertices)
            poly = Poly3DCollection(
                poly,
                zorder=d - 1,
                color=edge_fc[id],
                alpha=0.5,
                edgecolor=None,
            )
            ax.add_collection3d(poly)

    # now draw by order
    # draw lines connecting points on the different planes
    if conn_lines:
        lines3d_between = [
            (list(pos[i]) + [min_order * sep], list(pos[i]) + [max_order * sep])
            for i in H.nodes
        ]
        between_lines = Line3DCollection(
            lines3d_between,
            zorder=d,
            color=".5",
            alpha=0.4,
            linestyle=conn_lines_style,
            linewidth=1,
        )
        ax.add_collection3d(between_lines)

    (x, y, s, c, ec, lw,) = zip(
        *[
            (
                pos[i][0],
                pos[i][1],
                node_size[i] ** 2,
                node_fc[i],
                node_ec[i],
                node_lw[i],
            )
            for i in H.nodes
        ]
    )
    for d in range(min_order, max_order + 1):
        # draw nodes
        z = [sep * d] * H.num_nodes
        ax.scatter(
            x,
            y,
            z,
            s=s,
            c=c,
            edgecolors=ec,
            linewidths=lw,
            zorder=max_order + 1,
            alpha=1,
        )

        # draw surfaces corresponding to the different orders
        xdiff = np.max(xs) - np.min(xs)
        ydiff = np.max(ys) - np.min(ys)
        ymin = np.min(ys) - ydiff * 0.1
        ymax = np.max(ys) + ydiff * 0.1
        xmin = np.min(xs) - xdiff * 0.1 * (width / height)
        xmax = np.max(xs) + xdiff * 0.1 * (width / height)
        xx, yy = np.meshgrid([xmin, xmax], [ymin, ymax])
        zz = np.zeros(xx.shape) + d * sep
        ax.plot_surface(
            xx,
            yy,
            zz,
            color="grey",
            alpha=0.1,
            zorder=d,
        )

    ax.view_init(h_angle, v_angle)
    ax.set_ylim(np.min(ys) - ydiff * 0.1, np.max(ys) + ydiff * 0.1)
    ax.set_xlim(np.min(xs) - xdiff * 0.1, np.max(xs) + xdiff * 0.1)
    ax.set_axis_off()

    return ax


# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:11:05 2021
@author: John Meluso
"""


def set_fonts(extra_params={}):
    params = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"],
        "mathtext.fontset": "cm",
        "legend.fontsize": 12,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 20,
        "font.size": 20,
    }
    for key, value in extra_params.items():
        params[key] = value
    plt.rcParams.update(params)
