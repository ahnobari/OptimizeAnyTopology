import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import markers
from matplotlib.path import Path

def align_marker(marker, halign="center", valign="middle"):

    if isinstance(halign, (str)):
        halign = {
            "right": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "left": 1.0,
        }[halign]

    if isinstance(valign, (str)):
        valign = {
            "top": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "bottom": 1.0,
        }[valign]

    bm = markers.MarkerStyle(marker)

    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)


def plot_problem_2D(
    nodes: np.ndarray,
    elements: np.ndarray,
    c: np.ndarray,
    f: np.ndarray,
    ax=None,
    face_color="grey",
    edge_color="black",
    x_color="tomato",
    y_color="royalblue",
    f_color="#8e0000",
    rho=None,
    **kwargs,
):

    if rho is not None:
        if rho.ndim > 1:
            elements_ = []
            for i in range(rho.shape[1]):
                elements_.append(elements[rho[:,i]>0.5])
            elements = elements_
        else:
            elements = elements[rho>0.5]
    else:
        rho = np.empty([elements.shape[0]])
    
    if ax is None:
        ax = plt.gca()

    y = nodes[:, 0]
    z = nodes[:, 1]

    def quatplot(y, z, quatrangles, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    ax.set_aspect("equal")
    if rho.ndim > 1:
        if isinstance(face_color, list):
            fc = face_color
        else:
            # If face_color is not a list, then in multi-material case we use viridis colormap
            fc = plt.cm.viridis(np.linspace(0, 1, rho.shape[1]))
        for j in range(rho.shape[1]):
            elements_ = []
            for e in elements[j]:
                if len(e) == 4:
                    elements_.append([e[0], e[1], e[2], e[3]])
                else:
                    elements_.append([e[0], e[1], e[2], e[2]])

            quatplot(y, z, np.asarray(elements_), ax=ax, color=edge_color, facecolor=fc[j])
        # add legend for each material
        for i in range(rho.shape[1]):
            ax.plot([], [], color=fc[i], label=f"Material {i+1}", lw=8)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, handlelength=1)
    else:
        elements_ = []
        for e in elements:
            if len(e) == 4:
                elements_.append([e[0], e[1], e[2], e[3]])
            else:
                elements_.append([e[0], e[1], e[2], e[2]])

        quatplot(y, z, np.asarray(elements_), ax=ax, color=edge_color, facecolor=face_color)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    x_bc = c[:, 0] != 0
    y_bc = c[:, 1] != 0

    ax.scatter(
        nodes[x_bc][:, 0],
        nodes[x_bc][:, 1],
        marker=align_marker(">", "right", "middle"),
        s=500,
        color=x_color,
        alpha=0.7,
    )
    ax.scatter(
        nodes[y_bc][:, 0],
        nodes[y_bc][:, 1],
        marker=align_marker("^", "middle", "top"),
        s=500,
        color=y_color,
        alpha=0.7,
    )

    force_nodes = (f != 0).sum(1) > 0

    ax.quiver(
        nodes[force_nodes, 0],
        nodes[force_nodes, 1],
        f[force_nodes, 0] / np.abs(f).max(),
        f[force_nodes, 1] / np.abs(f).max(),
        color=f_color,
        scale=15,
        width=0.005,
    )

    return ax


def plot_mesh_2D(
    nodes: np.ndarray,
    elements: np.ndarray,
    ax=None,
    face_color="grey", 
    edge_color="black",
    rho=None,
    **kwargs
):

    if nodes.shape[1] != 2:
        raise ValueError("This function only supports 2D meshes")

    if ax is None:
        ax = plt.gca()

    y = nodes[:, 0]
    z = nodes[:, 1]
        
    def quatplot(y, z, quatrangles, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    ax.set_aspect("equal")

    elements_ = []
        
    if rho is not None:
        elements = elements[rho>0.5]
        
    for e in elements:
        if len(e) == 4:
            elements_.append([e[0], e[1], e[2], e[3]])
        else:
            elements_.append([e[0], e[1], e[2], e[2]])

    quatplot(y, z, np.asarray(elements_), ax=ax, color=edge_color, facecolor=face_color)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    return ax

def plot_field_2D(
    nodes: np.ndarray,
    elements: np.ndarray,
    field: np.ndarray,
    rho = None,
    ax=None,
    face_color="grey",
    edge_color="black",
    colormap='viridis',
    show_colorbar=True,
    colorbar_label=None,
    **kwargs,
):

    if rho is not None:
        elements = elements[rho>0.5]
        field = field[rho>0.5]
    
    if ax is None:
        ax = plt.gca()

    y = nodes[:, 0]
    z = nodes[:, 1]

    def quatplot(y, z, quatrangles, field_values, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        pc.set_array(field_values)
        ax.add_collection(pc)
        ax.autoscale()
        return pc

    ax.set_aspect("equal")

    elements_ = []
    for e in elements:
        if len(e) == 4:
            elements_.append([e[0], e[1], e[2], e[3]])
        else:
            elements_.append([e[0], e[1], e[2], e[2]])

    pc = quatplot(y, z, np.asarray(elements_), field, ax=ax, 
                 edgecolor=edge_color, cmap=colormap, **kwargs)

    if show_colorbar:
        cbar = plt.colorbar(pc, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    return ax