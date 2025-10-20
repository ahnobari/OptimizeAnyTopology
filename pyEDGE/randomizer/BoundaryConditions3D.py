import numpy as np


def generate_random_configuration(
    min_element,
    max_element,
    dim=2,
    min_nel_per_axis=10,
    multi_load=True,
    line_loads=0.5,
    surface_loads=0.0,
    point_loads=0.5,
    inner_loads=0.0,
    max_loads=5,
    line_bc=0.5,
    point_bc=0.5,
    surface_bc=0.0,
    inner_bc=0.0,
    max_bc=10,
    n_material=1,
    min_vf=0.1,
    max_vf=0.75,
    min_vf_per_material=0.02,
    dims=None,
):

    E_mat = np.random.uniform(size=n_material)
    E_mat /= E_mat.max()
    E_mat = E_mat.reshape(-1, 1)

    total_vf = np.random.uniform(min_vf, max_vf)
    vf = np.random.uniform(size=n_material)
    vf /= vf.sum()
    vf *= total_vf

    while np.any(vf < min_vf_per_material):
        vf = np.random.uniform(size=n_material)
        vf /= vf.sum()
        vf *= total_vf

    vf = vf.reshape(-1, 1)

    if dims is None:
        nel = np.random.randint(min_element, max_element + 1)
        dims = generate_random_domain(nel, min_nel_per_axis, dim)
    else:
        dims = dims

    if multi_load:
        n_loads = np.random.randint(1, max_loads + 1)
    else:
        n_loads = 1

    loads = []
    load_v = []

    load_mags = np.random.uniform(size=n_loads)
    load_mags /= load_mags.max()

    for i in range(n_loads):
        load, v = generate_random_load(
            dims, dim, point_loads, line_loads, surface_loads, inner_loads
        )
        loads.append(load)
        load_v.append(v * load_mags[i])

    loads = np.concatenate(loads)
    load_v = np.concatenate(load_v)

    # we need at least one boundary condition per dimension
    n_bc = np.random.randint(dim, max_bc + 1)

    bcs = []
    bc_c = []

    for i in range(n_bc):
        bc, c = generate_random_boundary_condition(
            dims, dim, line_bc, point_bc, surface_bc, inner_bc
        )
        bcs.append(bc)
        bc_c.append(c)

    bcs = np.concatenate(bcs)
    bc_c = np.concatenate(bc_c)

    while not np.all(bc_c.sum(axis=0) > 0):
        bcs = []
        bc_c = []

        for i in range(n_bc):
            bc, c = generate_random_boundary_condition(
                dims, dim, line_bc, point_bc, surface_bc, inner_bc
            )
            bcs.append(bc)
            bc_c.append(c)

        bcs = np.concatenate(bcs)
        bc_c = np.concatenate(bc_c)

    return dims, loads, load_v, bcs, bc_c, E_mat, vf


def generate_random_boundary_condition(
    dims, dim, line_bc=1 / 3, point_bc=1 / 3, surface_bc=1 / 3, inner_bc=0.0
):

    if dim == 2:
        surface_bc = 0
        line_bc /= point_bc + line_bc
        point_bc /= point_bc + line_bc
    else:
        surface_bc /= point_bc + line_bc + surface_bc
        point_bc /= point_bc + line_bc + surface_bc
        line_bc /= point_bc + line_bc + surface_bc

    is_inner = np.random.uniform() < inner_bc

    bc_type = np.random.uniform()
    if bc_type < point_bc:
        bc_type = 0
    elif bc_type < point_bc + line_bc:
        bc_type = 1
    else:
        bc_type = 2

    if bc_type == 0:
        if is_inner:
            location = np.random.randint(1, dims)
        else:
            location = np.random.randint(0, dims + 1)
            # pick a random axis to max or min
            axis = np.random.randint(0, dim)
            location[axis] = np.random.choice([0, dims[axis]])

        constraint = np.random.randint(0, 2, size=dim)

        return location[None], constraint[None]

    elif bc_type == 1:
        if is_inner:
            # pick two random points in the domain for the line
            p1 = np.random.randint(1, dims)
            p2 = np.random.randint(1, dims)

            while np.all(p1 == p2):
                p2 = np.random.randint(1, dims)

            # make a list of all points on the line
            line_length = np.linalg.norm(p2 - p1)
            line = np.zeros((int(line_length * 5), dim))
            increment = (p2 - p1) / line.shape[0]
            line += p1[None]
            line += np.cumsum(np.ones((line.shape[0], dim)) * increment[None], axis=0)
            line = line.astype(int)
            line = np.unique(line, axis=0)

            constraint = np.random.randint(0, 2, size=dim)[None].repeat(
                line.shape[0], axis=0
            )

            return line, constraint.astype(bool)

        else:
            # pick a random axis to max or min
            axis = np.random.randint(0, dim)
            side = np.random.choice([0, dims[axis]])

            # pick two random points in the domain for the line
            p1 = np.random.randint(0, dims)
            p1[axis] = side
            p2 = np.random.randint(0, dims)
            p2[axis] = side

            while np.all(p1 == p2):
                p2 = np.random.randint(0, dims)
                p2[axis] = side

            # make a list of all points on the line
            line_length = np.linalg.norm(p2 - p1)
            line = np.zeros((int(line_length * 5), dim))
            increment = (p2 - p1) / line.shape[0]
            line += p1[None]
            line += np.cumsum(np.ones((line.shape[0], dim)) * increment[None], axis=0)
            line = line.astype(int)
            line = np.unique(line, axis=0)

            constraint = np.random.randint(0, 2, size=dim)[None].repeat(
                line.shape[0], axis=0
            )

            return line, constraint.astype(bool)

    else:
        # surface load only on the outer surfaces
        # pick a random axis to max or min
        axis = np.random.randint(0, dim)
        side = np.random.choice([0, dims[axis]])

        # pick two points on the surface to define the recangle for the surface load
        p1 = np.random.randint(0, dims)
        p1[axis] = side
        p2 = np.random.randint(0, dims)
        p2[axis] = side

        while np.all(p1 == p2):
            p2 = np.random.randint(0, dims)
            p2[axis] = side

        # make a list of all points on the surface
        axes = np.arange(dim)[np.arange(dim) != axis]
        mesh_grid = np.meshgrid(
            np.arange(
                np.min([p1[axes[0]], p2[axes[0]]]),
                np.max([p1[axes[0]], p2[axes[0]]]) + 1,
            ),
            np.arange(
                np.min([p1[axes[1]], p2[axes[1]]]),
                np.max([p1[axes[1]], p2[axes[1]]]) + 1,
            ),
        )
        # surface = np.concatenate([mesh_grid[0].reshape(-1,1), mesh_grid[1].reshape(-1,1)], axis=1)
        surface = np.zeros((mesh_grid[0].size, dim))
        surface[:, axes[0]] = mesh_grid[0].reshape(-1)
        surface[:, axes[1]] = mesh_grid[1].reshape(-1)
        surface[:, axis] = side
        surface = np.unique(surface, axis=0)

        constraint = np.random.randint(0, 2, size=dim)[None].repeat(
            surface.shape[0], axis=0
        )

        return surface, constraint.astype(bool)


def generate_random_load(
    dims, dim, point_load=1 / 3, line_loads=1 / 3, surface_loads=1 / 3, inner_loads=0.5
):

    if dim == 2:
        surface_loads = 0
        line_loads /= point_load + line_loads
        point_load /= point_load + line_loads
    else:
        surface_loads /= point_load + line_loads + surface_loads
        point_load /= point_load + line_loads + surface_loads
        line_loads /= point_load + line_loads + surface_loads

    is_inner = np.random.uniform() < inner_loads

    load_type = np.random.uniform()
    if load_type < point_load:
        load_type = 0
    elif load_type < point_load + line_loads:
        load_type = 1
    else:
        load_type = 2

    if load_type == 0:
        if is_inner:
            location = np.random.randint(1, dims)
        else:
            location = np.random.randint(0, dims + 1)
            # pick a random axis to max or min
            axis = np.random.randint(0, dim)
            location[axis] = np.random.choice([0, dims[axis]])

        vector = np.random.uniform(size=dim) - 0.5
        vector /= np.linalg.norm(vector)

        return location[None], vector[None]

    elif load_type == 1:
        if is_inner:
            # pick two random points in the domain for the line
            p1 = np.random.randint(1, dims)
            p2 = np.random.randint(1, dims)

            while np.all(p1 == p2):
                p2 = np.random.randint(1, dims)

            # make a list of all points on the line
            line_length = np.linalg.norm(p2 - p1)
            line = np.zeros((int(line_length * 5), dim))
            increment = (p2 - p1) / line.shape[0]
            line += p1[None]
            line += np.cumsum(np.ones((line.shape[0], dim)) * increment[None], axis=0)
            line = line.astype(int)
            line = np.unique(line, axis=0)

            vector = np.ones_like(line) * (np.random.uniform(size=dim)-0.5)
            vector /= np.linalg.norm(vector, axis=1)[:, None]
            vector /= vector.shape[0]

            return line, vector

        else:
            # pick a random axis to max or min
            axis = np.random.randint(0, dim)
            side = np.random.choice([0, dims[axis]])

            # pick two random points in the domain for the line
            p1 = np.random.randint(0, dims)
            p1[axis] = side
            p2 = np.random.randint(0, dims)
            p2[axis] = side

            while np.all(p1 == p2):
                p2 = np.random.randint(0, dims)
                p2[axis] = side

            # make a list of all points on the line
            line_length = np.linalg.norm(p2 - p1)
            line = np.zeros((int(line_length * 5), dim))
            increment = (p2 - p1) / line.shape[0]
            line += p1[None]
            line += np.cumsum(np.ones((line.shape[0], dim)) * increment[None], axis=0)
            line = line.astype(int)
            line = np.unique(line, axis=0)

            vector = np.ones_like(line) * (np.random.uniform(size=dim)-0.5)
            vector /= np.linalg.norm(vector, axis=1)[:, None]
            vector /= vector.shape[0]

            return line, vector

    else:
        # surface load only on the outer surfaces
        # pick a random axis to max or min
        axis = np.random.randint(0, dim)
        side = np.random.choice([0, dims[axis]])

        # pick two points on the surface to define the recangle for the surface load
        p1 = np.random.randint(0, dims)
        p1[axis] = side
        p2 = np.random.randint(0, dims)
        p2[axis] = side

        while np.all(p1 == p2):
            p2 = np.random.randint(0, dims)
            p2[axis] = side

        # make a list of all points on the surface
        axes = np.arange(dim)[np.arange(dim) != axis]
        mesh_grid = np.meshgrid(
            np.arange(
                np.min([p1[axes[0]], p2[axes[0]]]),
                np.max([p1[axes[0]], p2[axes[0]]]) + 1,
            ),
            np.arange(
                np.min([p1[axes[1]], p2[axes[1]]]),
                np.max([p1[axes[1]], p2[axes[1]]]) + 1,
            ),
        )
        # surface = np.concatenate([mesh_grid[0].reshape(-1,1), mesh_grid[1].reshape(-1,1)], axis=1)
        surface = np.zeros((mesh_grid[0].size, dim))
        surface[:, axes[0]] = mesh_grid[0].reshape(-1)
        surface[:, axes[1]] = mesh_grid[1].reshape(-1)
        surface[:, axis] = side
        surface = np.unique(surface, axis=0)

        vector = np.ones_like(surface) * (np.random.uniform(size=dim)-0.5)
        vector /= np.linalg.norm(vector, axis=1)[:, None]
        vector /= vector.shape[0]

        return surface, vector


def generate_random_domain(nel, min_nel_per_axis=10, dim=2):
    max_nel_per_axis = int(np.floor(nel / min_nel_per_axis ** (dim - 1)))

    dims = np.zeros(dim).astype(int)

    for i in range(dim - 1):
        dims[i] = np.random.randint(min_nel_per_axis, max_nel_per_axis + 1)
        max_nel_per_axis = int(
            np.floor(nel / dims[i] / min_nel_per_axis ** (dim - 2 - i))
        )

    dims[-1] = nel // np.prod(dims[:-1])
    return dims


def create_random_configuration_from_list(domains, BCs, dim_idx = None, bc_idx = None, multi_load=False, line_loads=0.0, surface_loads=0.0, point_loads=1.0, inner_loads=0.0, max_loads=1, n_material=4, min_vf=0.16, max_vf=0.6, min_vf_per_material=0.04):
    
    if dim_idx is None:
        dim_idx = np.random.choice(domains.shape[0])
    dims = domains[dim_idx]
    if bc_idx is None:
        bc_idx = np.random.choice(len(BCs[dim_idx]))
    bcs, bc_c = BCs[dim_idx][bc_idx]
    
    dim = dims.shape[0]
    
    if n_material > 1:
        n_material = np.random.randint(2, n_material + 1)
    
    _, loads, load_v, _, _, E_mat, vf = generate_random_configuration(0,0, dim, 0, multi_load, line_loads, surface_loads, point_loads, inner_loads, max_loads, 0, 1, 0, 0, dim, n_material, min_vf, max_vf, min_vf_per_material, dims=dims)
    
    return (dims, loads, load_v, bcs, bc_c, E_mat, vf), (dim_idx,bc_idx)
