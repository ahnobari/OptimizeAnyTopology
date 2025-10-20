import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

@njit(["f4[:](i4, i4, i4[:,:],f4[:])", "f8[:](i4, i4, i4[:,:], f8[:])"], parallel=True, cache=True, fastmath=True)
def get_filter_weights_2D(nelx, nely, offsets, weights):
    # Calculate normalization factors for each element
    normalization = np.zeros(nelx * nely, dtype=weights.dtype)
    for i in prange(nelx * nely):
        ix = i // nely
        iy = i % nely
        sum_weights = 0.0
        for k in range(len(weights)):
            dx, dy = offsets[k]
            ix_n = ix + dx
            iy_n = iy + dy
            if 0 <= ix_n < nelx and 0 <= iy_n < nely:
                sum_weights += weights[k]
        normalization[i] = sum_weights
    return normalization

@njit(["f4[:](i4, i4, i4, i4[:,:],f4[:])", "f8[:](i4, i4, i4, i4[:,:],f8[:])"], parallel=True, cache=True, fastmath=True)
def get_filter_weights_3D(nelx, nely, nelz, offsets, weights):
    # Calculate normalization factors for each element
    normalization = np.zeros(nelx * nely * nelz, dtype=weights.dtype)
    for i in prange(nelx * nely * nelz):
        ix = i // (nely * nelz)
        iy = (i // nelz) % nely
        iz = i % nelz
        sum_weights = 0.0
        for k in range(len(weights)):
            dx, dy, dz = offsets[k]
            ix_n = ix + dx
            iy_n = iy + dy
            iz_n = iz + dz
            if 0 <= ix_n < nelx and 0 <= iy_n < nely and 0 <= iz_n < nelz:
                sum_weights += weights[k]
        normalization[i] = sum_weights
    return normalization

@njit(["void(f4[:], f4[:], i4, i4, i4[:,:], f4[:], f4[:])", "void(f8[:], f8[:], i4, i4, i4[:,:], f8[:], f8[:])"], parallel=True, cache=True, fastmath=True)
def apply_filter_2D_parallel_transpose(v_in, v_out, nelx, nely, offsets, weights, normalization):
    for i in prange(nelx * nely):
        ix = i // nely
        iy = i % nely
        sum_value = 0.0
        
        for k in range(len(weights)):
            dx, dy = offsets[k]
            jx = ix + dx
            jy = iy + dy
            if 0 <= jx < nelx and 0 <= jy < nely:
                j_idx = jx * nely + jy
                sum_value += (weights[k] / normalization[j_idx]) * v_in[j_idx]
                
        v_out[i] = sum_value

@njit(["void(f4[:], f4[:], i4, i4, i4, i4[:,:], f4[:], f4[:])", "void(f8[:], f8[:], i4, i4, i4, i4[:,:], f8[:], f8[:])"], parallel=True, cache=True, fastmath=True)
def apply_filter_3D_parallel_transpose(v_in, v_out, nelx, nely, nelz, offsets, weights, normalization):
    for i in prange(nelx * nely * nelz):
        ix = i // (nely * nelz)
        iy = (i // nelz) % nely
        iz = i % nelz
        sum_value = 0.0
        
        for k in range(len(weights)):
            dx, dy, dz = offsets[k]
            jx = ix + dx
            jy = iy + dy
            jz = iz + dz
            if 0 <= jx < nelx and 0 <= jy < nely and 0 <= jz < nelz:
                j_idx = jx * nely * nelz + jy * nelz + jz
                sum_value += (weights[k] / normalization[j_idx]) * v_in[j_idx]
                
        v_out[i] = sum_value

@njit(["void(f4[:], f4[:], i4, i4, i4[:,:], f4[:])", "void(f8[:], f8[:], i4, i4, i4[:,:], f8[:])"], parallel=True, cache=True, fastmath=True)
def apply_filter_2D_parallel(v_in, v_out, nelx, nely, offsets, weights):
    # Loop over each element in the structured 2D grid in parallel
    for i in prange(nelx * nely):
        ix = i // nely
        iy = i % nely

        sum_weighted_values = 0.0
        sum_weights = 0.0
        for k in range(len(weights)):
            # Get the neighbor's offset
            dx, dy = offsets[k]
            ix_n = ix + dx
            iy_n = iy + dy
            # Check if neighbor is within bounds
            if 0 <= ix_n < nelx and 0 <= iy_n < nely:
                neighbor_idx = ix_n * nely + iy_n
                weight = weights[k]
                sum_weighted_values += weight * v_in[neighbor_idx]
                sum_weights += weight
        # Normalize by total weights and store result
        v_out[i] = sum_weighted_values / sum_weights if sum_weights > 0 else 0.0

@njit(["void(f4[:], f4[:], i4, i4, i4, i4[:,:], f4[:])", "void(f8[:], f8[:], i4, i4, i4, i4[:,:], f8[:])"], parallel=True, cache=True, fastmath=True)
def apply_filter_3D_parallel(v_in, v_out, nelx, nely, nelz, offsets, weights):
    # Loop over each element in the structured 3D grid in parallel
    for i in prange(nelx * nely * nelz):
        ix = i // (nely * nelz)
        iy = (i // nelz) % nely
        iz = i % nelz

        sum_weighted_values = 0.0
        sum_weights = 0.0
        for k in range(len(weights)):
            # Get the neighbor's offset
            dx, dy, dz = offsets[k]
            ix_n = ix + dx
            iy_n = iy + dy
            iz_n = iz + dz
            # Check if neighbor is within bounds
            if 0 <= ix_n < nelx and 0 <= iy_n < nely and 0 <= iz_n < nelz:
                neighbor_idx = ix_n * nely * nelz + iy_n * nelz + iz_n
                weight = weights[k]
                sum_weighted_values += weight * v_in[neighbor_idx]
                sum_weights += weight
        # Normalize by total weights and store result
        v_out[i] = sum_weighted_values / sum_weights if sum_weights > 0 else 0.0
    
def filter_kernel_3D_general(elements, element_centroids, r_min):

    search_tree = KDTree(element_centroids)
    Ne = search_tree.query_ball_point(element_centroids, r_min)

    filter_kernel_inds = []
    filter_kernel_vals = []
    for i in range(len(elements)):

        ws = (
            r_min
            - np.linalg.norm(element_centroids[Ne[i]] - element_centroids[i], axis=-1)
        ) / r_min
        ws = ws / ws.sum()
        filter_kernel_inds += np.pad(
            np.array(Ne[i]).reshape(-1, 1), [[0, 0], [1, 0]], constant_values=i
        ).tolist()
        filter_kernel_vals += ws.tolist()

    filter_kernel_inds = np.array(filter_kernel_inds, dtype=np.int32)
    filter_kernel_vals = np.array(filter_kernel_vals, dtype=element_centroids.dtype)

    filter_kernel = coo_matrix(
        (filter_kernel_vals, (filter_kernel_inds[:, 0], filter_kernel_inds[:, 1])),
        shape=[len(elements), len(elements)],
    dtype=element_centroids.dtype).tocsr()

    return filter_kernel

def filter_kernel_2D_general(elements, element_centroids, r_min):

    search_tree = KDTree(element_centroids)
    Ne = search_tree.query_ball_point(element_centroids, r_min)

    filter_kernel_inds = []
    filter_kernel_vals = []
    for i in range(len(elements)):

        ws = (
            r_min
            - np.linalg.norm(element_centroids[Ne[i]] - element_centroids[i], axis=-1)
        ) / r_min
        ws = ws / ws.sum()
        filter_kernel_inds += np.pad(
            np.array(Ne[i]).reshape(-1, 1), [[0, 0], [1, 0]], constant_values=i
        ).tolist()
        filter_kernel_vals += ws.tolist()

    filter_kernel_inds = np.array(filter_kernel_inds, dtype=np.int32)
    filter_kernel_vals = np.array(filter_kernel_vals, dtype=element_centroids.dtype)

    filter_kernel = coo_matrix(
        (filter_kernel_vals, (filter_kernel_inds[:, 0], filter_kernel_inds[:, 1])),
        shape=[len(elements), len(elements)], dtype=element_centroids.dtype
    ).tocsr()

    return filter_kernel