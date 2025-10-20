from numba import njit, prange
from scipy.sparse import csr_matrix
import numpy as np

from ...geom.CPU._mesh import StructuredMesh2D, StructuredMesh3D
from ...stiffness.CPU._FEA import StructuredStiffnessKernel
from typing import Union


@njit(["f4[:](f4[:], i4[:], i4)", "f8[:](f8[:], i4[:], i4)"], cache=True, parallel=True)
def restriction_matvec_3d(v, nel, dof):
    nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
    nel_coarse = (nel[0] // 2, nel[1] // 2, nel[2] // 2)
    nx_coarse, ny_coarse, nz_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1, nel_coarse[2] + 1
    
    n_coarse_nodes = nx_coarse * ny_coarse * nz_coarse
    result = np.zeros(n_coarse_nodes * dof, dtype=v.dtype)
    
    # Parallel loop over coarse grid points
    for idx_coarse in prange(n_coarse_nodes):
        # Convert linear index to 3D coordinates
        k_coarse = idx_coarse % nz_coarse
        tmp = idx_coarse // nz_coarse
        i_coarse = tmp % nx_coarse
        j_coarse = tmp // nx_coarse
        
        # Corresponding fine grid coordinates
        i_fine = i_coarse * 2
        j_fine = j_coarse * 2
        k_fine = k_coarse * 2
        
        # Initialize weight accumulator for normalization
        boundaries = 0
        
        if i_coarse == 0 or i_coarse == nx_coarse-1:
            boundaries += 1
        if j_coarse == 0 or j_coarse == ny_coarse-1:
            boundaries += 1
        if k_coarse == 0 or k_coarse == nz_coarse-1:
            boundaries += 1
        
        if boundaries == 3:
            total_weight = 3.375
        elif boundaries == 2:
            total_weight = 4.5
        elif boundaries == 1:
            total_weight = 6.0
        else:
            total_weight = 8.0
        
        # For each DOF
        for d in range(dof):
            coarse_idx_dof = idx_coarse * dof + d
            value = 0.0
            
            # Loop over neighborhood in fine grid
            for di in range(-1, 2):
                i_neighbor = i_fine + di
                if i_neighbor < 0 or i_neighbor >= nx_fine:
                    continue
                    
                for dj in range(-1, 2):
                    j_neighbor = j_fine + dj
                    if j_neighbor < 0 or j_neighbor >= ny_fine:
                        continue
                        
                    for dk in range(-1, 2):
                        k_neighbor = k_fine + dk
                        if k_neighbor < 0 or k_neighbor >= nz_fine:
                            continue
                            
                        # Calculate distance and weight
                        distance = abs(di) + abs(dj) + abs(dk)
                        if distance > 3:
                            continue
                            
                        weight = 1.0 if distance == 0 else (
                            0.5 if distance == 1 else (
                            0.25 if distance == 2 else 0.125))
                        
                        # Calculate fine grid linear index
                        idx_fine = (k_neighbor + nz_fine * i_neighbor + 
                                  nz_fine * nx_fine * j_neighbor)
                        
                        # Accumulate weighted value
                        value += weight * v[idx_fine * dof + d]
            
            # Normalize and store result
            if total_weight > 0:
                result[coarse_idx_dof] = value / total_weight
                
    return result

@njit(["f4[:](f4[:], i4[:], i4)", "f8[:](f8[:], i4[:], i4)"], cache=True, parallel=True)
def restriction_matvec_2d(v, nel, dof):
    nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
    nel_coarse = (nel[0] // 2, nel[1] // 2)
    nx_coarse, ny_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1
    
    n_coarse_nodes = nx_coarse * ny_coarse
    result = np.zeros(n_coarse_nodes * dof, dtype=v.dtype)
    
    # Parallel loop over coarse grid points
    for idx_coarse in prange(n_coarse_nodes):
        # Convert linear index to 2D coordinates
        i_coarse = idx_coarse % nx_coarse
        j_coarse = idx_coarse // nx_coarse
        
        # Corresponding fine grid coordinates
        i_fine = i_coarse * 2
        j_fine = j_coarse * 2
        
        # Initialize weight accumulator for normalization
        boundaries = 0
        
        if i_coarse == 0 or i_coarse == nx_coarse-1:
            boundaries += 1
        if j_coarse == 0 or j_coarse == ny_coarse-1:
            boundaries += 1
        
        if boundaries == 2:
            total_weight = 2.25
        elif boundaries == 1:
            total_weight = 3.0
        else:
            total_weight = 4.0
        
        # For each DOF
        for d in range(dof):
            coarse_idx_dof = idx_coarse * dof + d
            value = 0.0
            
            # Loop over neighborhood in fine grid
            for di in range(-1, 2):
                i_neighbor = i_fine + di
                if i_neighbor < 0 or i_neighbor >= nx_fine:
                    continue
                    
                for dj in range(-1, 2):
                    j_neighbor = j_fine + dj
                    if j_neighbor < 0 or j_neighbor >= ny_fine:
                        continue
                        
                    # Calculate distance and weight
                    distance = abs(di) + abs(dj)
                    if distance > 2:
                        continue
                        
                    weight = 1.0 if distance == 0 else (
                        0.5 if distance == 1 else 0.25)
                    
                    # Calculate fine grid linear index
                    idx_fine = i_neighbor + nx_fine * j_neighbor
                    
                    # Accumulate weighted value
                    value += weight * v[idx_fine * dof + d]
            
            # Normalize and store result
            if total_weight > 0:
                result[coarse_idx_dof] = value / total_weight
                
    return result

def apply_restriction(v, nel, dof):

    if len(nel) == 3:
        return restriction_matvec_3d(v, nel, dof)
    else:
        return restriction_matvec_2d(v, nel, dof)
    
@njit(["f4[:](f4[:], i4[:], i4)", "f8[:](f8[:], i4[:], i4)"], cache=True, parallel=True)
def prolongation_matvec_3d(v, nel, dof):

    # Fine grid dimensions
    nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
    
    # Coarse grid dimensions
    nx_coarse = nel[0] // 2 + 1
    ny_coarse = nel[1] // 2 + 1
    nz_coarse = nel[2] // 2 + 1
    
    n_fine_nodes = nx_fine * ny_fine * nz_fine
    result = np.zeros(n_fine_nodes * dof, dtype=v.dtype)
    
    # Parallel loop over fine grid points
    for idx_fine in prange(n_fine_nodes):
        # Convert linear index to 3D coordinates
        k = idx_fine % nz_fine
        tmp = idx_fine // nz_fine
        i = tmp % nx_fine
        j = tmp // nx_fine
        
        # Get coarse grid indices and ranges for interpolation
        i_coarse = i // 2
        j_coarse = j // 2
        k_coarse = k // 2
        
        # Count contributing nodes and accumulate values
        
        for d in range(dof):
            value = 0.0
            count = 0
            for ci in range(i_coarse, i_coarse + i % 2 + 1):
                if ci >= nx_coarse:
                    continue
                for cj in range(j_coarse, j_coarse + j % 2 + 1):
                    if cj >= ny_coarse:
                        continue
                    for ck in range(k_coarse, k_coarse + k % 2 + 1):
                        if ck >= nz_coarse:
                            continue
                        
                        boundaries = 0
        
                        if ci == 0 or ci == nx_coarse-1:
                            boundaries += 1
                        if cj == 0 or cj == ny_coarse-1:
                            boundaries += 1
                        if ck == 0 or ck == nz_coarse-1:
                            boundaries += 1
                        
                        if boundaries == 3:
                            total_weight = 3.375
                        elif boundaries == 2:
                            total_weight = 4.5
                        elif boundaries == 1:
                            total_weight = 6.0
                        else:
                            total_weight = 8.0
                        
                        
                        idx_coarse = ck + nz_coarse * ci + nz_coarse * nx_coarse * cj
                        value += v[idx_coarse * dof + d]/total_weight
                        count += 1
                            
            if count > 0:
                result[idx_fine * dof + d] = value / count
                    
    return result

@njit(["f4[:](f4[:], i4[:], i4)", "f8[:](f8[:], i4[:], i4)"], cache=True, parallel=True)
def prolongation_matvec_2d(v, nel, dof):

    # Fine grid dimensions
    nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
    
    # Coarse grid dimensions
    nx_coarse = nel[0] // 2 + 1
    ny_coarse = nel[1] // 2 + 1
    
    n_fine_nodes = nx_fine * ny_fine
    result = np.zeros(n_fine_nodes * dof, dtype=v.dtype)
    
    # Parallel loop over fine grid points
    for idx_fine in prange(n_fine_nodes):
        # Convert linear index to 2D coordinates
        i = idx_fine % nx_fine
        j = idx_fine // nx_fine
        
        # Get coarse grid indices and ranges for interpolation
        i_coarse = i // 2
        j_coarse = j // 2
        
        # Count contributing nodes and accumulate values
        
        for d in range(dof):
            value = 0.0
            count = 0
            for ci in range(i_coarse, i_coarse + i % 2 + 1):
                if ci >= nx_coarse:
                    continue
                for cj in range(j_coarse, j_coarse + j % 2 + 1):
                    if cj >= ny_coarse:
                        continue
                    
                    boundaries = 0
                    
                    if ci == 0 or ci == nx_coarse-1:
                        boundaries += 1
                    if cj == 0 or cj == ny_coarse-1:
                        boundaries += 1
                    
                    if boundaries == 2:
                        total_weight = 2.25
                    elif boundaries == 1:
                        total_weight = 3.0
                    else:
                        total_weight = 4.0
                    
                    idx_coarse = ci + nx_coarse * cj
                    value += v[idx_coarse * dof + d]/total_weight
                    count += 1
                        
            if count > 0:
                result[idx_fine * dof + d] = value / count
                    
    return result

def apply_prolongation(v, nel, dof):

    if len(nel) == 3:
        return prolongation_matvec_3d(v, nel, dof)
    else:
        return prolongation_matvec_2d(v, nel, dof)
    
@njit(["void(f4[:], f4[:], i4, i4, i4, i4[:], i4[:], i4[:], i4[:], f4[:], i4, i4, bool_[:], i4, i4, i4, i4, i4, i4)", 
       "void(f8[:], f8[:], i4, i4, i4, i4[:], i4[:], i4[:], i4[:], f8[:], i4, i4, bool_[:], i4, i4, i4, i4, i4, i4)"], cache=True)
def get_target_vals(vals, K_flat, idx_fine, dof, elements_size, el_ids, node_ids, sorter, elements_flat, weights, elem_flat_size, n_nodes, con_map, i_fine_target, j_fine_target, k_fine_target, nx_fine, ny_fine, nz_fine):
    i = idx_fine
    if i < n_nodes-1:
        st = node_ids[i]
        en = node_ids[i+1]
    else:
        st = node_ids[i]
        en = elem_flat_size

    n_elements = en-st

    for k in range(dof):
        if con_map[idx_fine*dof+k]:
            fine_node = idx_fine
            k_fine = fine_node % nz_fine
            tmp = fine_node // nz_fine
            i_fine = tmp % nx_fine
            j_fine = tmp // nx_fine

            di = i_fine - i_fine_target
            dj = j_fine - j_fine_target
            dk = k_fine - k_fine_target

            if di < -1 or di > 1:
                continue
            if dj < -1 or dj > 1:
                continue
            if dk < -1 or dk > 1:
                continue

            distance = abs(di) + abs(dj) + abs(dk)
            if distance > 3:
                continue

            weight = 1.0 if distance == 0 else (0.5 if distance == 1 else (0.25 if distance == 2 else 0.125))
            vals[k*dof+k] = weight

    for j in range(n_elements):
        e_ind_j = sorter[st+j]
        elements_ids_j = el_ids[e_ind_j]
        start = elements_ids_j * elements_size
        weight = weights[elements_ids_j]
        
        relative_dof = e_ind_j - start
        k_start = relative_dof * dof * elements_size * dof

        for l in range(elements_size):
            fine_node = elements_flat[start+l]
            k_fine = fine_node % nz_fine
            tmp = fine_node // nz_fine
            i_fine = tmp % nx_fine
            j_fine = tmp // nx_fine

            di = i_fine - i_fine_target
            dj = j_fine - j_fine_target
            dk = k_fine - k_fine_target

            if di < -1 or di > 1:
                continue
            if dj < -1 or dj > 1:
                continue
            if dk < -1 or dk > 1:
                continue

            distance = abs(di) + abs(dj) + abs(dk)
            if distance > 3:
                continue

            weight = 1.0 if distance == 0 else (0.5 if distance == 1 else (0.25 if distance == 2 else 0.125))

            for k in range(dof):
                if con_map[i*dof+k]:
                    continue
                for kk in range(dof):
                    if con_map[fine_node*dof+kk]:
                        continue
                    vals[k*dof+kk] += K_flat[k_start + k*elements_size*dof + l*dof + kk] * weight * weights[elements_ids_j]

@njit(["void(f4[:], i4, i4, i4, i4, i4, i4, i4[:], i4[:], f4[:], i4, i4[:], i4[:], i4[:], i4[:], f4[:], i4, bool_[:], i4, i4)",
       "void(f8[:], i4, i4, i4, i4, i4, i4, i4[:], i4[:], f8[:], i4, i4[:], i4[:], i4[:], i4[:], f8[:], i4, bool_[:], i4, i4)"], cache=True, parallel=True)
def get_restricted_3d_l0_nnz_based(K_flat, nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse, Cp, Cj, Cx, dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes):
    for thread in prange(nx_coarse * ny_coarse * nz_coarse * 27):
        idx_coarse = thread // 27
        neighbour_idx = thread % 27

        k_coarse = idx_coarse % nz_coarse
        tmp = idx_coarse // nz_coarse
        i_coarse = tmp % nx_coarse
        j_coarse = tmp // nx_coarse

        i_offset = neighbour_idx % 3 - 1
        j_offset = (neighbour_idx // 3) % 3 - 1
        k_offset = neighbour_idx // 9 - 1

        if i_coarse + i_offset < 0 or i_coarse + i_offset >= nx_coarse:
            continue
        if j_coarse + j_offset < 0 or j_coarse + j_offset >= ny_coarse:
            continue
        if k_coarse + k_offset < 0 or k_coarse + k_offset >= nz_coarse:
            continue

        target_coarse = k_coarse + k_offset + nz_coarse * (i_coarse + i_offset) + nz_coarse * nx_coarse * (j_coarse + j_offset)

        i_fine_target = (i_coarse + i_offset) * 2
        j_fine_target = (j_coarse + j_offset) * 2
        k_fine_target = (k_coarse + k_offset) * 2

        i_fine = i_coarse * 2
        j_fine = j_coarse * 2
        k_fine = k_coarse * 2

        boundaries = 0
        if i_coarse == 0 or i_coarse == nx_coarse-1:
            boundaries += 1
        if j_coarse == 0 or j_coarse == ny_coarse-1:
            boundaries += 1
        if k_coarse == 0 or k_coarse == nz_coarse-1:
            boundaries += 1

        total_weight = 3.375 if boundaries == 3 else (4.5 if boundaries == 2 else (6.0 if boundaries == 1 else 8.0))

        boundaries = 0
        if i_coarse + i_offset == 0 or i_coarse + i_offset == nx_coarse-1:
            boundaries += 1
        if j_coarse + j_offset == 0 or j_coarse + j_offset == ny_coarse-1:
            boundaries += 1
        if k_coarse + k_offset == 0 or k_coarse + k_offset == nz_coarse-1:
            boundaries += 1

        total_weight_target = 3.375 if boundaries == 3 else (4.5 if boundaries == 2 else (6.0 if boundaries == 1 else 8.0))

        for k in range(dof):
            start = Cp[idx_coarse*dof + k]
            for kk in range(dof):
                Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk

        vals = np.zeros(dof*dof, dtype=K_flat.dtype)

        for di in range(-1, 2):
            i_neighbor = i_fine + di
            if i_neighbor < 0 or i_neighbor >= nx_fine:
                continue

            for dj in range(-1, 2):
                j_neighbor = j_fine + dj
                if j_neighbor < 0 or j_neighbor >= ny_fine:
                    continue

                for dk in range(-1, 2):
                    k_neighbor = k_fine + dk
                    if k_neighbor < 0 or k_neighbor >= nz_fine:
                        continue

                    distance = abs(di) + abs(dj) + abs(dk)
                    if distance > 3:
                        continue

                    weight = 1.0 if distance == 0 else (0.5 if distance == 1 else (0.25 if distance == 2 else 0.125))
                    idx_fine = k_neighbor + nz_fine * i_neighbor + nz_fine * nx_fine * j_neighbor

                    vals.fill(0.0)
                    get_target_vals(vals, K_flat, idx_fine, dof, elements_size, el_ids, node_ids, sorter, elements_flat, weights, elem_flat_size, n_nodes, con_map, i_fine_target, j_fine_target, k_fine_target, nx_fine, ny_fine, nz_fine)

                    for k in range(dof):
                        start = Cp[idx_coarse*dof + k]
                        for kk in range(dof):
                            Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target
                            
@njit(["void(f4[:], f4[:], i4, i4, i4, i4[:], i4[:], i4[:], i4[:], f4[:], i4, i4, bool_[:], i4, i4, i4, i4)", 
       "void(f8[:], f8[:], i4, i4, i4, i4[:], i4[:], i4[:], i4[:], f8[:], i4, i4, bool_[:], i4, i4, i4, i4)"], cache=True)
def get_target_vals_2d(vals, K_flat, idx_fine, dof, elements_size, el_ids, node_ids, sorter, elements_flat, weights, elem_flat_size, n_nodes, con_map, i_fine_target, j_fine_target, nx_fine, ny_fine):
    i = idx_fine
    if i < n_nodes-1:
        st = node_ids[i]
        en = node_ids[i+1]
    else:
        st = node_ids[i]
        en = elem_flat_size

    n_elements = en-st

    for k in range(dof):
        if con_map[idx_fine*dof+k]:
            fine_node = idx_fine
            i_fine = fine_node % nx_fine
            j_fine = fine_node // nx_fine

            di = i_fine - i_fine_target
            dj = j_fine - j_fine_target

            if di < -1 or di > 1:
                continue
            if dj < -1 or dj > 1:
                continue

            distance = abs(di) + abs(dj)
            if distance > 2:
                continue

            weight = 1.0 if distance == 0 else (0.5 if distance == 1 else 0.25)
            vals[k*dof+k] = weight

    for j in range(n_elements):
        e_ind_j = sorter[st+j]
        elements_ids_j = el_ids[e_ind_j]
        start = elements_ids_j * elements_size
        weight = weights[elements_ids_j]
        
        relative_dof = e_ind_j - start
        k_start = relative_dof * dof * elements_size * dof

        for l in range(elements_size):
            fine_node = elements_flat[start+l]
            i_fine = fine_node % nx_fine
            j_fine = fine_node // nx_fine

            di = i_fine - i_fine_target
            dj = j_fine - j_fine_target

            if di < -1 or di > 1:
                continue
            if dj < -1 or dj > 1:
                continue

            distance = abs(di) + abs(dj)
            if distance > 2:
                continue

            weight = 1.0 if distance == 0 else (0.5 if distance == 1 else 0.25)

            for k in range(dof):
                if con_map[i*dof+k]:
                    continue
                for kk in range(dof):
                    if con_map[fine_node*dof+kk]:
                        continue
                    vals[k*dof+kk] += K_flat[k_start + k*elements_size*dof + l*dof + kk] * weight * weights[elements_ids_j]

@njit(["void(f4[:], i4, i4, i4, i4, i4[:], i4[:], f4[:], i4, i4[:], i4[:], i4[:], i4[:], f4[:], i4, bool_[:], i4, i4)",
       "void(f8[:], i4, i4, i4, i4, i4[:], i4[:], f8[:], i4, i4[:], i4[:], i4[:], i4[:], f8[:], i4, bool_[:], i4, i4)"], cache=True, parallel=True)
def get_restricted_2d_l0_nnz_based(K_flat, nx_fine, ny_fine, nx_coarse, ny_coarse, Cp, Cj, Cx, dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes):
    for thread in prange(nx_coarse * ny_coarse * 9):
        idx_coarse = thread // 9
        neighbour_idx = thread % 9

        i_coarse = idx_coarse % nx_coarse
        j_coarse = idx_coarse // nx_coarse

        i_offset = neighbour_idx % 3 - 1
        j_offset = neighbour_idx // 3 - 1

        if i_coarse + i_offset < 0 or i_coarse + i_offset >= nx_coarse:
            continue
        if j_coarse + j_offset < 0 or j_coarse + j_offset >= ny_coarse:
            continue

        target_coarse = i_coarse + i_offset + nx_coarse * (j_coarse + j_offset)

        i_fine_target = (i_coarse + i_offset) * 2
        j_fine_target = (j_coarse + j_offset) * 2

        i_fine = i_coarse * 2
        j_fine = j_coarse * 2

        boundaries = 0
        if i_coarse == 0 or i_coarse == nx_coarse-1:
            boundaries += 1
        if j_coarse == 0 or j_coarse == ny_coarse-1:
            boundaries += 1

        total_weight = 2.25 if boundaries == 2 else (3.0 if boundaries == 1 else 4.0)

        boundaries = 0
        if i_coarse + i_offset == 0 or i_coarse + i_offset == nx_coarse-1:
            boundaries += 1
        if j_coarse + j_offset == 0 or j_coarse + j_offset == ny_coarse-1:
            boundaries += 1

        total_weight_target = 2.25 if boundaries == 2 else (3.0 if boundaries == 1 else 4.0)

        for k in range(dof):
            start = Cp[idx_coarse*dof + k]
            for kk in range(dof):
                Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk

        vals = np.zeros(dof*dof, dtype=K_flat.dtype)

        for di in range(-1, 2):
            i_neighbor = i_fine + di
            if i_neighbor < 0 or i_neighbor >= nx_fine:
                continue

            for dj in range(-1, 2):
                j_neighbor = j_fine + dj
                if j_neighbor < 0 or j_neighbor >= ny_fine:
                    continue

                distance = abs(di) + abs(dj)
                if distance > 2:
                    continue

                weight = 1.0 if distance == 0 else (0.5 if distance == 1 else 0.25)
                idx_fine = i_neighbor + nx_fine * j_neighbor

                vals.fill(0.0)
                get_target_vals_2d(vals, K_flat, idx_fine, dof, elements_size, el_ids, node_ids, sorter, elements_flat, weights, elem_flat_size, n_nodes, con_map, i_fine_target, j_fine_target, nx_fine, ny_fine)

                for k in range(dof):
                    start = Cp[idx_coarse*dof + k]
                    for kk in range(dof):
                        Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target

@njit(["void(f4[:], i4, i4[:], i4[:], f4[:], i4, i4, i4, i4, i4, i4, i4)",
       "void(f8[:], i4, i4[:], i4[:], f8[:], i4, i4, i4, i4, i4, i4, i4)"], cache=True)
def get_target_vals_l1p_3d(vals, idx_fine, Ap, Aj, Ax, i_fine_target, j_fine_target, k_fine_target, nx_fine, ny_fine, nz_fine, dof):
    for k in range(dof):
        start = Ap[idx_fine*dof + k]
        end = Ap[idx_fine*dof + k + 1]
        for j in range(start, end):
            fine_node = Aj[j]//dof
            local_dof = Aj[j]%dof
            
            k_fine = fine_node % nz_fine
            tmp = fine_node // nz_fine
            i_fine = tmp % nx_fine
            j_fine = tmp // nx_fine
            
            di = i_fine - i_fine_target
            dj = j_fine - j_fine_target
            dk = k_fine - k_fine_target
            
            if di < -1 or di > 1:
                continue
            if dj < -1 or dj > 1:
                continue
            if dk < -1 or dk > 1:
                continue
                
            distance = abs(di) + abs(dj) + abs(dk)
            if distance > 3:
                continue
                
            weight = 1.0 if distance == 0 else (0.5 if distance == 1 else (0.25 if distance == 2 else 0.125))
            vals[k*dof+local_dof] += Ax[j]*weight

@njit(["void(f4[:], i4[:], i4[:], i4, i4, i4, i4, i4, i4, i4[:], i4[:], f4[:], i4)",
       "void(f8[:], i4[:], i4[:], i4, i4, i4, i4, i4, i4, i4[:], i4[:], f8[:], i4)"], cache=True, parallel=True)
def get_restricted_3d_l1p_nnz_based(Ax, Aj, Ap, nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse, Cp, Cj, Cx, dof):
    for thread in prange(nx_coarse * ny_coarse * nz_coarse * 27):
        idx_coarse = thread // 27
        neighbour_idx = thread % 27
        
        k_coarse = idx_coarse % nz_coarse
        tmp = idx_coarse // nz_coarse
        i_coarse = tmp % nx_coarse
        j_coarse = tmp // nx_coarse
        
        i_offset = neighbour_idx % 3 - 1
        j_offset = (neighbour_idx // 3) % 3 - 1
        k_offset = neighbour_idx // 9 - 1
        
        if i_coarse + i_offset < 0 or i_coarse + i_offset >= nx_coarse:
            continue
        if j_coarse + j_offset < 0 or j_coarse + j_offset >= ny_coarse:
            continue
        if k_coarse + k_offset < 0 or k_coarse + k_offset >= nz_coarse:
            continue
            
        target_coarse = k_coarse + k_offset + nz_coarse * (i_coarse + i_offset) + nz_coarse * nx_coarse * (j_coarse + j_offset)
        
        i_fine_target = (i_coarse + i_offset) * 2
        j_fine_target = (j_coarse + j_offset) * 2
        k_fine_target = (k_coarse + k_offset) * 2
        
        i_fine = i_coarse * 2
        j_fine = j_coarse * 2
        k_fine = k_coarse * 2
        
        boundaries = 0
        if i_coarse == 0 or i_coarse == nx_coarse-1:
            boundaries += 1
        if j_coarse == 0 or j_coarse == ny_coarse-1:
            boundaries += 1
        if k_coarse == 0 or k_coarse == nz_coarse-1:
            boundaries += 1
            
        total_weight = 3.375 if boundaries == 3 else (4.5 if boundaries == 2 else (6.0 if boundaries == 1 else 8.0))
        
        boundaries = 0
        if i_coarse + i_offset == 0 or i_coarse + i_offset == nx_coarse-1:
            boundaries += 1
        if j_coarse + j_offset == 0 or j_coarse + j_offset == ny_coarse-1:
            boundaries += 1
        if k_coarse + k_offset == 0 or k_coarse + k_offset == nz_coarse-1:
            boundaries += 1
            
        total_weight_target = 3.375 if boundaries == 3 else (4.5 if boundaries == 2 else (6.0 if boundaries == 1 else 8.0))
        
        for k in range(dof):
            start = Cp[idx_coarse*dof + k]
            for kk in range(dof):
                Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk
                
        vals = np.zeros(dof*dof, dtype=Ax.dtype)
        
        for di in range(-1, 2):
            i_neighbor = i_fine + di
            if i_neighbor < 0 or i_neighbor >= nx_fine:
                continue
                
            for dj in range(-1, 2):
                j_neighbor = j_fine + dj
                if j_neighbor < 0 or j_neighbor >= ny_fine:
                    continue
                    
                for dk in range(-1, 2):
                    k_neighbor = k_fine + dk
                    if k_neighbor < 0 or k_neighbor >= nz_fine:
                        continue
                        
                    distance = abs(di) + abs(dj) + abs(dk)
                    if distance > 3:
                        continue
                        
                    weight = 1.0 if distance == 0 else (0.5 if distance == 1 else (0.25 if distance == 2 else 0.125))
                    idx_fine = k_neighbor + nz_fine * i_neighbor + nz_fine * nx_fine * j_neighbor
                    
                    vals.fill(0.0)
                    get_target_vals_l1p_3d(vals, idx_fine, Ap, Aj, Ax, i_fine_target, j_fine_target, k_fine_target, nx_fine, ny_fine, nz_fine, dof)
                    
                    for k in range(dof):
                        start = Cp[idx_coarse*dof + k]
                        for kk in range(dof):
                            Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target

@njit(["void(f4[:], i4, i4[:], i4[:], f4[:], i4, i4, i4, i4, i4)",
       "void(f8[:], i4, i4[:], i4[:], f8[:], i4, i4, i4, i4, i4)"], cache=True)
def get_target_vals_l1p_2d(vals, idx_fine, Ap, Aj, Ax, i_fine_target, j_fine_target, nx_fine, ny_fine, dof):
    for k in range(dof):
        start = Ap[idx_fine*dof + k]
        end = Ap[idx_fine*dof + k + 1]
        for j in range(start, end):
            fine_node = Aj[j]//dof
            local_dof = Aj[j]%dof
            
            i_fine = fine_node % nx_fine
            j_fine = fine_node // nx_fine

            di = i_fine - i_fine_target
            dj = j_fine - j_fine_target
            
            if di < -1 or di > 1:
                continue
            if dj < -1 or dj > 1:
                continue
                
            distance = abs(di) + abs(dj)
            if distance > 2:
                continue
                
            weight = 1.0 if distance == 0 else (0.5 if distance == 1 else 0.25)
            vals[k*dof+local_dof] += Ax[j]*weight

@njit(["void(f4[:], i4[:], i4[:], i4, i4, i4, i4, i4[:], i4[:], f4[:], i4)",
       "void(f8[:], i4[:], i4[:], i4, i4, i4, i4, i4[:], i4[:], f8[:], i4)"], cache=True, parallel=True)
def get_restricted_2d_l1p_nnz_based(Ax, Aj, Ap, nx_fine, ny_fine, nx_coarse, ny_coarse, Cp, Cj, Cx, dof):
    for thread in prange(nx_coarse * ny_coarse * 9):
        idx_coarse = thread // 9
        neighbour_idx = thread % 9
        
        i_coarse = idx_coarse % nx_coarse
        j_coarse = idx_coarse // nx_coarse
        
        i_offset = neighbour_idx % 3 - 1
        j_offset = neighbour_idx // 3 - 1
        
        if i_coarse + i_offset < 0 or i_coarse + i_offset >= nx_coarse:
            continue
        if j_coarse + j_offset < 0 or j_coarse + j_offset >= ny_coarse:
            continue
            
        target_coarse = i_coarse + i_offset + nx_coarse * (j_coarse + j_offset)
        
        i_fine_target = (i_coarse + i_offset) * 2
        j_fine_target = (j_coarse + j_offset) * 2
        
        i_fine = i_coarse * 2
        j_fine = j_coarse * 2
        
        boundaries = 0
        if i_coarse == 0 or i_coarse == nx_coarse-1:
            boundaries += 1
        if j_coarse == 0 or j_coarse == ny_coarse-1:
            boundaries += 1
        
        total_weight = 2.25 if boundaries == 2 else (3.0 if boundaries == 1 else 4.0)
        
        boundaries = 0
        if i_coarse + i_offset == 0 or i_coarse + i_offset == nx_coarse-1:
            boundaries += 1
        if j_coarse + j_offset == 0 or j_coarse + j_offset == ny_coarse-1:
            boundaries += 1
            
        total_weight_target = 2.25 if boundaries == 2 else (3.0 if boundaries == 1 else 4.0)
        
        for k in range(dof):
            start = Cp[idx_coarse*dof + k]
            for kk in range(dof):
                Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk
                
        vals = np.zeros(dof*dof, dtype=Ax.dtype)
        
        for di in range(-1, 2):
            i_neighbor = i_fine + di
            if i_neighbor < 0 or i_neighbor >= nx_fine:
                continue
                
            for dj in range(-1, 2):
                j_neighbor = j_fine + dj
                if j_neighbor < 0 or j_neighbor >= ny_fine:
                    continue
                    
                distance = abs(di) + abs(dj)
                if distance > 2:
                    continue
                    
                weight = 1.0 if distance == 0 else (0.5 if distance == 1 else 0.25)
                idx_fine = i_neighbor + nx_fine * j_neighbor
                
                vals.fill(0.0)
                get_target_vals_l1p_2d(vals, idx_fine, Ap, Aj, Ax, i_fine_target, j_fine_target, nx_fine, ny_fine, dof)
                
                for k in range(dof):
                    start = Cp[idx_coarse*dof + k]
                    for kk in range(dof):
                        Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target
                        
def get_restricted_l0(mesh: Union[StructuredMesh2D, StructuredMesh3D], kernel: StructuredStiffnessKernel, Cp = None):
    dim = mesh.nodes.shape[1]

    if dim == 2:
        nx_fine = mesh.nelx+1
        ny_fine = mesh.nely+1
        nx_coarse = mesh.nelx // 2 + 1
        ny_coarse = mesh.nely // 2 + 1
        dof = mesh.dof
        node_ids = kernel.node_ids
        el_ids = kernel.el_ids
        sorter = kernel.sorter
        elements_flat = kernel.elements_flat
        elements_size = kernel.elements_size
        con_map = kernel.constraints
        elem_flat_size = elements_flat.shape[0]
        n_nodes = kernel.n_nodes

        if Cp is None:
            nnz = np.ones(nx_coarse*ny_coarse*dof, dtype=np.int32) * 18
            Cp = np.zeros(nx_coarse*ny_coarse*dof + 1, dtype=np.int32)
            Cp[1:] = np.cumsum(nnz)

        K_flat = kernel.K_single.flatten()
        Cj = np.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = np.zeros(int(Cp[-1]), dtype=K_flat.dtype)
        weights = kernel.rho
        
        get_restricted_2d_l0_nnz_based(K_flat, nx_fine, ny_fine, nx_coarse, ny_coarse, Cp, Cj, Cx, dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes)
    
    else:
        nx_fine = mesh.nelx+1
        ny_fine = mesh.nely+1
        nz_fine = mesh.nelz+1
        nx_coarse = mesh.nelx // 2 + 1
        ny_coarse = mesh.nely // 2 + 1
        nz_coarse = mesh.nelz // 2 + 1
        dof = mesh.dof
        node_ids = kernel.node_ids
        el_ids = kernel.el_ids
        sorter = kernel.sorter
        elements_flat = kernel.elements_flat
        elements_size = kernel.elements_size
        con_map = kernel.constraints
        elem_flat_size = elements_flat.shape[0]
        n_nodes = kernel.n_nodes

        if Cp is None:
            nnz = np.ones(nx_coarse*ny_coarse*nz_coarse*dof, dtype=np.int32) * 81
            Cp = np.zeros(nx_coarse*ny_coarse*nz_coarse*dof + 1, dtype=np.int32)
            Cp[1:] = np.cumsum(nnz)

        K_flat = kernel.K_single.flatten()
        Cj = np.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = np.zeros(int(Cp[-1]), dtype=K_flat.dtype)
        weights = kernel.rho
        
        get_restricted_3d_l0_nnz_based(K_flat, nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse, Cp, Cj, Cx, dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes)
    
    return csr_matrix((Cx, Cj, Cp), shape=(Cp.shape[0]-1, Cp.shape[0]-1))

def get_restricted_l1p(A: csr_matrix, nel, dof, Cp = None):
    dim = len(nel)
    if dim == 2:
        nx_fine = int(nel[0] + 1)
        ny_fine = int(nel[1] + 1)
        nx_coarse = int(nel[0] // 2 + 1)
        ny_coarse = int(nel[1] // 2 + 1)
        
        if Cp is None:
            nnz = np.ones(nx_coarse*ny_coarse*dof, dtype=np.int32) * 18
            Cp = np.zeros(nx_coarse*ny_coarse*dof + 1, dtype=np.int32)
            Cp[1:] = np.cumsum(nnz)
        
        Cj = np.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = np.zeros(int(Cp[-1]), dtype=A.dtype)
        
        get_restricted_2d_l1p_nnz_based(A.data, A.indices, A.indptr, nx_fine, ny_fine, nx_coarse, ny_coarse, Cp, Cj, Cx, dof)
                
    else:
        nx_fine = int(nel[0] + 1)
        ny_fine = int(nel[1] + 1)
        nz_fine = int(nel[2] + 1)
        nx_coarse = int(nel[0] // 2 + 1)
        ny_coarse = int(nel[1] // 2 + 1)
        nz_coarse = int(nel[2] // 2 + 1)
        
        if Cp is None:
            nnz = np.ones(nx_coarse*ny_coarse*nz_coarse*dof, dtype=np.int32) * 81
            Cp = np.zeros(nx_coarse*ny_coarse*nz_coarse*dof + 1, dtype=np.int32)
            Cp[1:] = np.cumsum(nnz)

        Cj = np.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = np.zeros(int(Cp[-1]), dtype=A.dtype)

        get_restricted_3d_l1p_nnz_based(A.data, A.indices, A.indptr, nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse, Cp, Cj, Cx, dof)
                
    return csr_matrix((Cx, Cj, Cp), shape=(Cp.shape[0]-1, Cp.shape[0]-1))