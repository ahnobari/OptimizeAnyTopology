from numba import njit, prange, jit
from scipy.sparse import csr_matrix
import numpy as np

@njit(["f4[:](f4[:, :], i4[:, :], f4[:], i4, i4)", "f8[:](f8[:, :], i4[:, :], f8[:], i4, i4)"], cache=True, parallel=True)
def process_dk(K_single, elements, U, dof=3, elements_size=8):
    out = np.zeros(elements.shape[0],dtype=K_single.dtype)
    
    for i in prange(elements.shape[0]):
        dof_id = elements[i].repeat(dof)*dof + np.array([list(range(dof))]*elements_size).flatten()
        out[i] = -U[dof_id].dot(K_single).dot(U[dof_id])
          
    return out

@njit(["f4[:](f4[:, :, :], i4[:, :], f4[:], i4, i4)", "f8[:](f8[:, :, :], i4[:, :], f8[:], i4, i4)"], cache=True, parallel=True)
def process_dk_full(Ks, elements, U, dof=3, elements_size=8):
    out = np.zeros(elements.shape[0],dtype=Ks.dtype)
    
    for i in prange(elements.shape[0]):
        dof_id = elements[i].repeat(dof)*dof + np.array([list(range(dof))]*elements_size).flatten()
        out[i] = -U[dof_id].dot(Ks[i]).dot(U[dof_id])
          
    return out

@njit(["f4[:](f4[:], i4[:], i4[:], i4[:], f4[:], i4)", "f8[:](f8[:], i4[:], i4[:], i4[:], f8[:], i4)"], cache=True, parallel=True)
def process_dk_flat(K_flat, elements_flat, K_ptr, elements_ptr, U, dof=3):
    out = np.zeros(elements_ptr.shape[0]-1,dtype=K_flat.dtype)
    
    for i in prange(elements_ptr.shape[0]-1):
        start = elements_ptr[i]
        end = elements_ptr[i+1]
        size = end-start
        elem_map = elements_flat[start:end]
        # dof_id = elem_map.repeat(dof)*dof + np.array([list(range(dof))]*size).flatten()
        K = K_flat[K_ptr[i]:K_ptr[i+1]]
        
        for j in range(dof*size):
            val = 0
            for k in range(dof*size):
                val += K[j*dof*size + k] * U[elem_map[k//dof]*dof + k % dof]
            out[i] -= val * U[elem_map[j//dof]*dof + j % dof]
        
    return out

@njit(["f4[:](f4[:, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4, bool_[:])",
       "f8[:](f8[:, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4, bool_[:])"], cache=True, parallel=True)
def get_diagonal_node_basis(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, cons_map):
    diag = np.zeros(n_nodes*dof, dtype=K_single.dtype)
    
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            for k in range(dof):
                if cons_map[i*dof+k]:
                    diag[i*dof+k] = 1
                    continue
                for l in range(elements_size*dof):
                    if elements_map[l//dof]*dof+l%dof == i*dof+k:
                        diag[i*dof+k] += K_single[relative_dof*dof+k,l] * weight
    return diag

@njit(["f4[:](f4[:, :, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4, bool_[:])",
       "f8[:](f8[:, :, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4, bool_[:])"], cache=True, parallel=True)
def get_diagonal_node_basis_full(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, cons_map):
    diag = np.zeros(n_nodes*dof, dtype=Ks.dtype)
    
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            for k in range(dof):
                if cons_map[i*dof+k]:
                    diag[i*dof+k] = 1
                    continue
                for l in range(elements_size*dof):
                    if elements_map[l//dof]*dof+l%dof == i*dof+k:
                        diag[i*dof+k] += Ks[elements_ids_j,relative_dof*dof+k,l] * weight
            
    return diag

@njit(["f4[:](f4[:], i4[:], i4[:], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, bool_[:])",
       "f8[:](f8[:], i4[:], i4[:], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, bool_[:])"], cache=True, parallel=True)
def get_diagonal_node_basis_flat(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, cons_map):
    diag = np.zeros(n_nodes*dof, dtype=K_flat.dtype)
    
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            
            relative_dof = e_ind_j - start
            k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof
            
            for k in range(dof):
                if cons_map[i*dof+k]:
                    diag[i*dof+k] = 1
                    continue
                for l in range(size*dof):
                    if elements_map[l//dof]*dof+l%dof == i*dof+k:
                        diag[i*dof+k] += K_flat[k_start + k*size*dof + l] * weight
    return diag
    
@njit(["f4[:](f4[:], i4[:], i4[:], i4[:], i4[:], f4[:], i4[:], i4[:], i4, f4[:], i4)",
       "f8[:](f8[:], i4[:], i4[:], i4[:], i4[:], f8[:], i4[:], i4[:], i4, f8[:], i4)"], cache=True, parallel=True)
def mat_vec_node_basis_parallel_flat(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, vec, dof=3):
    out = np.zeros_like(vec, dtype=K_flat.dtype)
    
    for i in prange(n_nodes):
        if i<n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            
            relative_dof = e_ind_j - start
            k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof
            
            for k in range(dof):
                for l in range(size*dof):
                    out[i*dof+k] += K_flat[k_start + k*size*dof + l] * vec[elements_map[l//dof]*dof+l%dof] * weight
        
    return out

@njit(["void(f4[:, :, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, f4[:], f4[:], i4, i4)",
       "void(f8[:, :, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, f8[:], f8[:], i4, i4)"], cache=True, parallel=True)
def mat_vec_node_basis_parallel_full(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, out, dof=3, elements_size=8):
    # out = np.zeros_like(vec, dtype=Ks.dtype)
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        for k in range(dof):
            out[i*dof+k] = 0.0
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            for k in range(dof):
                val = 0.0
                for l in range(elements_size*dof):
                    # out[i*dof+k] += Ks[elements_ids_j,relative_dof*dof+k,l] * vec[elements_map[l//dof]*dof+l%dof] * weight
                    val += Ks[elements_ids_j,relative_dof*dof+k,l] * vec[elements_map[l//dof]*dof+l%dof]
                out[i*dof+k] += val * weight
            

@njit(["void(f4[:, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, f4[:], f8[:], i4, i4)",
       "void(f8[:, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, f8[:], f8[:], i4, i4)"], cache=True, parallel=True)
def mat_vec_node_basis_parallel(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, out, dof=3, elements_size=8):
    # out = np.zeros_like(vec, dtype=K_single.dtype)
    
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        
        for k in range(dof):
            out[i*dof+k] = 0.0
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            for k in range(dof):
                val = 0.0
                for l in range(elements_size*dof):
                    # out[i*dof+k] += K_single[relative_dof*dof+k,l] * vec[elements_map[l//dof]*dof+l%dof] * weight
                    val += K_single[relative_dof*dof+k,l] * vec[elements_map[l//dof]*dof+l%dof]
                out[i*dof+k] += val * weight


@njit(["f4[:](f4[:], i4[:], i4[:], i4[:], i4[:], f4[:], i4[:], i4[:], i4, f4[:], bool_[:], i4)",
       "f8[:](f8[:], i4[:], i4[:], i4[:], i4[:], f8[:], i4[:], i4[:], i4, f8[:], bool_[:], i4)"], cache=True, parallel=True)
def mat_vec_node_basis_parallel_flat_wcon(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, vec, con_map, dof=3):
    out = np.zeros_like(vec, dtype=K_flat.dtype)
    
    for i in prange(n_nodes):
        if i<n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            
            relative_dof = e_ind_j - start
            k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof
            
            for k in range(dof):
                if con_map[i*dof+k]:
                    continue
                for l in range(size*dof):
                    kk = elements_map[l//dof]*dof+l%dof
                    if con_map[kk]:
                        continue
                    out[i*dof+k] += K_flat[k_start + k*size*dof + l] * vec[kk] * weight
        
        for k in range(dof):
            if con_map[i*dof+k]:
                out[i*dof+k] = vec[i*dof+k]
    return out

@njit(["void(f4[:, :, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, f4[:], bool_[:], f4[:], i4, i4)",
       "void(f8[:, :, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, f8[:], bool_[:], f8[:], i4, i4)"], cache=True, parallel=True)
def mat_vec_node_basis_parallel_full_wcon(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, con_map, out, dof=3, elements_size=8):
    # out = np.zeros_like(vec, dtype=Ks.dtype)
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        
        for k in range(dof):
            if con_map[i*dof+k]:
                out[i*dof+k] = vec[i*dof+k]
            else:
                out[i*dof+k] = 0.0
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            for k in range(dof):
                if con_map[i*dof+k]:
                    continue
                val = 0.0
                for l in range(elements_size*dof):
                    kk = elements_map[l//dof]*dof+l%dof
                    if con_map[kk]:
                        continue
                    # out[i*dof+k] += Ks[elements_ids_j,relative_dof*dof+k,l] * vec[kk] * weight
                    val += Ks[elements_ids_j,relative_dof*dof+k,l] * vec[kk]
                out[i*dof+k] += val * weight


@njit(["void(f4[:, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, f4[:], bool_[:], f4[:], i4, i4)",
       "void(f8[:, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, f8[:], bool_[:], f8[:], i4, i4)"], cache=True, parallel=True)
def mat_vec_node_basis_parallel_wcon(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, con_map, out, dof=3, elements_size=8):
    # out = np.zeros_like(vec, dtype=K_single.dtype)
    
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        for k in range(dof):
            if con_map[i*dof+k]:
                out[i*dof+k] = vec[i*dof+k]
            else:
                out[i*dof+k] = 0.0
        
        n_elements = en-st
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            for k in range(dof):
                val = 0.0
                if con_map[i*dof+k]:
                    continue
                for l in range(elements_size*dof):
                    kk = elements_map[l//dof]*dof+l%dof
                    if con_map[kk]:
                        continue
                    val += K_single[relative_dof*dof+k,l] * vec[kk]
                out[i*dof+k] += val * weight

@njit("i4[:](i4[:], i4[:], i4[:], i4[:], i4, i4, i4, i4, i4[:], i4[:])")
def matmat_node_basis_nnz_per_row(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj):
    nnz_per_row = np.zeros(n_nodes*dof, dtype=np.int32)
    mask = np.zeros(n_col, dtype=np.int32)-1
    for i in range(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            for l in range(elements_size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    if mask[k] != i:
                        mask[k] = i
                        nnz_per_row[i*dof:(i+1)*dof] += 1
    return nnz_per_row

@njit("i4[:](i4[:], i4[:], i4[:], i4[:], i4, i4, i4, i4, i4[:], i4[:], bool_[:])")
def matmat_node_basis_nnz_per_row_wcon(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, con_map):
    nnz_per_row = np.zeros(n_nodes*dof, dtype=np.int32)
    mask = np.zeros(n_col, dtype=np.int32)-1
    for i in range(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            for l in range(elements_size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                
                if con_map[jj]:
                    continue
                
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    if mask[k] != i:
                        mask[k] = i
                        nnz_per_row[i*dof:(i+1)*dof] += 1
                        
        for d in range(dof):
            if con_map[i*dof+d]:
                nnz_per_row[i*dof+d] = Bp[i*dof+d+1] - Bp[i*dof+d]
    
    return nnz_per_row

@njit("i4[:](i4[:], i4[:], i4[:], i4[:], i4, i4, i4, i4, i4[:], i4[:], i4)", cache=True, parallel=True)
def matmat_node_basis_nnz_per_row_parallel(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz):
    nnz_per_row = np.zeros(n_nodes*dof, dtype=np.int32)
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        rolling_list = np.zeros(max_nnz, dtype=np.int32)-1
        n_elements = en-st
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            for l in range(elements_size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for ll in range(max_nnz):
                        if rolling_list[ll] == k:
                            break
                        elif rolling_list[ll] == -1:
                            rolling_list[ll] = k
                            nnz_per_row[i*dof:(i+1)*dof] += 1
                            break
                        if ll == max_nnz-1:
                            print("Error: max_nnz reached ... use a larger max_nnz")
    return nnz_per_row

@njit(["void(f4[:, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f4[:], i4[:], i4[:], f4[:])",
       "void(f8[:, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:])"], cache=True, parallel=True)
def matmat_node_basis_prallel_kernel(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx):
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        
        start_ = Cp[i*dof]
        end_ = Cp[i*dof+1]
        length = end_ - start_
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            
            for l in range(elements_size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                start__ = Bp[jj]
                end__ = Bp[jj+1]
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for ll in range(start_, end_):
                        if Cj[ll] == k:
                            for d in range(dof):
                                Cx[ll + length * d] += K_single[relative_dof*dof+d,l] * Bx[kk] * weight
                            break
                        elif Cj[ll] == -1:
                            for d in range(dof):
                                Cj[ll + length * d] = k
                                Cx[ll + length * d] += K_single[relative_dof*dof+d,l] * Bx[kk] * weight
                            break

@njit(["void(f4[:, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f4[:], i4[:], i4[:], f4[:], bool_[:])",
       "void(f8[:, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:], bool_[:])"], cache=True, parallel=True)
def matmat_node_basis_prallel_kernel_wcon(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, con_map):
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            
            for l in range(elements_size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                if con_map[jj]:
                    continue
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for d in range(dof):
                        if con_map[i*dof+d]:
                            continue
                        start_ = Cp[i*dof+d]
                        end_ = Cp[i*dof+d+1]
                        for ll in range(start_, end_):
                            if Cj[ll] == k:
                                Cx[ll] += K_single[relative_dof*dof+d,l] * Bx[kk] * weight
                                break
                            elif Cj[ll] == -1:
                                Cj[ll] = k
                                Cx[ll] += K_single[relative_dof*dof+d,l] * Bx[kk] * weight
                                break
        
        for d in range(dof):
            if con_map[i*dof+d]:
                jj = i*dof+d
                count = 0
                start_ = Cp[jj]
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    Cj[start_ + count] = k
                    Cx[start_ + count] = Bx[kk]
                    count += 1

def matmat_node_basis_prallel(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, B, parallel=False, max_source_nnz=None, cons=None):
    n_col = B.shape[1]
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    if parallel:
        if max_source_nnz is None:
            max_nnz = (np.diff(Bp).max() * (elements_size * dof) * np.unique(elements_flat, return_counts=True)[1].max())
        else:
            max_nnz = max_source_nnz *  np.diff(Bp).max()
        
        if cons is not None:
            nnz_per_row = matmat_node_basis_nnz_per_row_wcon(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, cons)
        else:
            nnz_per_row = matmat_node_basis_nnz_per_row_parallel(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz)
    else:
        nnz_per_row = matmat_node_basis_nnz_per_row(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj)
    Cp = np.zeros(n_nodes*dof+1, dtype=np.int32)
    Cp[1:] = np.cumsum(nnz_per_row)
    Cj = -np.ones(int(Cp[-1]), dtype=np.int32)
    Cx = np.zeros(int(Cp[-1]), dtype=K_single.dtype)
    
    if cons is None:
        matmat_node_basis_prallel_kernel(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx)
    else:
        matmat_node_basis_prallel_kernel_wcon(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, cons)
    
    M = csr_matrix((n_nodes*dof, n_col), dtype=K_single.dtype)
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M


def matmat_node_basis_prallel_(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, B, Cp, cons=None):
    n_col = B.shape[1]
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    Cj = -np.ones(int(Cp[-1]), dtype=np.int32)
    Cx = np.zeros(int(Cp[-1]), dtype=K_single.dtype)
    if cons is None:
        matmat_node_basis_prallel_kernel(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx)
    else:
        matmat_node_basis_prallel_kernel_wcon(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, cons)
    M = csr_matrix((n_nodes*dof, n_col), dtype=K_single.dtype)
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

@njit(["void(f4[:, :, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f4[:], i4[:], i4[:], f4[:])",
       "void(f8[:, :, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:])"], cache=True, parallel=True)
def matmat_node_basis_full_prallel_kernel(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx):
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        
        start_ = Cp[i*dof]
        end_ = Cp[i*dof+1]
        length = end_ - start_
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            
            for l in range(elements_size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for ll in range(start_, end_):
                        if Cj[ll] == k:
                            for d in range(dof):
                                Cx[ll + length * d] += Ks[elements_ids_j,relative_dof*dof+d,l] * Bx[kk] * weight
                            break
                        elif Cj[ll] == -1:
                            for d in range(dof):
                                Cj[ll + length * d] = k
                                Cx[ll + length * d] += Ks[elements_ids_j,relative_dof*dof+d,l] * Bx[kk] * weight
                            break
                        
@njit(["void(f4[:, :, :], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f4[:], i4[:], i4[:], f4[:], bool_[:])",
       "void(f8[:, :, :], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:], bool_[:])"], cache=True, parallel=True)
def matmat_node_basis_full_prallel_kernel_wcon(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, con_map):
    for i in prange(n_nodes):
        if i < n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)

        n_elements = en-st
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ids_j * elements_size
            end = start + elements_size
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            relative_dof = e_ind_j - start
            
            
            for l in range(elements_size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                if con_map[jj]:
                    continue
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for d in range(dof):
                        if con_map[i*dof+d]:
                            continue
                            
                        start_ = Cp[i*dof+d]
                        end_ = Cp[i*dof+d+1]
                        for ll in range(start_, end_):
                            if Cj[ll] == k:
                                Cx[ll] += Ks[elements_ids_j,relative_dof*dof+d,l] * Bx[kk] * weight
                                break
                            elif Cj[ll] == -1:
                                Cj[ll] = k
                                Cx[ll] += Ks[elements_ids_j,relative_dof*dof+d,l] * Bx[kk] * weight
                                break
        
        for d in range(dof):
            if con_map[i*dof+d]:
                jj = i*dof+d
                count = 0
                start_ = Cp[jj]
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    Cj[start_ + count] = k
                    Cx[start_ + count] = Bx[kk]
                    count += 1
                    
def matmat_node_basis_full_prallel(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, B, parallel=False, max_source_nnz=None, cons=None):
    n_col = B.shape[1]
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    if parallel:
        if max_source_nnz is None:
            max_nnz = (np.diff(Bp).max() * (elements_size * dof) * np.unique(elements_flat, return_counts=True)[1].max())
        else:
            max_nnz = max_source_nnz *  np.diff(Bp).max()
            
        if cons is not None:
            nnz_per_row = matmat_node_basis_nnz_per_row_wcon(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, cons)
        else:
            nnz_per_row = matmat_node_basis_nnz_per_row_parallel(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz)
    else:
        nnz_per_row = matmat_node_basis_nnz_per_row(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj)
    Cp = np.zeros(n_nodes*dof+1, dtype=np.int32)
    Cp[1:] = np.cumsum(nnz_per_row)
    Cj = -np.ones(int(Cp[-1]), dtype=np.int32)
    Cx = np.zeros(int(Cp[-1]), dtype=Ks.dtype)
    
    if cons is None:
        matmat_node_basis_full_prallel_kernel(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx)
    else:
        matmat_node_basis_full_prallel_kernel_wcon(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, cons)
    M = csr_matrix((n_nodes*dof, n_col), dtype=Ks.dtype)
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

def matmat_node_basis_full_prallel_(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, B, Cp, cons=None):
    n_col = B.shape[1]
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    Cj = -np.ones(int(Cp[-1]), dtype=np.int32)
    Cx = np.zeros(int(Cp[-1]), dtype=Ks.dtype)
    if cons is None:
        matmat_node_basis_full_prallel_kernel(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx)
    else:
        matmat_node_basis_full_prallel_kernel_wcon(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, cons)
        
    M = csr_matrix((n_nodes*dof, n_col), dtype=Ks.dtype)
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

@njit("i4[:](i4[:], i4[:], i4[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:])")
def matmat_node_basis_flat_nnz_per_row(elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes, dof, n_col, Bp, Bj):
    nnz_per_row = np.zeros(n_nodes*dof, dtype=np.int32)
    mask = np.zeros(n_col, dtype=np.int32)-1
    for i in range(n_nodes):
        if i<n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            
            for l in range(size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    if mask[k] != i:
                        mask[k] = i
                        nnz_per_row[i*dof:(i+1)*dof] += 1
    return nnz_per_row

@njit("i4[:](i4[:], i4[:], i4[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:],i4)", cache=True, parallel=True)
def matmat_node_basis_flat_nnz_per_row_parallel(elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes, dof, n_col, Bp, Bj, max_nnz):
    nnz_per_row = np.zeros(n_nodes*dof, dtype=np.int32)
    for i in prange(n_nodes):
        if i<n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        rolling_list = np.zeros(max_nnz, dtype=np.int32)-1
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            
            for l in range(size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for ll in range(max_nnz):
                        if rolling_list[ll] == k:
                            break
                        elif rolling_list[ll] == -1:
                            rolling_list[ll] = k
                            nnz_per_row[i*dof:(i+1)*dof] += 1
                            break
                        if ll == max_nnz-1:
                            print("Error: max_nnz reached ... use a larger max_nnz")
    return nnz_per_row

@njit("i4[:](i4[:], i4[:], i4[:], i4[:], i4[:], i4, i4, i4, i4[:], i4[:],i4, bool_[:])", cache=True, parallel=True)
def matmat_node_basis_flat_nnz_per_row_parallel_wcon(elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes, dof, n_col, Bp, Bj, max_nnz, con_map):
    nnz_per_row = np.zeros(n_nodes*dof, dtype=np.int32)
    for i in prange(n_nodes):
        if i<n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        rolling_list = np.zeros(max_nnz, dtype=np.int32)-1
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            
            for l in range(size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                
                if con_map[jj]:
                    continue
                
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for ll in range(max_nnz):
                        if rolling_list[ll] == k:
                            break
                        elif rolling_list[ll] == -1:
                            rolling_list[ll] = k
                            nnz_per_row[i*dof:(i+1)*dof] += 1
                            break
                        if ll == max_nnz-1:
                            print("Error: max_nnz reached ... use a larger max_nnz")
                            
        for d in range(dof):
            if con_map[i*dof+d]:
                nnz_per_row[i*dof+d] = Bp[i*dof+d+1] - Bp[i*dof+d]
    return nnz_per_row

@njit(["void(f4[:], i4[:], i4[:], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4[:], i4[:], f4[:], i4[:], i4[:], f4[:])",
       "void(f8[:], i4[:], i4[:], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:])"], cache=True, parallel=True)
def matmat_node_basis_flat_prallel_kernel(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx):
    for i in prange(n_nodes):
        if i<n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        start_ = Cp[i*dof]
        end_ = Cp[i*dof+1]
        length = end_ - start_
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            
            relative_dof = e_ind_j - start
            k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof
            
            for l in range(size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for ll in range(start_, end_):
                        if Cj[ll] == k:
                            for d in range(dof):
                                Cx[ll + length * d] += K_flat[k_start + d*size*dof + l] * Bx[kk] * weight
                            break
                        elif Cj[ll] == -1:
                            for d in range(dof):
                                Cj[ll + length * d] = k
                                Cx[ll + length * d] += K_flat[k_start + d*size*dof + l] * Bx[kk] * weight
                            break

@njit(["void(f4[:], i4[:], i4[:], i4[:], i4[:], f4[:], i4[:], i4[:], i4, i4, i4[:], i4[:], f4[:], i4[:], i4[:], f4[:], bool_[:])",
       "void(f8[:], i4[:], i4[:], i4[:], i4[:], f8[:], i4[:], i4[:], i4, i4, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:], bool_[:])"], cache=True, parallel=True)
def matmat_node_basis_flat_prallel_kernel_wcon(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx, con_map):
    for i in prange(n_nodes):
        if i<n_nodes-1:
            st = node_ids[i]
            en = node_ids[i+1]
        else:
            st = node_ids[i]
            en = len(elements_flat)
        
        n_elements = en-st
        
        for j in range(n_elements):
            e_ind_j = sorter[st+j]
            elements_ids_j = el_ids[e_ind_j]
            start = elements_ptr[elements_ids_j]
            end = elements_ptr[elements_ids_j+1]
            size = end - start
            elements_map = elements_flat[start:end]
            weight = weights[elements_ids_j]
            
            relative_dof = e_ind_j - start
            k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof
            
            for l in range(size*dof):
                jj = elements_map[l//dof]*dof+l%dof
                
                jj = elements_map[l//dof]*dof+l%dof
                if con_map[jj]:
                    continue
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    for d in range(dof):
                        if con_map[i*dof+d]:
                            continue
                        
                        start_ = Cp[i*dof+d]
                        end_ = Cp[i*dof+d+1]
                        for ll in range(start_, end_):
                            if Cj[ll] == k:
                                Cx[ll] += K_flat[k_start + d*size*dof + l] * Bx[kk] * weight
                                break
                            elif Cj[ll] == -1:
                                Cj[ll] = k
                                Cx[ll] += K_flat[k_start + d*size*dof + l] * Bx[kk] * weight
                                break
        for d in range(dof):
            if con_map[i*dof+d]:
                jj = i*dof+d
                count = 0
                start_ = Cp[jj]
                for kk in range(Bp[jj], Bp[jj+1]):
                    k = Bj[kk]
                    Cj[start_ + count] = k
                    Cx[start_ + count] = Bx[kk]
                    count += 1

def matmat_node_basis_flat_prallel(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, B, parallel=False, max_source_nnz=None, cons=None):
    n_col = B.shape[1]
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    if parallel:
        if max_source_nnz is None:
            max_nnz = (np.diff(Bp).max() * (np.diff(elements_ptr).max() * dof) * np.unique(elements_flat, return_counts=True)[1].max())
        else:
            max_nnz = max_source_nnz *  np.diff(Bp).max()
        
        if cons is not None:
            nnz_per_row = matmat_node_basis_flat_nnz_per_row_parallel_wcon(elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes, dof, n_col, Bp, Bj, max_nnz, cons)
        else:
            nnz_per_row = matmat_node_basis_flat_nnz_per_row_parallel(elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes, dof, n_col, Bp, Bj, max_nnz)
    else:
        nnz_per_row = matmat_node_basis_flat_nnz_per_row(elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes, dof, n_col, Bp, Bj)
    Cp = np.zeros(n_nodes*dof+1, dtype=np.int32)
    Cp[1:] = np.cumsum(nnz_per_row)
    Cj = -np.ones(int(Cp[-1]), dtype=np.int32)
    Cx = np.zeros(int(Cp[-1]), dtype=K_flat.dtype)
    if cons is None:
        matmat_node_basis_flat_prallel_kernel(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx)
    else:
        matmat_node_basis_flat_prallel_kernel_wcon(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx, cons)
        
    M = csr_matrix((n_nodes*dof, n_col), dtype=K_flat.dtype)
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

def matmat_node_basis_flat_prallel_(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, B, Cp, cons=None):
    n_col = B.shape[1]
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    Cj = -np.ones(int(Cp[-1]), dtype=np.int32)
    Cx = np.zeros(int(Cp[-1]), dtype=K_flat.dtype)
    if cons is None:
        matmat_node_basis_flat_prallel_kernel(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx)
    else:
        matmat_node_basis_flat_prallel_kernel_wcon(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx, cons)
    M = csr_matrix((n_nodes*dof, n_col), dtype=K_flat.dtype)
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

@njit(["Tuple((f4[:, :], f4[:, :], f4[:]))(f4[:, :], f4[:, :], f4[:, :], i4[:], i4, f4[:], f4[:], i4, i4, i4)",
       "Tuple((f8[:, :], f8[:, :], f8[:]))(f8[:, :], f8[:, :], f8[:, :], i4[:], i4, f8[:], f8[:], i4, i4, i4)"], cache=True, parallel=True)
def FEA_locals_node_basis_parallel(K_single, D_single, B_single, elements_flat, nel, weights, U, dof=3, elements_size=8, B_size = 6):
    strain = np.zeros((nel, B_size), dtype=K_single.dtype)
    stress = np.zeros((nel, B_size), dtype=K_single.dtype)
    strain_energy = np.zeros((nel), dtype=K_single.dtype)
    
    for i in prange(nel):
        start = i*elements_size
        end = start + elements_size
        elements_map = elements_flat[start:end]
        weight = weights[i]
        
        # strain = B_e @ U_e
        # stress = D_e @ strain
        # strain_energy = 1/2 * U_e.T @ K_e @ U_e
        for j in range(elements_size*dof):
            val = 0
            for k in range(B_size):
                strain[i,k] += B_single[k,j] * U[elements_map[j//dof]*dof+j%dof]
            for k in range(elements_size*dof):
                val += weight * K_single[j,k] * U[elements_map[k//dof]*dof+k%dof]
            strain_energy[i] += 0.5 * U[elements_map[j//dof]*dof+j%dof] * val
        
        for j in range(B_size):
            for k in range(B_size):
                stress[i,j] += D_single[j,k] * strain[i,k]
        
    return strain, stress, strain_energy

@njit(["Tuple((f4[:, :], f4[:, :], f4[:]))(f4[:, :, :], f4[:, :, :], f4[:, :, :], i4[:], i4, f4[:], f4[:], i4, i4, i4)",
       "Tuple((f8[:, :], f8[:, :], f8[:]))(f8[:, :, :], f8[:, :, :], f8[:, :, :], i4[:], i4, f8[:], f8[:], i4, i4, i4)"], cache=True, parallel=True)
def FEA_locals_node_basis_parallel_full(Ks, Ds, Bs, elements_flat, nel, weights, U, dof=3, elements_size=8, B_size = 6):
    strain = np.zeros((nel, B_size), dtype=Ks.dtype)
    stress = np.zeros((nel, B_size), dtype=Ks.dtype)
    strain_energy = np.zeros((nel), dtype=Ks.dtype)
    
    for i in prange(nel):
        start = i*elements_size
        end = start + elements_size
        elements_map = elements_flat[start:end]
        weight = weights[i]
        
        # strain = B_e @ U_e
        # stress = D_e @ strain
        # strain_energy = 1/2 * U_e.T @ K_e @ U_e
        for j in range(elements_size*dof):
            val = 0
            for k in range(B_size):
                strain[i,k] += Bs[i,k,j] * U[elements_map[j//dof]*dof+j%dof]
            for k in range(elements_size*dof):
                val += weight* Ks[i,j,k] * U[elements_map[k//dof]*dof+k%dof]
            strain_energy[i] += 0.5 * U[elements_map[j//dof]*dof+j%dof] * val
        
        for j in range(B_size):
            for k in range(B_size):
                stress[i,j] += Ds[i,j,k] * strain[i,k]
        
    return strain, stress, strain_energy

@njit(["Tuple((f4[:, :], f4[:, :], f4[:]))(f4[:], f4[:], f4[:], i4[:], i4[:], i4[:], i4[:], i4[:], i4, f4[:], f4[:], i4, i4)",
       "Tuple((f8[:, :], f8[:, :], f8[:]))(f8[:], f8[:], f8[:], i4[:], i4[:], i4[:], i4[:], i4[:], i4, f8[:], f8[:], i4, i4)"], cache=True, parallel=True)
def FEA_locals_node_basis_parallel_flat(K_flat, D_flat, B_flat, elements_flat, elements_ptr, K_ptr, B_ptr, D_ptr, nel, weights, U, dof=3, B_size = 6):
    strain = np.zeros((nel, B_size), dtype=K_flat.dtype)
    stress = np.zeros((nel, B_size), dtype=K_flat.dtype)
    strain_energy = np.zeros((nel), dtype=K_flat.dtype)
    
    for i in prange(nel):
        start = elements_ptr[i]
        end = elements_ptr[i+1]
        size = end - start
        elements_map = elements_flat[start:end]
        weight = weights[i]
        
        # strain = B_e @ U_e
        # stress = D_e @ strain
        # strain_energy = 1/2 * U_e.T @ K_e @ U_e
        for j in range(size*dof):
            val = 0
            for k in range(B_size):
                strain[i,k] += B_flat[B_ptr[i]+k*size*dof+j] * U[elements_map[j//dof]*dof+j%dof]
            for k in range(size*dof):
                val += weight * K_flat[K_ptr[i]+j*size*dof+k] * U[elements_map[k//dof]*dof+k%dof]
            strain_energy[i] += 0.5 * U[elements_map[j//dof]*dof+j%dof] * val
        
        for j in range(B_size):
            for k in range(B_size):
                stress[i,j] += D_flat[D_ptr[i]+j*B_size+k] * strain[i,k]
        
    return strain, stress, strain_energy