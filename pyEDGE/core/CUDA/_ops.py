import numpy as np
import cupy as cp
from .Kernels import (
    mat_vec_node_basis_parallel_flat_cuda_kernel,
    mat_vec_node_basis_parallel_full_cuda_kernel,
    mat_vec_node_basis_parallel_cuda_kernel,
    mat_vec_node_basis_parallel_flat_wcon_cuda_kernel,
    mat_vec_node_basis_parallel_full_wcon_cuda_kernel,
    mat_vec_node_basis_parallel_wcon_cuda_kernel,
    process_dk_kernel_cuda,
    process_dk_full_kernel_cuda,
    process_dk_flat_kernel_cuda,
    matmat_node_basis_nnz_per_row_kernel,
    matmat_node_basis_nnz_per_row_wcon_kernel,
    matmat_node_basis_parallel_kernel,
    matmat_node_basis_full_parallel_kernel,
    matmat_node_basis_flat_parallel_kernel,
    matmat_node_basis_flat_nnz_per_row_kernel,
    matmat_node_basis_flat_nnz_per_row_wcon_kernel,
    matmat_node_basis_parallel_wcon_kernel,
    matmat_node_basis_full_parallel_wcon_kernel,
    matmat_node_basis_flat_parallel_wcon_kernel,
    get_diagonal_node_basis_cuda_kernel,
    get_diagonal_node_basis_full_cuda_kernel,
    get_diagonal_node_basis_flat_cuda_kernel,
    FEA_locals_node_basis_parallel_full_cuda_kernel,
    FEA_locals_node_basis_parallel_flat_cuda_kernel,
    FEA_locals_node_basis_parallel_cuda_kernel,
)

def mat_vec_node_basis_parallel_flat_cuda(K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, vec, dof=3, cons=None, out=None):
    if out is None:
        out = cp.zeros(n_nodes*dof, dtype=K_flat.dtype)
    elem_flat_size = elements_flat.shape[0]
    
    threadsperblock = 256
    blockspergrid = (node_ids.shape[0] + (threadsperblock - 1)) // threadsperblock
    
    if cons is not None:
        mat_vec_node_basis_parallel_flat_wcon_cuda_kernel((blockspergrid,), (threadsperblock,), (K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, vec, dof, elem_flat_size, out, cons))
    else:
        mat_vec_node_basis_parallel_flat_cuda_kernel((blockspergrid,), (threadsperblock,), (K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, sorter, node_ids, n_nodes, vec, dof, elem_flat_size, out))
    return out

def mat_vec_node_basis_parallel_full_cuda(Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, dof=3, elements_size=8, cons=None, out=None):
    if out is None:
        out = cp.zeros(n_nodes*dof, dtype=Ks.dtype)
    elem_flat_size = elements_flat.shape[0]
    
    threadsperblock = 256
    blockspergrid = (node_ids.shape[0] + (threadsperblock - 1)) // threadsperblock

    if cons is not None:
        mat_vec_node_basis_parallel_full_wcon_cuda_kernel((blockspergrid,), (threadsperblock,), (Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, dof, elements_size, elem_flat_size, out, cons))
    else:
        mat_vec_node_basis_parallel_full_cuda_kernel((blockspergrid,), (threadsperblock,), (Ks, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, dof, elements_size, elem_flat_size, out))
    return out

def mat_vec_node_basis_parallel_cuda(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, dof=3, elements_size=8, cons=None, out=None):
    if out is None:
        out = cp.zeros(n_nodes*dof, dtype=K_single.dtype)
    elem_flat_size = elements_flat.shape[0]
    
    threadsperblock = 256
    blockspergrid = (node_ids.shape[0] + (threadsperblock - 1)) // threadsperblock

    if cons is not None:
        mat_vec_node_basis_parallel_wcon_cuda_kernel((blockspergrid,), (threadsperblock,), (K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, dof, elements_size, elem_flat_size, out, cons))
    else:
        mat_vec_node_basis_parallel_cuda_kernel((blockspergrid,), (threadsperblock,), (K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, vec, dof, elements_size, elem_flat_size, out))
    
    return out

def process_dk_cuda(K_single, elements, U, dof=3, elements_size=8, out=None):
    nel = elements.shape[0]//elements_size
    if out is None:
        out = cp.zeros(nel, dtype=K_single.dtype)
    threadsperblock = 256
    blockspergrid = (nel + (threadsperblock - 1)) // threadsperblock
    process_dk_kernel_cuda((blockspergrid,), (threadsperblock,), (K_single, elements, U, dof, elements_size, nel, out))
    return out

def process_dk_full_cuda(Ks, elements, U, dof=3, elements_size=8, out=None):
    nel = elements.shape[0]//elements_size
    if out is None:
        out = cp.zeros(nel, dtype=Ks.dtype)
    threadsperblock = 256
    blockspergrid = (nel + (threadsperblock - 1)) // threadsperblock
    process_dk_full_kernel_cuda((blockspergrid,), (threadsperblock,), (Ks, elements, U, dof, elements_size, nel, out))
    return out

def process_dk_flat_cuda(K_flat, elements_flat, K_ptr, elements_ptr, U, dof=3, out=None):
    
    nel = elements_ptr.shape[0] - 1
    if out is None:
        out = cp.zeros(nel, dtype=K_flat.dtype)
    threadsperblock = 256
    blockspergrid = (nel + (threadsperblock - 1)) // threadsperblock
    process_dk_flat_kernel_cuda((blockspergrid,), (threadsperblock,), (K_flat, elements_flat, K_ptr, elements_ptr, U, dof, nel, out))
    return out

'''
MatMat kernels
'''

def matmat_node_basis_nnz_per_row_cuda(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz):
    
    # Initialize output array
    nnz_per_row = cp.zeros(n_nodes*dof, dtype=np.int32)

    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    # print(matmat_node_basis_nnz_per_row_kernel_code.replace('max_nnz', str(int(np.ceil(max_nnz/32))*32)))
    # matmat_node_basis_nnz_per_row_kernel = cp.RawKernel(matmat_node_basis_nnz_per_row_kernel_code.replace('max_nnz', str(int(np.ceil(max_nnz/32))*32)), 'matmat_node_basis_nnz_per_row_kernel')
    max_nnz = int(np.ceil(max_nnz/32))*32
    # Launch kernel
    matmat_node_basis_nnz_per_row_kernel((blockspergrid,), (threadsperblock,),
                                        (elements_flat, el_ids, sorter, node_ids, n_nodes, 
                                        dof, elements_size, n_col, Bp, Bj, 
                                        elements_flat.shape[0], nnz_per_row),max_nnz)
                        
    return nnz_per_row

def matmat_node_basis_nnz_per_row_wcon_cuda(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz, cons):
    
    # Initialize output array
    nnz_per_row = cp.zeros(n_nodes*dof, dtype=np.int32)

    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    # print(matmat_node_basis_nnz_per_row_kernel_code.replace('max_nnz', str(int(np.ceil(max_nnz/32))*32)))
    # matmat_node_basis_nnz_per_row_kernel = cp.RawKernel(matmat_node_basis_nnz_per_row_kernel_code.replace('max_nnz', str(int(np.ceil(max_nnz/32))*32)), 'matmat_node_basis_nnz_per_row_kernel')
    max_nnz = int(np.ceil(max_nnz/32))*32
    # Launch kernel
    matmat_node_basis_nnz_per_row_wcon_kernel((blockspergrid,), (threadsperblock,),
                                        (elements_flat, el_ids, sorter, node_ids, n_nodes, 
                                        dof, elements_size, n_col, Bp, Bj, 
                                        elements_flat.shape[0], nnz_per_row, cons),max_nnz)
                        
    return nnz_per_row


def matmat_node_basis_parallel_cuda(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, B, max_source_nnz=None, cons=None):
    n_col = B.shape[1]
    
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    max_elem_size = elements_flat.shape[0]
    
    
    if max_source_nnz is None:
        max_nnz = int(cp.diff(Bp).max() * (elements_size * dof) * cp.unique(elements_flat, return_counts=True)[1].max())
    else:
        max_nnz = int(max_source_nnz *  cp.diff(Bp).max())
    
    if cons is None:
        nnz_per_row = matmat_node_basis_nnz_per_row_cuda(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz)
    else:
        nnz_per_row = matmat_node_basis_nnz_per_row_wcon_cuda(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz, cons)
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    Cp = cp.zeros(n_nodes*dof+1, dtype=cp.int32)
    Cp[1:] = cp.cumsum(cp.array(nnz_per_row))
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32) - 1
    Cx = cp.zeros(int(Cp[-1]), dtype=K_single.dtype)

    # Launch kernel
    if cons is None:
        matmat_node_basis_parallel_kernel((blockspergrid,), (threadsperblock,), (K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size))
    else:
        matmat_node_basis_parallel_wcon_kernel((blockspergrid,), (threadsperblock,), (K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size, cons))
    
    M = cp.sparse.csr_matrix((n_nodes*dof, n_col))
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

def matmat_node_basis_parallel_cuda_(K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, B, Cp, cons=None):
    n_col = B.shape[1]
    
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    max_elem_size = elements_flat.shape[0]
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32) - 1
    Cx = cp.zeros(int(Cp[-1]), dtype=K_single.dtype)
    
    # Launch kernel
    if cons is None:
        matmat_node_basis_parallel_kernel((blockspergrid,), (threadsperblock,), (K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size))
    else:
        matmat_node_basis_parallel_wcon_kernel((blockspergrid,), (threadsperblock,), (K_single, elements_flat, el_ids, weights, sorter, node_ids, n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size, cons))
    
    M = cp.sparse.csr_matrix((n_nodes*dof, n_col))
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

def matmat_node_basis_flat_nnz_per_row_cuda(elements_flat, elements_ptr, el_ids, 
                                           sorter, node_ids, n_nodes, dof, n_col, 
                                           Bp, Bj, max_nnz):
    # Initialize output array
    nnz_per_row = cp.zeros(n_nodes*dof, dtype=cp.int32)
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    max_nnz = int(np.ceil(max_nnz/32))*32
    # Launch kernel
    matmat_node_basis_flat_nnz_per_row_kernel(
        (blockspergrid,), (threadsperblock,),
        (elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes, 
         dof, n_col, Bp, Bj, elements_flat.shape[0], nnz_per_row), max_nnz
    )
    
    return nnz_per_row

def matmat_node_basis_flat_nnz_per_row_wcon_cuda(elements_flat, elements_ptr, el_ids,
                                                sorter, node_ids, n_nodes, dof, n_col,
                                                Bp, Bj, max_nnz, cons):
    # Initialize output array
    nnz_per_row = cp.zeros(n_nodes*dof, dtype=cp.int32)
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    max_nnz = int(np.ceil(max_nnz/32))*32
    # Launch kernel
    matmat_node_basis_flat_nnz_per_row_wcon_kernel(
        (blockspergrid,), (threadsperblock,),
        (elements_flat, elements_ptr, el_ids, sorter, node_ids, n_nodes,
            dof, n_col, Bp, Bj, elements_flat.shape[0], nnz_per_row, cons), max_nnz
    )
    
    return nnz_per_row

def matmat_node_basis_full_parallel_cuda(Ks, elements_flat, el_ids, weights, sorter, node_ids, 
                                       n_nodes, dof, elements_size, B, max_source_nnz=None, cons=None):
    n_col = B.shape[1]
    
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    max_elem_size = elements_flat.shape[0]
    
    
    if max_source_nnz is None:
        max_nnz = int(cp.diff(Bp).max() * (elements_size * dof) * cp.unique(elements_flat, return_counts=True)[1].max())
    else:
        max_nnz = int(max_source_nnz *  cp.diff(Bp).max())
    
    if cons is None:
        nnz_per_row = matmat_node_basis_nnz_per_row_cuda(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz)
    else:
        nnz_per_row = matmat_node_basis_nnz_per_row_wcon_cuda(elements_flat, el_ids, sorter, node_ids, n_nodes, dof, elements_size, n_col, Bp, Bj, max_nnz, cons)
        
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    Cp = cp.zeros(n_nodes*dof+1, dtype=cp.int32)
    Cp[1:] = cp.cumsum(cp.array(nnz_per_row))
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32) - 1
    Cx = cp.zeros(int(Cp[-1]), dtype=Ks.dtype)

    # Launch kernel
    if cons is None:
        matmat_node_basis_full_parallel_kernel(
            (blockspergrid,), (threadsperblock,),
            (Ks, elements_flat, el_ids, weights, sorter, node_ids, 
            n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size)
        )
    else:
        matmat_node_basis_full_parallel_wcon_kernel(
            (blockspergrid,), (threadsperblock,),
            (Ks, elements_flat, el_ids, weights, sorter, node_ids, 
            n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size, cons)
        )
    
    M = cp.sparse.csr_matrix((n_nodes*dof, n_col))
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

def matmat_node_basis_full_parallel_cuda_(Ks, elements_flat, el_ids, weights, sorter, 
                                        node_ids, n_nodes, dof, elements_size, B, Cp, cons=None):
    n_col = B.shape[1]
    
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    max_elem_size = elements_flat.shape[0]
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32) - 1
    Cx = cp.zeros(int(Cp[-1]), dtype=Ks.dtype)
    
    # Launch kernel
    if cons is None:
        matmat_node_basis_full_parallel_kernel(
            (blockspergrid,), (threadsperblock,),
            (Ks, elements_flat, el_ids, weights, sorter, node_ids, 
            n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size)
        )
    else:
        matmat_node_basis_full_parallel_wcon_kernel(
            (blockspergrid,), (threadsperblock,),
            (Ks, elements_flat, el_ids, weights, sorter, node_ids, 
            n_nodes, dof, elements_size, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size, cons)
        )
    
    M = cp.sparse.csr_matrix((n_nodes*dof, n_col))
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

def matmat_node_basis_flat_parallel_cuda(K_flat, elements_flat, K_ptr, elements_ptr, 
                                       el_ids, weights, sorter, node_ids, n_nodes, 
                                       dof, B, max_source_nnz=None, cons=None):
    n_col = B.shape[1]
    
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    max_elem_size = elements_flat.shape[0]
    
    if max_source_nnz is None:
        max_nnz = int(cp.diff(Bp).max() * (cp.diff(elements_ptr).max() * dof) * 
                     cp.unique(elements_flat, return_counts=True)[1].max())
    else:
        max_nnz = int(max_source_nnz * cp.diff(Bp).max())
    
    if cons is None:
        nnz_per_row = matmat_node_basis_flat_nnz_per_row_cuda(
            elements_flat, elements_ptr, el_ids, sorter, node_ids, 
            n_nodes, dof, n_col, Bp, Bj, max_nnz
        )
    else:
        nnz_per_row = matmat_node_basis_flat_nnz_per_row_wcon_cuda(
            elements_flat, elements_ptr, el_ids, sorter, node_ids, 
            n_nodes, dof, n_col, Bp, Bj, max_nnz, cons
        )

    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    Cp = cp.zeros(n_nodes*dof+1, dtype=cp.int32)
    Cp[1:] = cp.cumsum(nnz_per_row)
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32) - 1
    Cx = cp.zeros(int(Cp[-1]), dtype=K_flat.dtype)

    # Launch kernel
    if cons is None:
        matmat_node_basis_flat_parallel_kernel(
            (blockspergrid,), (threadsperblock,),
            (K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, 
            sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size)
        )
    else:
        matmat_node_basis_flat_parallel_wcon_kernel(
            (blockspergrid,), (threadsperblock,),
            (K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, 
            sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size, cons)
        )
        
    M = cp.sparse.csr_matrix((n_nodes*dof, n_col))
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M

def matmat_node_basis_flat_parallel_cuda_(K_flat, elements_flat, K_ptr, elements_ptr, 
                                        el_ids, weights, sorter, node_ids, n_nodes, 
                                        dof, B, Cp, cons=None):
    n_col = B.shape[1]
    
    Bp = B.indptr
    Bj = B.indices
    Bx = B.data
    max_elem_size = elements_flat.shape[0]
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32) - 1
    Cx = cp.zeros(int(Cp[-1]), dtype=K_flat.dtype)
    
    # Launch kernel
    if cons is None:
        matmat_node_basis_flat_parallel_kernel(
            (blockspergrid,), (threadsperblock,),
            (K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, 
            sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size)
        )
    else:
        matmat_node_basis_flat_parallel_wcon_kernel(
            (blockspergrid,), (threadsperblock,),
            (K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights, 
            sorter, node_ids, n_nodes, dof, Bp, Bj, Bx, Cp, Cj, Cx, max_elem_size, cons)
        )
    
    M = cp.sparse.csr_matrix((n_nodes*dof, n_col))
    M.data = Cx
    M.indices = Cj
    M.indptr = Cp
    return M
    
def get_diagonal_node_basis_cuda(K_single, elements_flat, el_ids, weights, sorter, 
                               node_ids, n_nodes, dof, elements_size, cons_map, diag=None):
    """CUDA version of get_diagonal_node_basis"""
    if diag is None:
        diag = cp.zeros(n_nodes*dof, dtype=K_single.dtype)
    
    max_elem_size = elements_flat.shape[0]
    
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    get_diagonal_node_basis_cuda_kernel(
        (blockspergrid,), (threadsperblock,),
        (K_single, elements_flat, el_ids, weights, sorter, node_ids, 
         n_nodes, dof, elements_size, cons_map, diag, max_elem_size)
    )
    
    return diag

def get_diagonal_node_basis_full_cuda(Ks, elements_flat, el_ids, weights, sorter, 
                                    node_ids, n_nodes, dof, elements_size, cons_map, diag=None):
    """CUDA version of get_diagonal_node_basis_full"""
    if diag is None:
        diag = cp.zeros(n_nodes*dof, dtype=Ks.dtype)
    
    max_elem_size = elements_flat.shape[0]
    
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    get_diagonal_node_basis_full_cuda_kernel(
        (blockspergrid,), (threadsperblock,),
        (Ks, elements_flat, el_ids, weights, sorter, node_ids,
         n_nodes, dof, elements_size, cons_map, diag, max_elem_size)
    )
    
    return diag

def get_diagonal_node_basis_flat_cuda(K_flat, elements_flat, K_ptr, elements_ptr,
                                    el_ids, weights, sorter, node_ids, n_nodes, 
                                    dof, cons_map, diag=None):
    """CUDA version of get_diagonal_node_basis_flat"""
    if diag is None:
        diag = cp.zeros(n_nodes*dof, dtype=K_flat.dtype)
    
    max_elem_size = elements_flat.shape[0]
    
    threadsperblock = 256
    blockspergrid = (n_nodes + (threadsperblock - 1)) // threadsperblock
    
    get_diagonal_node_basis_flat_cuda_kernel(
        (blockspergrid,), (threadsperblock,),
        (K_flat, elements_flat, K_ptr, elements_ptr, el_ids, weights,
         sorter, node_ids, n_nodes, dof, cons_map, diag, max_elem_size)
    )
    
    return diag

def FEA_locals_node_basis_parallel_cuda(K_single, D_single, B_single, elements_flat, nel, weights, U, dof=3, elements_size=8, B_size=6):
    """CUDA version of FEA_locals_node_basis_parallel"""
    strain = cp.zeros((nel, B_size), dtype=K_single.dtype)
    stress = cp.zeros((nel, B_size), dtype=K_single.dtype)
    strain_energy = cp.zeros(nel, dtype=K_single.dtype)
    
    threadsperblock = 256
    blockspergrid = (nel + (threadsperblock - 1)) // threadsperblock
    
    # Call the CUDA kernel
    FEA_locals_node_basis_parallel_cuda_kernel((blockspergrid,), (threadsperblock,),
        (K_single, D_single, B_single, elements_flat, nel, weights, U, 
         dof, elements_size, B_size, strain, stress, strain_energy))
    
    return strain, stress, strain_energy

def FEA_locals_node_basis_parallel_full_cuda(Ks, Ds, Bs, elements_flat, nel, weights, U, dof=3, elements_size=8, B_size=6):
    """CUDA version of FEA_locals_node_basis_parallel_full"""
    strain = cp.zeros((nel, B_size), dtype=Ks.dtype)
    stress = cp.zeros((nel, B_size), dtype=Ks.dtype)
    strain_energy = cp.zeros(nel, dtype=Ks.dtype)
    
    threadsperblock = 256
    blockspergrid = (nel + (threadsperblock - 1)) // threadsperblock
    
    # Call the CUDA kernel
    FEA_locals_node_basis_parallel_full_cuda_kernel((blockspergrid,), (threadsperblock,),
        (Ks, Ds, Bs, elements_flat, nel, weights, U,
         dof, elements_size, B_size, strain, stress, strain_energy))
    
    return strain, stress, strain_energy

def FEA_locals_node_basis_parallel_flat_cuda(K_flat, D_flat, B_flat, elements_flat, elements_ptr, K_ptr, B_ptr, D_ptr, nel, weights, U, dof=3, B_size=6):
    """CUDA version of FEA_locals_node_basis_parallel_flat"""
    strain = cp.zeros((nel, B_size), dtype=K_flat.dtype)
    stress = cp.zeros((nel, B_size), dtype=K_flat.dtype)
    strain_energy = cp.zeros(nel, dtype=K_flat.dtype)
    
    threadsperblock = 256
    blockspergrid = (nel + (threadsperblock - 1)) // threadsperblock
    
    # Call the CUDA kernel
    FEA_locals_node_basis_parallel_flat_cuda_kernel((blockspergrid,), (threadsperblock,),
        (K_flat, D_flat, B_flat, elements_flat, elements_ptr, K_ptr, B_ptr, D_ptr,
         nel, weights, U, dof, B_size, strain, stress, strain_energy))
    
    return strain, stress, strain_energy