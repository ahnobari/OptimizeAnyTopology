import numpy as np
import cupy as cp
from .Kernels import (
    restriction_3d_kernel,
    restriction_2d_kernel,
    prolongation_3d_kernel,
    prolongation_2d_kernel,
    get_restricted_3d_l0_nnz_based,
    get_restricted_3d_l1p_nnz_based,
    get_restricted_2d_l0_nnz_based,
    get_restricted_2d_l1p_nnz_based,
    csr_diagonal
)

from ...geom.CUDA._mesh import CuStructuredMesh2D, CuStructuredMesh3D
from ...stiffness.CUDA import StructuredStiffnessKernel as CuStructuredStiffnessKernel
from typing import Union

def get_restricted_l0_cuda(mesh: Union[CuStructuredMesh2D, CuStructuredMesh3D], kernel: CuStructuredStiffnessKernel, Cp = None):
    
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
        
        n_tasks = nx_coarse * ny_coarse * 9
        
        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//9*dof, dtype=cp.int32) * 18 # 18 is hard coded for 2D linear quad
            Cp = cp.zeros(n_tasks//9*dof + 1, dtype=cp.int32)
            Cp[1:] = cp.cumsum(nnz)

        K_flat = kernel.K_single
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=K_flat.dtype)
        weights = kernel.rho
        
        get_restricted_2d_l0_nnz_based((blocks_per_grid,),
                                        (threads_per_block,),
                                        (K_flat,
                                         nx_fine, ny_fine, nx_coarse, ny_coarse,
                                         Cp, Cj, Cx,
                                         dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes))
    
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
        
        n_tasks = nx_coarse * ny_coarse * nz_coarse * 27

        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//27*dof, dtype=cp.int32) * 81 # 81 is hard coded for 3D linear hex
            Cp = cp.zeros(n_tasks//27*dof + 1, dtype=cp.int32)
            Cp[1:] = cp.cumsum(nnz)
        
        K_flat = kernel.K_single
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=K_flat.dtype)
        weights = kernel.rho
        get_restricted_3d_l0_nnz_based((blocks_per_grid,), 
                                       (threads_per_block,), 
                                       (K_flat,
                                        nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse,
                                        Cp, Cj, Cx,
                                        dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes))    
    
    K = cp.sparse.csr_matrix((Cx, Cj, Cp), shape=(Cp.shape[0]-1, Cp.shape[0]-1))
    K.has_canonical_format = True
    diag = cp.zeros(K.shape[0], dtype=K.dtype)
    n_row = K.shape[0]
    threads_per_block = 256
    blocks_per_grid = (n_row + threads_per_block - 1) // threads_per_block
    
    csr_diagonal((blocks_per_grid,), (threads_per_block,), (K.data, K.indptr, K.indices, diag, n_row))
    
    K.diagonal = lambda: diag
    
    return K

def get_restricted_l1p_cuda(A : cp.sparse.csr_matrix, nel, dof, Cp = None):
    
    dim = len(nel)
    if dim == 2:
        nx_fine = int(nel[0] + 1)
        ny_fine = int(nel[1] + 1)
        nx_coarse = int(nel[0] // 2 + 1)
        ny_coarse = int(nel[1] // 2 + 1)
        
        n_tasks = nx_coarse*ny_coarse*9
        
        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//9*dof, dtype=cp.int32) * 18 # 18 is hard coded for 2D linear quad
            Cp = cp.zeros(n_tasks//9*dof + 1, dtype=cp.int32)
            Cp[1:] = cp.cumsum(nnz)
        
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=A.dtype)
        
        get_restricted_2d_l1p_nnz_based((blocks_per_grid,), (threads_per_block,), 
                              (A.data, A.indices, A.indptr, nx_fine, ny_fine, nx_coarse, ny_coarse, Cp, Cj, Cx, dof))
                
    else:
        nx_fine = int(nel[0] + 1)
        ny_fine = int(nel[1] + 1)
        nz_fine = int(nel[2] + 1)
        nx_coarse = int(nel[0] // 2 + 1)
        ny_coarse = int(nel[1] // 2 + 1)
        nz_coarse = int(nel[2] // 2 + 1)
        
        n_tasks = nx_coarse*ny_coarse*nz_coarse*27
        
        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//27 * dof, dtype=np.int32) * 81 # 81 is hard coded for 3D linear hex
            Cp = cp.zeros(n_tasks//27 * dof + 1, dtype=np.int32)
            Cp[1:] = cp.cumsum(nnz)

        
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=A.dtype)

        get_restricted_3d_l1p_nnz_based((blocks_per_grid,), (threads_per_block,),
                              (A.data, A.indices, A.indptr, nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse, Cp, Cj, Cx, dof))
                
    K = cp.sparse.csr_matrix((Cx, Cj, Cp), shape=(Cp.shape[0]-1, Cp.shape[0]-1))
    # K.has_canonical_format = True
    # diag = cp.zeros(K.shape[0], dtype=K.dtype)
    # n_row = K.shape[0]
    # threads_per_block = 256
    # blocks_per_grid = (n_row + threads_per_block - 1) // threads_per_block
    
    # csr_diagonal((blocks_per_grid,), (threads_per_block,), (K.data, K.indptr, K.indices, diag, n_row))
    
    # K.diagonal = lambda: diag
    return K

def apply_restriction_cuda(v, nel, dof):

    # Convert input to GPU if needed
    if isinstance(v, np.ndarray):
        v = cp.array(v, dtype=v.dtype)
    
    if len(nel) == 3:
        nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2, nel[2] // 2)
        nx_coarse, ny_coarse, nz_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1, nel_coarse[2] + 1
        
        # Allocate output array
        n_coarse_nodes = nx_coarse * ny_coarse * nz_coarse
        v_coarse = cp.zeros(n_coarse_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_coarse_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        restriction_3d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_coarse, nx_fine, ny_fine, nz_fine,
             nx_coarse, ny_coarse, nz_coarse, dof)
        )
        
    else:  # 2D case
        nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2)
        nx_coarse, ny_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1
        
        # Allocate output array
        n_coarse_nodes = nx_coarse * ny_coarse
        v_coarse = cp.zeros(n_coarse_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_coarse_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        restriction_2d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_coarse, nx_fine, ny_fine, nx_coarse, ny_coarse, dof)
        )
    
    return v_coarse

def apply_prolongation_cuda(v, nel, dof):
    # Convert input to GPU if needed
    if isinstance(v, np.ndarray):
        v = cp.array(v, dtype=v.dtype)
    
    if len(nel) == 3:
        nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2, nel[2] // 2)
        nx_coarse, ny_coarse, nz_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1, nel_coarse[2] + 1
        
        # Allocate output array
        n_fine_nodes = nx_fine * ny_fine * nz_fine
        v_fine = cp.zeros(n_fine_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_fine_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        prolongation_3d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_fine, nx_fine, ny_fine, nz_fine,
             nx_coarse, ny_coarse, nz_coarse, dof)
        )
        
    else:  # 2D case
        nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2)
        nx_coarse, ny_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1
        
        # Allocate output array
        n_fine_nodes = nx_fine * ny_fine
        v_fine = cp.zeros(n_fine_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_fine_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        prolongation_2d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_fine, nx_fine, ny_fine, nx_coarse, ny_coarse, dof)
        )
    
    return v_fine