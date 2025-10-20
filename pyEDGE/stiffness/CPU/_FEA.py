from ...core.CPU._ops import (
    get_diagonal_node_basis,
    get_diagonal_node_basis_full,
    get_diagonal_node_basis_flat,
    process_dk,
    process_dk_full,
    process_dk_flat,
    mat_vec_node_basis_parallel,
    mat_vec_node_basis_parallel_full,
    mat_vec_node_basis_parallel_flat,
    mat_vec_node_basis_parallel_wcon,
    mat_vec_node_basis_parallel_full_wcon,
    mat_vec_node_basis_parallel_flat_wcon,
    matmat_node_basis_prallel,
    matmat_node_basis_prallel_,
    matmat_node_basis_full_prallel,
    matmat_node_basis_full_prallel_,
    matmat_node_basis_flat_prallel,
    matmat_node_basis_flat_prallel_,
    matmat_node_basis_flat_prallel_)
from ...geom.CPU._mesh import StructuredMesh, GeneralMesh
import numpy as np
from scipy.sparse import csr_matrix, eye, issparse

class StiffnessKernel:
    def __init__(self):
        self.has_rho = False
        self.rho = None
        self.shape = None
        self.matvec = self.dot
        self.rmatvec = self.dot
        self.matmat = self.dot
        self.rmatmat = self.dot
        
    def construct(self, rho):
        pass
    def _matvec(self, rho, vec):
        pass
    def _rmatvec(self, rho, vec):
        pass
    def _matmat(self, rho, mat):
        pass
    def _rmatmat(self, rho, mat):
        pass
    def set_rho(self, rho):
        self.rho = rho
        self.has_rho = True
    def dot(self, rhs):
        pass
    def diagonal(self):
        pass
    def reset(self):
        self.has_rho = False
        self.rho = None
        self.CSR = None
        self.ptr = None
        self.has_been_constructed = False
        self.has_cons = False
        self.constraints[:] = False
    
    def __matmul__(self, rhs):
        return self.dot(rhs)

class StructuredStiffnessKernel(StiffnessKernel):
    def __init__(self, mesh: StructuredMesh):
        super().__init__()
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.K_single = mesh.K_single
        
        self.dtype = mesh.dtype
        
        self.elements_flat = self.elements.flatten()
        self.el_ids = np.arange(self.elements.shape[0], dtype=np.int32).repeat(self.elements.shape[1])
        self.sorter = np.argsort(self.elements_flat).astype(np.int32)
        self.node_ids = np.searchsorted(self.elements_flat, np.arange(self.nodes.shape[0],dtype=np.int32), sorter=self.sorter, side='left').astype(np.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]
        
        self.has_been_constructed = False
        self.ptr = None
        
        self.max_con_count = np.unique(self.elements_flat, return_counts=True)[1].max() * self.dof * self.elements_size
        
        self.constraints = np.zeros(self.n_nodes*self.dof, dtype=np.bool_)
        
        self.shape = (self.n_nodes*self.dof, self.n_nodes*self.dof)
        
        self.has_cons = False
        self.idx_map = np.arange(self.n_nodes*self.dof, dtype=np.int32)
        self.mat_vec = np.zeros(self.n_nodes*self.dof, dtype=self.dtype)
    
    def diagonal(self, rho=None):
        if rho is None and not self.has_rho:
            raise ValueError("Rho has not been set. diagonal works only after setting rho or if rho is provided.")
        elif rho is None:
            return get_diagonal_node_basis(self.K_single, self.elements_flat, self.el_ids, self.rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
        else:
            return get_diagonal_node_basis(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
                
    def set_constraints(self, constraints):
        self.constraints[:] = False
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def add_constraints(self, constraints):
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def process_grad(self, U):
        return process_dk(self.K_single, self.elements, U, self.dof, self.elements_size)
    
    def construct(self, rho):
        size = self.n_nodes*self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), parallel=True)
            self.ptr = np.copy(self.CSR.indptr)
        else:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), self.ptr, parallel=True)

        self.has_been_constructed = True
        
        return self.CSR

    def _matvec(self, rho, vec):
        if not self.has_cons:
            mat_vec_node_basis_parallel(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.mat_vec, self.dof, self.elements_size)
        else:
            mat_vec_node_basis_parallel_wcon(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.constraints, self.mat_vec, self.dof, self.elements_size)
        return self.mat_vec
    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)
    
    def _matmat(self, rho, mat, Cp=None, parallel=True):
        if not Cp is None:
            if not self.has_cons:
                return matmat_node_basis_prallel_(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp)
            else:
                out = matmat_node_basis_prallel_(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp, self.constraints)
                out.indices[out.indices < 0] = 0
                return out
            
        if not self.has_cons:
            return matmat_node_basis_prallel(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count)
        else:
            return matmat_node_basis_prallel(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count, self.constraints)
            
    def _rmatmat(self, rho, mat, Cp=None, parallel=True):
        return self._matmat(rho, mat, Cp, parallel)
    
    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, np.ndarray):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError("Shape of the input vector does not match the number of nodes and dof.")
            elif issparse(rhs):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    if isinstance(rhs, csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError("Shape of the input matrix does not match the number of nodes and dof.")
            else:
                raise NotImplementedError("Only numpy arrays as vectors and scipy sparse matrices are supported.")
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")

class UniformStiffnessKernel(StiffnessKernel):
    def __init__(self, mesh: GeneralMesh):
        super().__init__()
        
        if not mesh.is_uniform:
            raise ValueError("The mesh is not uniform, you should use GeneralStiffnessKernel instead.")
        
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.elements_flat = mesh.elements_flat
        self.el_ids = np.arange(self.elements.shape[0], dtype=np.int32).repeat(self.elements.shape[1])
        self.sorter = np.argsort(self.elements_flat).astype(np.int32)
        self.node_ids = np.searchsorted(self.elements_flat, np.arange(self.nodes.shape[0]), sorter=self.sorter, side='left').astype(np.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]
        
        self.dtype = mesh.dtype
        
        self.Ks = mesh.Ks
        
        self.has_been_constructed = False
        self.ptr = None
        
        self.max_con_count = np.unique(self.elements_flat, return_counts=True)[1].max() * self.dof * self.elements_size
        
        self.constraints = np.zeros(self.n_nodes*self.dof, dtype=np.bool_)
        
        self.shape = (self.n_nodes*self.dof, self.n_nodes*self.dof)
        
        self.has_cons = False
        self.idx_map = np.arange(self.n_nodes*self.dof, dtype=np.int32)
        self.mat_vec = np.zeros(self.n_nodes*self.dof, dtype=self.dtype)
    
    def diagonal(self, rho=None):
        if rho is None and not self.has_rho:
            raise ValueError("Rho has not been set. diagonal works only after setting rho or if rho is provided.")
        elif rho is None:
            return get_diagonal_node_basis_full(self.Ks, self.elements_flat, self.el_ids, self.rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
        else:
            return get_diagonal_node_basis_full(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
    
    def set_constraints(self, constraints):
        self.constraints[:] = False
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def add_constraints(self, constraints):
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def process_grad(self, U):
        return process_dk_full(self.Ks, self.elements, U, self.dof, self.elements_size)
    
    def construct(self, rho):
        size = self.n_nodes*self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), parallel=True)
            self.ptr = np.copy(self.CSR.indptr)
        else:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), self.ptr, parallel=True)

        self.has_been_constructed = True
        
        return self.CSR
    
    def _matvec(self, rho, vec):
        if not self.has_cons:
            mat_vec_node_basis_parallel_full(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.mat_vec, self.dof, self.elements_size)
        else:
            mat_vec_node_basis_parallel_full_wcon(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.constraints, self.mat_vec, self.dof, self.elements_size)

        return self.mat_vec
    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)
        
    def _matmat(self, rho, mat, Cp=None, parallel=True):
        if not Cp is None:
            if not self.has_cons:
                return matmat_node_basis_full_prallel_(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp)
            else:
                out = matmat_node_basis_full_prallel_(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp, self.constraints)
                out.indices[out.indices < 0] = 0
                return out
            
        if not self.has_cons:
            return matmat_node_basis_full_prallel(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count)
        else:
            out = matmat_node_basis_full_prallel(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count, self.constraints)
            out.indices[out.indices < 0] = 0
            return out
    
    def _rmatmat(self, rho, mat, Cp=None, parallel=True):
        return self._matmat(rho, mat, Cp, parallel)

    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, np.ndarray):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError("Shape of the input vector does not match the number of nodes and dof.")
            elif issparse(rhs):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    if isinstance(rhs, csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError("Shape of the input matrix does not match the number of nodes and dof.")
            else:
                raise NotImplementedError("Only numpy arrays as vectors and scipy sparse matrices are supported.")
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")
    
class GeneralStiffnessKernel(StiffnessKernel):
    def __init__(self, mesh: GeneralMesh):
        super().__init__()
        
        if mesh.is_uniform:
            raise ValueError("The mesh is uniform, you should use UniformStiffnessKernel instead.")
        
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.elements_flat = mesh.elements_flat
        self.element_sizes = mesh.element_sizes
        self.K_flat = mesh.K_flat
        
        self.dtype = mesh.dtype
    
        self.K_ptr = mesh.K_ptr
        self.elements_ptr = mesh.elements_ptr
        
        self.el_ids = np.arange(self.elements.shape[0], dtype=np.int32).repeat(self.element_sizes)
        self.sorter = np.argsort(self.elements_flat).astype(np.int32)
        self.node_ids = np.searchsorted(self.elements_flat, np.arange(self.nodes.shape[0]), sorter=self.sorter, side='left').astype(np.int32)
        self.n_nodes = self.nodes.shape[0]
        
        self.has_been_constructed = False
        self.ptr = None
        
        self.max_con_count = np.unique(self.elements_flat, return_counts=True)[1].max() * self.dof * self.element_sizes.max()
        
        self.constraints = np.zeros(self.n_nodes*self.dof, dtype=np.bool_)
        
        self.shape = (self.n_nodes*self.dof, self.n_nodes*self.dof)
        
        self.has_cons = False
        self.idx_map = np.arange(self.n_nodes*self.dof, dtype=np.int32)
    
    def diagonal(self, rho=None):
        if rho is None and not self.has_rho:
            raise ValueError("Rho has not been set. diagonal works only after setting rho or if rho is provided.")
        elif rho is None:
            return get_diagonal_node_basis_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, self.rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.constraints)
        else:
            return get_diagonal_node_basis_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.constraints)
    
    def set_constraints(self, constraints):
        self.constraints[:] = False
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def add_constraints(self, constraints):
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def process_grad(self, U):
        return process_dk_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, U, self.dof)
    
    def construct(self, rho):
        size = self.n_nodes*self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), parallel=True)
            self.ptr = np.copy(self.CSR.indptr)
        else:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), self.ptr, parallel=True)

        self.has_been_constructed = True
        
        return self.CSR
    
    def _matvec(self, rho, vec):
        if not self.has_cons:
            return mat_vec_node_basis_parallel_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.dof)
        else:
            return mat_vec_node_basis_parallel_flat_wcon(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.constraints, self.dof)
    
    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)
    
    def _matmat(self, rho, mat, Cp=None, parallel=False):
        if not Cp is None:
            if not self.has_cons:
                return matmat_node_basis_flat_prallel_(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, Cp)
            else:
                out = matmat_node_basis_flat_prallel_(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, Cp, self.constraints)
                out.indices[out.indices < 0] = 0
                return out
        if not self.has_cons:
            return matmat_node_basis_flat_prallel(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, parallel, self.max_con_count)
        else:
            out = matmat_node_basis_flat_prallel(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, parallel, self.max_con_count, self.constraints)
            out.indices[out.indices < 0] = 0
            return out

    def _rmatmat(self, rho, mat, Cp=None, parallel=False):
        return self._matmat(rho, mat, Cp, parallel)
    
    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, np.ndarray):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError("Shape of the input vector does not match the number of nodes and dof.")
            elif issparse(rhs):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    if isinstance(rhs, csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError("Shape of the input matrix does not match the number of nodes and dof.")
            else:
                raise NotImplementedError("Only numpy arrays as vectors and scipy sparse matrices are supported.")
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")