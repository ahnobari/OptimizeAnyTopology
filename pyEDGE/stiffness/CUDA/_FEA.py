from ...core.CUDA import *
from ...geom.CUDA._mesh import CuStructuredMesh2D, CuStructuredMesh3D, CuGeneralMesh
import cupy as cp
from typing import Union

class StiffnessKernel:
    def __init__(self):
        self.has_rho = False
        self.rho = None
        self.shape = None
        self.matvec = self.dot
        self.rmatvec = self.dot
        self.matmat = self.dot
        self.rmatmat = self.dot
        
        self.mat_vec = None
        self.diag = None
        self.dk = None

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
        self.rho = cp.array(rho, dtype=rho.dtype, copy=False)
        self.has_rho = True

    def dot(self, rhs):
        pass

    def reset(self):
        self.has_rho = False
        self.rho = None
        self.CSR = None
        self.ptr = None
        self.has_been_constructed = False
        self.constraints[:] = False
        self.has_con = False
        self.mat_vec = None
        self.diag = None
        self.dk = None

    def diagonal(self):
        pass

    def __matmul__(self, rhs):
        return self.dot(rhs)


class StructuredStiffnessKernel(StiffnessKernel):
    def __init__(self, mesh: Union[CuStructuredMesh2D, CuStructuredMesh3D]):
        super().__init__()
        self.mesh = mesh
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.K_single = mesh.K_single
        
        self.dtype = mesh.dtype

        # move to gpu and cast to 32 bit
        self.nodes = cp.array(self.nodes, dtype=self.nodes.dtype, copy=False)
        self.elements = cp.array(self.elements, dtype=self.elements.dtype, copy=False)
        self.K_single = cp.array(self.K_single, dtype=self.K_single.dtype, copy=False)

        self.elements_flat = self.elements.flatten()
        self.el_ids = cp.arange(self.elements.shape[0], dtype=cp.int32).repeat(
            self.elements.shape[1]
        )
        self.sorter = cp.argsort(self.elements_flat).astype(cp.int32)
        self.node_ids = cp.searchsorted(
            self.elements_flat,
            cp.arange(self.nodes.shape[0], dtype=cp.int32),
            sorter=self.sorter,
            side="left",
        ).astype(cp.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]

        self.has_been_constructed = False
        self.ptr = None

        self.max_con_count = (self.elements_size * int(self.dof)) * cp.unique(
            self.elements_flat, return_counts=True
        )[1].max()

        self.constraints = cp.zeros(self.n_nodes * self.dof, dtype=cp.bool_)

        self.shape = (self.n_nodes * self.dof, self.n_nodes * self.dof)

        self.has_con = False
        self.idx_map = cp.arange(self.n_nodes * self.dof, dtype=cp.int32)

    def diagonal(self, rho=None):
        if rho is None and not self.has_rho:
            raise ValueError(
                "Rho has not been set. diagonal works only after setting rho or if rho is provided."
            )
        elif rho is None:
            self.diag =  get_diagonal_node_basis(
                self.K_single,
                self.elements_flat,
                self.el_ids,
                self.rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag
        else:
            self.diag = get_diagonal_node_basis(
                self.K_single,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag

    def set_constraints(self, constraints):
        self.constraints[:] = False
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def add_constraints(self, constraints):
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def process_grad(self, U):
        self.dk = process_dk(
            self.K_single, self.elements_flat, U, self.dof, self.elements_size, self.dk
        )
        return self.dk

    def construct(self, rho):
        size = self.n_nodes * self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype)
            )
            self.ptr = self.CSR.indptr
        else:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype), self.ptr
            )
        self.has_been_constructed = True

        return self.CSR

    def _matvec(self, rho, vec):
        if not self.has_con:
            self.mat_vec = mat_vec_node_basis_parallel(
                self.K_single,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.elements_size,
                out=self.mat_vec
            )
            return self.mat_vec

        self.mat_vec = mat_vec_node_basis_parallel(
            self.K_single,
            self.elements_flat,
            self.el_ids,
            rho,
            self.sorter,
            self.node_ids,
            self.n_nodes,
            vec,
            self.dof,
            self.elements_size,
            self.constraints,
            self.mat_vec
        )
        return self.mat_vec

    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)

    def _matmat(self, rho, mat, Cp=None):
        if Cp is None:
            if not self.has_con:
                return matmat_node_basis_parallel(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                )
            else:
                out = matmat_node_basis_parallel(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out
        else:
            if not self.has_con:
                return matmat_node_basis_parallel_(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp
                )
            else:
                out = matmat_node_basis_parallel_(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp,
                    self.constraints
                )
                out.indices[out.indices < 0] = 0
                return out

    def _rmatmat(self, rho, mat, Cp=None):
        return self._matmat(rho, mat, Cp)

    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, cp.ndarray):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError(
                        "Shape of the input vector does not match the number of nodes and dof."
                    )
            elif cp.sparse.issparse(rhs):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    if isinstance(rhs, cp.sparse.csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError(
                        "Shape of the input matrix does not match the number of nodes and dof."
                    )
            else:
                raise NotImplementedError(
                    "Only numpy arrays as vectors and scipy sparse matrices are supported."
                )
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")

class UniformStiffnessKernel(StiffnessKernel):
    def __init__(self, mesh: CuGeneralMesh):
        super().__init__()

        if not mesh.is_uniform:
            raise ValueError(
                "Mesh is not uniform, you should use GeneralStiffnessKernel instead."
            )
        self.mesh = mesh
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        
        self.dtype = mesh.dtype

        # move to gpu and cast appropriately
        self.nodes = cp.array(self.nodes, dtype=self.dtype, copy=False)
        self.elements = cp.array(self.elements, dtype=cp.int32, copy=False)
        self.Ks = cp.array(mesh.Ks, dtype=self.dtype, copy=False)
        
        self.elements_flat = self.elements.flatten()
        self.el_ids = cp.arange(self.elements.shape[0], dtype=cp.int32).repeat(
            self.elements.shape[1]
        )
        self.sorter = cp.argsort(self.elements_flat).astype(cp.int32)
        self.node_ids = cp.searchsorted(
            self.elements_flat,
            cp.arange(self.nodes.shape[0], dtype=cp.int32),
            sorter=self.sorter,
            side="left",
        ).astype(cp.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]

        self.has_been_constructed = False
        self.ptr = None
        self.indices = None
        self.data = None

        self.max_con_count = (self.elements_size * int(self.dof)) * cp.unique(
            self.elements_flat, return_counts=True
        )[1].max()

        self.constraints = cp.zeros(self.n_nodes * self.dof, dtype=cp.bool_)
        self.shape = (self.n_nodes * self.dof, self.n_nodes * self.dof)

        self.has_con = False
        self.idx_map = cp.arange(self.n_nodes * self.dof, dtype=cp.int32)

    def diagonal(self, rho=None):
        if rho is None and not self.has_rho:
            raise ValueError(
                "Rho has not been set. diagonal works only after setting rho or if rho is provided."
            )
        elif rho is None:
            self.diag =  get_diagonal_node_basis(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                self.rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag
        else:
            self.diag = get_diagonal_node_basis(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag

    def set_constraints(self, constraints):
        self.constraints[:] = False
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def add_constraints(self, constraints):
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def process_grad(self, U):
        self.dk = process_dk_full(
            self.Ks, self.elements_flat, U, self.dof, self.elements_size, self.dk
        )
        
        return self.dk

    def construct(self, rho):
        size = self.n_nodes * self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype)
            )
            self.ptr = self.CSR.indptr
        else:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype), self.ptr
            )
        self.has_been_constructed = True

        return self.CSR

    def _matvec(self, rho, vec):
        if not self.has_con:
            self.mat_vec = mat_vec_node_basis_parallel_full(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.elements_size,
                out=self.mat_vec
            )
            return self.mat_vec
        else:
            self.mat_vec = mat_vec_node_basis_parallel_full(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.elements_size,
                self.constraints,
                self.mat_vec
            )
            return self.mat_vec

    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)

    def _matmat(self, rho, mat, Cp=None):
        if Cp is None:
            if not self.has_con:
                return matmat_node_basis_full_parallel(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                )
            else:
                out = matmat_node_basis_full_parallel(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out
        else:
            if not self.has_con:
                return matmat_node_basis_full_parallel_(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp,
                )
            else:
                out = matmat_node_basis_full_parallel_(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out

    def _rmatmat(self, rho, mat, Cp=None):
        return self._matmat(rho, mat, Cp)

    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, cp.ndarray):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError(
                        "Shape of the input vector does not match the number of nodes and dof."
                    )
            elif cp.sparse.issparse(rhs):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    if isinstance(rhs, cp.sparse.csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError(
                        "Shape of the input matrix does not match the number of nodes and dof."
                    )
            else:
                raise NotImplementedError(
                    "Only cupy arrays and sparse matrices are supported."
                )
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")


class GeneralStiffnessKernel(StiffnessKernel):
    def __init__(self, mesh: CuGeneralMesh):
        super().__init__()

        if mesh.is_uniform:
            raise ValueError(
                "Mesh is uniform, you should use UniformStiffnessKernel instead."
            )
        self.mesh = mesh
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        
        self.dtype = mesh.dtype

        # Initialize lists for flattening
        elements_flat = cp.array(mesh.elements_flat, dtype=cp.int32, copy=False)
        element_sizes = cp.array(mesh.element_sizes, dtype=cp.int32, copy=False)
        K_flat = mesh.K_flat

        # Move everything to GPU with appropriate types
        self.nodes = cp.array(self.nodes, dtype=self.dtype, copy=False)
        self.elements_flat = cp.array(elements_flat, dtype=cp.int32, copy=False)
        self.K_flat = cp.array(K_flat, dtype=self.dtype, copy=False)

        # Calculate pointers
        self.K_ptr = cp.array(mesh.K_ptr, dtype=cp.int32, copy=False)
        self.elements_ptr = cp.array(mesh.elements_ptr, dtype=cp.int32, copy=False)

        self.el_ids = cp.arange(self.elements.shape[0], dtype=cp.int32).repeat(
            element_sizes.get().tolist()
        )
        self.sorter = cp.argsort(self.elements_flat).astype(cp.int32)
        self.node_ids = cp.searchsorted(
            self.elements_flat,
            cp.arange(self.nodes.shape[0], dtype=cp.int32),
            sorter=self.sorter,
            side="left",
        ).astype(cp.int32)
        self.n_nodes = self.nodes.shape[0]

        self.has_been_constructed = False
        self.ptr = None
        self.indices = None
        self.data = None

        self.max_con_count = (
            cp.diff(self.elements_ptr).max() * int(self.dof)
        ) * cp.unique(self.elements_flat, return_counts=True)[1].max()

        self.constraints = cp.zeros(self.n_nodes * self.dof, dtype=cp.bool_)

        self.shape = (self.n_nodes * self.dof, self.n_nodes * self.dof)

        self.has_con = False
        self.idx_map = cp.arange(self.n_nodes * self.dof, dtype=cp.int32)

    def diagonal(self, rho=None):
        if rho is None and not self.has_rho:
            raise ValueError(
                "Rho has not been set. diagonal works only after setting rho or if rho is provided."
            )
        elif rho is None:
            self.diag =  get_diagonal_node_basis_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                self.rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.constraints,
                self.diag
            )
            return self.diag
        else:
            self.diag = get_diagonal_node_basis_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.constraints,
                self.diag
            )
            return self.diag

    def set_constraints(self, constraints):
        self.constraints[:] = False
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def add_constraints(self, constraints):
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def process_grad(self, U):
        self.dk = process_dk_flat(
            self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, U, self.dof, self.dk
        )
        return self.dk

    def construct(self, rho):
        size = self.n_nodes * self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype)
            )
            self.ptr = self.CSR.indptr
        else:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype), self.ptr
            )
        self.has_been_constructed = True

        return self.CSR

    def _matvec(self, rho, vec):
        if not self.has_con:
            self.mat_vec = mat_vec_node_basis_parallel_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                out=self.mat_vec
            )
            return self.mat_vec
        else:
            self.mat_vec = mat_vec_node_basis_parallel_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.constraints,
                self.mat_vec
            )
            return self.mat_vec

    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)

    def _matmat(self, rho, mat, Cp=None):
        if Cp is None:
            if not self.has_con:
                return matmat_node_basis_flat_parallel(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    self.max_con_count,
                )
            else:
                out = matmat_node_basis_flat_parallel(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out
        else:
            if not self.has_con:
                return matmat_node_basis_flat_parallel_(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    Cp,
                )
            else:
                out = matmat_node_basis_flat_parallel(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out

    def _rmatmat(self, rho, mat, Cp=None):
        return self._matmat(rho, mat, Cp)

    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, cp.ndarray):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError(
                        "Shape of the input vector does not match the number of nodes and dof."
                    )
            elif cp.sparse.issparse(rhs):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    if isinstance(rhs, cp.sparse.csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError(
                        "Shape of the input matrix does not match the number of nodes and dof."
                    )
            else:
                raise NotImplementedError(
                    "Only cupy arrays and sparse matrices are supported."
                )
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")
