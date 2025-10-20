from ...core.CUDA._filter import (apply_filter_3D_cuda,
                                 apply_filter_2D_cuda,
                                 get_filter_weights_2D_cuda,
                                 get_filter_weights_3D_cuda,
                                 apply_filter_2D_transpose_cuda,
                                 apply_filter_3D_transpose_cuda)

from ._mesh import CuStructuredMesh2D, CuStructuredMesh3D, CuGeneralMesh
import cupy as cp
import numpy as np
from ..commons._filters import FilterKernel
from ..CPU._filters import filter_kernel_2D_general, filter_kernel_3D_general

class CuStructuredFilter3D(FilterKernel):
    def __init__(self, mesh: CuStructuredMesh3D, r_min):
        super().__init__()
        self.nelx = mesh.nel[0]
        self.nely = mesh.nel[1]
        self.nelz = mesh.nel[2]
        self.r_min = r_min
        self.shape = (self.nelx * self.nely * self.nelz, self.nelx * self.nely * self.nelz)
        
        self.dtype = mesh.dtype
        
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        self.scales = cp.array([dx, dy, dz], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = cp.arange(-n_neighbours, n_neighbours + 1, dtype=cp.int32)
        
        a, b, c = cp.meshgrid(offset_range, offset_range, offset_range, indexing='ij')
        offsets = cp.vstack([a.ravel(), b.ravel(), c.ravel()]).T
        offsets_adjusted = offsets * self.scales[None]
        
        distances = cp.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        
        self.weights = cp.array(weights, dtype=self.dtype)
        self.offsets = cp.array(offsets, dtype=cp.int32)
        
        self.normalizer = cp.zeros(self.shape[0], dtype=self.dtype)
        get_filter_weights_3D_cuda(self.nelx, self.nely, self.nelz, self.offsets, self.weights, self.normalizer)

    def _matvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_3D_cuda(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        return v_out

    def _rmatvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_3D_transpose_cuda(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights, self.normalizer)
        return v_out


class CuStructuredFilter2D(FilterKernel):
    def __init__(self, mesh: CuStructuredMesh2D, r_min):
        super().__init__()
        self.nelx = int(mesh.nelx)
        self.nely = int(mesh.nely)
        self.r_min = r_min
        self.shape = (self.nelx * self.nely, self.nelx * self.nely)
        self.dtype = mesh.dtype
        
        dx, dy = mesh.dx, mesh.dy
        self.scales = cp.array([dx, dy], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = cp.arange(-n_neighbours, n_neighbours + 1, dtype=cp.int32)
        
        a, b = cp.meshgrid(offset_range, offset_range, indexing='ij')
        offsets = cp.vstack([a.ravel(), b.ravel()]).T
        offsets_adjusted = offsets * self.scales[None]
        
        distances = cp.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        
        self.weights = cp.array(weights, dtype=self.dtype)
        self.offsets = cp.array(offsets, dtype=cp.int32)
        
        self.normalizer = cp.zeros(self.shape[0], dtype=self.dtype)
        get_filter_weights_2D_cuda(self.nelx, self.nely, self.offsets, self.weights, self.normalizer)
    
    def _matvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_2D_cuda(rho, v_out, self.nelx, self.nely, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_2D_transpose_cuda(rho, v_out, self.nelx, self.nely, self.offsets, self.weights, self.normalizer)
        return v_out
    
class CuGeneralFilter(FilterKernel):
    def __init__(self, mesh: CuGeneralMesh, r_min):
        super().__init__()
        self.dtype = mesh.dtype
        self.nd = mesh.nodes.shape[1]
        if self.nd == 2:
            self.kernel = filter_kernel_2D_general(mesh.elements, mesh.centeroids, r_min)
        else:
            self.kernel = filter_kernel_3D_general(mesh.elements, mesh.centeroids, r_min)
        
        self.kernel = cp.sparse.csr_matrix(self.kernel, dtype=self.dtype)
        
        self.shape = self.kernel.shape 
        
        self.weights = cp.empty(self.shape[0], dtype=self.dtype)
        
    def _matvec(self, rho):
        return (self.kernel @ rho).reshape(rho.shape)
    
    def _rmatvec(self, rho):
        return (self.kernel.T @ rho).reshape(rho.shape)