from ...core.CPU._filter import (apply_filter_2D_parallel,
                                apply_filter_3D_parallel,
                                filter_kernel_2D_general,
                                filter_kernel_3D_general,
                                get_filter_weights_2D,
                                get_filter_weights_3D,
                                apply_filter_2D_parallel_transpose,
                                apply_filter_3D_parallel_transpose,
                                )
import numpy as np
from ..commons._filters import FilterKernel
from ._mesh import StructuredMesh2D, StructuredMesh3D, GeneralMesh

class StructuredFilter3D(FilterKernel):
    def __init__(self, mesh: StructuredMesh3D, r_min):
        super().__init__()
        self.dtype = mesh.dtype

        self.nelx = mesh.nelx
        self.nely = mesh.nely
        self.nelz = mesh.nelz
        self.r_min = r_min
        self.shape = (self.nelx * self.nely * self.nelz, self.nelx * self.nely * self.nelz)
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        self.scales = np.array([dx, dy, dz], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = np.arange(-n_neighbours, n_neighbours + 1, dtype=np.int32)
        
        a, b, c = np.meshgrid(offset_range, offset_range, offset_range, indexing='ij')
        offsets = np.vstack([a.ravel(), b.ravel(), c.ravel()]).T.astype(np.int32)
        offsets_adjusted = offsets * self.scales[None]
        
        distances = np.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        weights = weights.astype(self.dtype)
        
        self.weights = weights
        self.offsets = offsets
        
        self.normalizer = get_filter_weights_3D(self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        
    def _matvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_3D_parallel(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_3D_parallel_transpose(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights, self.normalizer)
        return v_out


class StructuredFilter2D(FilterKernel):
    def __init__(self, mesh: StructuredMesh2D, r_min):
        super().__init__()
        self.nelx = mesh.nel[0]
        self.nely = mesh.nel[1]
        self.r_min = r_min
        self.shape = (self.nelx * self.nely, self.nelx * self.nely)
        self.dtype = mesh.dtype
        
        dx, dy = mesh.dx, mesh.dy
        self.scales = np.array([dx, dy], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = np.arange(-n_neighbours, n_neighbours + 1, dtype=np.int32)
        
        a, b = np.meshgrid(offset_range, offset_range, indexing='ij')
        offsets = np.vstack([a.ravel(), b.ravel()]).T.astype(np.int32)
        offsets_adjusted = offsets * self.scales[None]
        
        distances = np.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        weights = weights.astype(self.dtype)
        
        self.weights = weights
        self.offsets = offsets
        
        self.normalizer = get_filter_weights_2D(self.nelx, self.nely, self.offsets, self.weights)
        
        
    def _matvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_2D_parallel(rho, v_out, self.nelx, self.nely, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_2D_parallel_transpose(rho, v_out, self.nelx, self.nely, self.offsets, self.weights, self.normalizer)
        return v_out

class GeneralFilter(FilterKernel):
    def __init__(self, mesh: GeneralMesh, r_min):
        super().__init__()
        self.dtype = mesh.dtype
        self.nd = mesh.nodes.shape[1]
        if self.nd == 2:
            self.kernel = filter_kernel_2D_general(mesh.elements, mesh.centeroids, r_min)
        else:
            self.kernel = filter_kernel_3D_general(mesh.elements, mesh.centeroids, r_min)
        
        self.kernel = self.kernel.astype(self.dtype)
        self.shape = self.kernel.shape 
        
        self.weights = np.empty(self.shape[0], dtype=self.dtype)
        
    def _matvec(self, rho):
        return (self.kernel @ rho).reshape(rho.shape)
    
    def _rmatvec(self, rho):
        return (self.kernel.T @ rho).reshape(rho.shape)