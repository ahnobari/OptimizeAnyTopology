from ...core.CPU._geom import generate_structured_mesh
import numpy as np
from ...physics._physx import Physx
from ...physics.LinearElasticity import LinearElasticity
import logging
logger = logging.getLogger(__name__)
from ..commons._mesh import Mesh, StructuredMesh

class StructuredMesh2D(StructuredMesh):
    def __init__(self, nx, ny, lx, ly, dtype=np.float64, physics: Physx = LinearElasticity(E=1.0, nu=1/3)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.lx = lx
        self.ly = ly
        self.nel = np.array([nx, ny], dtype=np.int32)
        self.dim = np.array([lx, ly], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        
        K = physics.K(self.nodes[self.elements[0]])
        self.K_single = K.astype(dtype)
        
        self.locals = physics.locals(self.nodes[self.elements[0]])
        for i in range(len(self.locals)):
            if isinstance(self.locals[i], np.ndarray):
                self.locals[i] = self.locals[i].astype(dtype)
        
        self.A_single = np.array([physics.volume(self.nodes[self.elements[0]])], dtype=dtype)

        self.As = self.A_single
        self.volume = self.A_single[0] * self.nelx * self.nely

        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype
        
        self.physics = physics
        
        self.centroids = np.meshgrid(
            np.linspace(self.dx/2, self.lx - self.dx/2, self.nelx, dtype=dtype),
            np.linspace(self.dy/2, self.ly - self.dy/2, self.nely, dtype=dtype),
            indexing='ij'
        )
        self.centroids = np.stack(self.centroids, axis=-1).reshape(-1, 2)
        
class StructuredMesh3D(StructuredMesh):
    def __init__(self, nx, ny, nz, lx, ly, lz, dtype=np.float64, physics: Physx = LinearElasticity(E=1.0, nu=1/3)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.nelz = nz
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.nel = np.array([nx, ny, nz], dtype=np.int32)
        self.dim = np.array([lx, ly, lz], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        self.dz = lz / nz

        K = physics.K(self.nodes[self.elements[0]])
        self.K_single = K.astype(dtype)

        self.locals = physics.locals(self.nodes[self.elements[0]])
        for i in range(len(self.locals)):
            if isinstance(self.locals[i], np.ndarray):
                self.locals[i] = self.locals[i].astype(dtype)

        self.A_single = np.array([physics.volume(self.nodes[self.elements[0]])], dtype=dtype)

        self.As = self.A_single
        
        self.volume = self.A_single[0] * self.nelx * self.nely * self.nelz
        
        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype
        
        self.physics = physics
        
        self.centroids = np.meshgrid(
            np.linspace(self.dx/2, self.lx - self.dx/2, self.nelx, dtype=dtype),
            np.linspace(self.dy/2, self.ly - self.dy/2, self.nely, dtype=dtype),
            np.linspace(self.dz/2, self.lz - self.dz/2, self.nelz, dtype=dtype),
            indexing='ij'
        )
        self.centroids = np.stack(self.centroids, axis=-1).reshape(-1, 3)
        
class GeneralMesh:
    def __init__(self, nodes, elements, dtype=np.float64, physics: Physx = LinearElasticity(E=1.0, nu=1/3)):
        self.nodes = nodes
        self.elements = elements
        
        self.centeroids = np.zeros((len(self.elements), self.nodes.shape[1]), dtype=dtype)
        
        self.is_uniform = True
        
        self.elements_flat = []
        self.element_sizes = np.zeros(len(self.elements), dtype=np.int32)
        self.K_flat = []
        size = len(self.elements[0])
        
        for i in range(self.elements.shape[0]):
            self.elements_flat += list(self.elements[i])
            self.element_sizes[i] = len(self.elements[i])
            self.centeroids[i] = np.mean(self.nodes[self.elements[i]], axis=0)
            if self.element_sizes[i] != size:
                self.is_uniform = False
    
        self.elements_flat = np.array(self.elements_flat, dtype=np.int32)
        self.elements_ptr = np.cumsum(self.element_sizes, dtype=np.int32)
        self.elements_ptr = np.concatenate(([0],self.elements_ptr),dtype=np.int32)

        # Clean Up Mesh To Remove Redundant Nodes
        logger.info("Checking Mesh ...")
        useful_idx = np.unique(self.elements_flat)
        if useful_idx.shape[0] != self.nodes.shape[0]:
            logger.info("Mesh has redundant nodes. Cleaning up ...")
            mapping = np.arange(self.nodes.shape[0])
            mapping = mapping[useful_idx]
            sorter = np.argsort(mapping).astype(np.int32)
            self.elements_flat = np.searchsorted(mapping, self.elements_flat, sorter=sorter).astype(np.int32)
        self.nodes = self.nodes[useful_idx]
        logger.info("Mesh Cleaned!")
        
        if self.is_uniform:
            self.Ks = physics.K(self.nodes[self.elements])
            self.Ks = self.Ks.astype(dtype)
            
            self.locals = physics.locals(self.nodes[self.elements])
            for i in range(len(self.locals)):
                if isinstance(self.locals[i], np.ndarray):
                    self.locals[i] = self.locals[i].astype(dtype)
                    
            self.As = physics.volume(self.nodes[self.elements]).astype(dtype)
            K_shape = self.Ks[0].shape
            
            self.dof = int(K_shape[0]/size)
            
            self.elements = np.array(self.elements, dtype=np.int32)
        else:
            n_locals = len(physics.locals(self.nodes[self.elements[0]]))
            self.K_flat = []
            self.locals_flat = [[] for _ in range(n_locals)]
            self.As = np.zeros((len(self.elements),1), dtype=dtype)
            
            self.K_ptr = np.zeros(len(self.elements)+1, dtype=np.int32)
            self.locals_ptr = [np.zeros(len(self.elements)+1, dtype=np.int32) for _ in range(n_locals)]
            
            for i in range(len(self.elements)):
                K_temp = physics.K(self.nodes[self.elements[i]])
                locals_temp = physics.locals(self.nodes[self.elements[i]])
                A_temp = physics.volume(self.nodes[self.elements[i]])
                self.K_flat += list(K_temp.flatten())
                for j in range(n_locals):
                    self.locals_flat[j] += list(locals_temp[j].flatten())
                    self.locals_ptr[j][i+1] = self.locals_ptr[j][i] + locals_temp[j].size
                self.As[i] = A_temp
                
                self.K_ptr[i+1] = self.K_ptr[i] + K_temp.size

            self.K_flat = np.array(self.K_flat, dtype=dtype)
            self.locals_flat = [np.array(self.locals_flat[j], dtype=dtype) for j in range(n_locals)]
            
            self.dof = int(K_temp.shape[0]/len(self.elements[-1]))
            
        self.volume = np.sum(self.As)
        
        self.dtype = dtype
        
        self.physics = physics