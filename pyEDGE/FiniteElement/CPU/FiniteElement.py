from ..FiniteElement import FiniteElement as FE
from ...geom.CPU._mesh import StructuredMesh, GeneralMesh, StructuredMesh2D, StructuredMesh3D
from ...stiffness.CPU._FEA import StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel
from ...solvers.CPU._solvers import CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, MultiGrid
from ...visualizers._2d import plot_mesh_2D, plot_problem_2D, plot_field_2D
from ...visualizers._3d import plot_problem_3D, plot_mesh_3D, plot_field_3D
from typing import Optional, Union, List
from scipy.spatial import KDTree
import numpy as np


class FiniteElement(FE):
    def __init__(self, 
                 mesh: Union[StructuredMesh2D, StructuredMesh3D, GeneralMesh],
                 kernel: Union[StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel],
                 solver: Union[CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, MultiGrid]):
        super().__init__()
        
        self.mesh = mesh
        self.kernel = kernel
        self.solver = solver
        self.dtype = mesh.dtype
        
        self.rhs = np.zeros([self.kernel.shape[0]], dtype=self.dtype)
        self.d_rhs = np.zeros([self.kernel.shape[0]], dtype=self.dtype) + np.nan
        self.KDTree = None
        self.nel = len(self.mesh.elements)
        
        self.is_3D = self.mesh.nodes.shape[1] == 3
        self.dof = self.mesh.dof

    def add_dirichlet_boundary_condition(self,
                                        node_ids: Optional[np.ndarray] = None,
                                        positions: Optional[np.ndarray] = None,
                                        dofs: Optional[np.ndarray] = None,
                                        rhs: Union[float,np.ndarray] = 0):
        
        if node_ids is None and positions is None:
            raise ValueError("Either node_ids or positions must be provided.")
        if node_ids is not None and positions is not None:
            raise ValueError("Only one of node_ids or positions should be provided.")
        
        N_con = node_ids.shape[0] if node_ids is not None else positions.shape[0]
        
        if isinstance(rhs, np.ndarray) and rhs.shape[0] != N_con:
            raise ValueError("rhs must have the same length as node_ids or positions.")
        
        if node_ids is not None:
            if dofs is None:
                # assume all dofs are being set
                for i in range(self.mesh.dof):
                    cons = node_ids * self.mesh.dof + i
                    self.kernel.add_constraints(cons)
                    self.d_rhs[cons] = rhs[:, i] if isinstance(rhs, np.ndarray) else rhs + np.nan_to_num(self.d_rhs[cons], nan=0)
            else:
                if dofs.shape[0] != node_ids.shape[0] and dofs.shape[0] != 1:
                    raise ValueError("dofs must have the same length as node_ids.")
                elif dofs.shape[0] == 1:
                    dofs = np.tile(dofs, (node_ids.shape[0], 1))
                    
                for i in range(self.mesh.dof):
                    cons = node_ids[dofs[:, i]==1] * self.mesh.dof + i
                    self.kernel.add_constraints(cons)
                    
                    self.d_rhs[cons] = rhs[dofs[:, i]==1, i] if isinstance(rhs, np.ndarray) else rhs + np.nan_to_num(self.d_rhs[cons], nan=0)
                    
        else:
            if self.KDTree is None:
                self.KDTree = KDTree(self.mesh.nodes)
                
            _, node_ids = self.KDTree.query(positions)
            
            self.add_dirichlet_boundary_condition(node_ids=node_ids, dofs=dofs, rhs=rhs)
            
    def add_neumann_boundary_condition(self, **kwargs):
        raise NotImplementedError("Neumann boundary conditions are not implemented in this version of FiniteElement.")
    
    def add_point_forces(self, 
                         forces: np.ndarray,
                         node_ids: Optional[np.ndarray] = None,
                         positions: Optional[np.ndarray] = None):
        
        if node_ids is None and positions is None:
            raise ValueError("Either node_ids or positions must be provided.")
        if node_ids is not None and positions is not None:
            raise ValueError("Only one of node_ids or positions should be provided.")

        N_forces = node_ids.shape[0] if node_ids is not None else positions.shape[0]
        
        if (forces.shape[0] != N_forces and forces.shape[0] != 1) or forces.shape[1] != self.mesh.dof:
            raise ValueError("forces must have shape (N_forces, mesh.dof).")
        
        if node_ids is not None:
            if forces.shape[0] == 1:
                forces = np.tile(forces, (node_ids.shape[0], 1))
                
            for i in range(self.mesh.dof):
                self.rhs[node_ids * self.mesh.dof + i] += forces[:, i]
                
            # set dirichlet rhs
            dirichlet_dofs = np.logical_not(np.isnan(self.d_rhs))
            self.rhs[dirichlet_dofs] = self.d_rhs[dirichlet_dofs]
            
        else:
            if self.KDTree is None:
                self.KDTree = KDTree(self.mesh.nodes)
                
            _, node_ids = self.KDTree.query(positions)
            
            self.add_point_forces(forces=forces, node_ids=node_ids)
            
    def reset_forces(self):
        self.rhs[:] = 0
    
    def reset_dirichlet_boundary_conditions(self):
        self.kernel.set_constraints([])
        self.kernel.has_cons = False
        self.d_rhs[:] = np.nan
        
    def visualize_problem(self, ax=None, **kwargs):
        if self.is_3D:
            return plot_problem_3D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                **kwargs)
        else:
            return plot_problem_2D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                ax=ax,
                **kwargs)
            
    def visualize_density(self, rho, ax=None, **kwargs):
        if self.is_3D:
            return plot_problem_3D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                rho = rho,
                **kwargs)
        else:
            return plot_problem_2D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                ax=ax,
                rho = rho,
                **kwargs)
            
    def visualize_field(self, field, ax=None, rho=None, **kwargs):
        
        if self.is_3D:
            return plot_field_3D(
                self.mesh.nodes,
                self.mesh.elements,
                field,
                rho=rho,
                **kwargs)
        else:
            return plot_field_2D(
                self.mesh.nodes,
                self.mesh.elements,
                field,
                rho=rho,
                ax=ax,
                **kwargs)
    
    def solve(self, rho=None):
        
        if rho is not None and rho.shape[0] != self.nel:
            raise ValueError("rho must have the same length as the number of elements in the mesh.")
        
        if rho is None:
            rho = np.ones(self.nel, dtype=self.dtype)
        
        self.kernel.set_rho(rho)
        U, residual = self.solver.solve(self.rhs, use_last=True)
        
        return U, residual