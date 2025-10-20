from .._problem import Problem
from ...FiniteElement.CPU.FiniteElement import FiniteElement
from ...geom.CPU._mesh import StructuredMesh
from ...geom.CPU._filters import StructuredFilter2D, StructuredFilter3D, GeneralFilter
from ...core.CPU._ops import FEA_locals_node_basis_parallel, FEA_locals_node_basis_parallel_flat, FEA_locals_node_basis_parallel_full
from typing import Union, List
from scipy.spatial import KDTree
import numpy as np

class WeightDistributionMinimumCompliance(Problem):
    def __init__(self,
                 FE: FiniteElement,
                 filter: Union[StructuredFilter2D, StructuredFilter3D, GeneralFilter],
                 target_centroid: List[float],
                 maximum_distance: float,
                 E_mul: list[float] = [1.0],
                 void: float = 1e-6,
                 penalty: float = 3.0,
                 volume_fraction: list[float] = [0.25],
                 penalty_schedule: list[float] = None,
                 heavyside: bool = True,
                 beta: float = 2,
                 eta: float = 0.5):

        super().__init__()
        
        if len(E_mul) != len(volume_fraction):
            raise ValueError("E and volume_fraction must have the same length.")
        
        if len(E_mul) == 1:
            self.is_single_material = True
            self.E_mul = E_mul[0]
            self.volume_fraction = volume_fraction[0]
            self.n_material = 1
        else:
            self.is_single_material = False
            self.E_mul = np.array(E_mul, dtype=FE.dtype)
            self.volume_fraction = np.array(volume_fraction, dtype=FE.dtype)
            self.n_material = len(E_mul)

        self.void = void
        self.penalty = penalty
        self.penalty_schedule = penalty_schedule
        self.heavyside = heavyside
        self.beta = beta
        self.eta = eta
        self.filter = filter
        self.FE = FE
        self.dtype = FE.dtype
        
        self.iteration = 0
        self.desvars = None
        
        self._f = None
        self._g = None
        self._nabla_f = None
        self._nabla_g = self.FE.mesh.As/self.FE.mesh.volume
        self._residual = None
        
        self.num_vars = len(self.FE.mesh.elements) * len(E_mul)
        self.nel = len(self.FE.mesh.elements)
        
        if self._nabla_g.shape[0] == 1 and self.is_single_material:
            self._nabla_g = np.tile(self._nabla_g, self.num_vars).reshape(1, -1)
        elif self._nabla_g.shape[0] == 1 and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As[0]/self.FE.mesh.volume
        elif self._nabla_g.shape[0] != self.num_vars and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As/self.FE.mesh.volume
        self._nabla_g = np.concatenate([self._nabla_g, np.zeros((1, self.num_vars), dtype=self.dtype)], axis=0)
        
        self.is_3D = self.FE.mesh.nodes.shape[1] == 3
        
        if self.is_3D and len(target_centroid) != 3:
            raise ValueError("Target centroid must be a 3D point.")
        if not self.is_3D and len(target_centroid) != 2:
            raise ValueError("Target centroid must be a 2D point.")
        
        self.target_centroid = np.array(target_centroid, dtype=self.dtype)
        self.maximum_distance = maximum_distance
            
    def N(self):
        """
        This function returns the number of design variables.
        """
        return self.num_vars

    def m(self):
        """
        This function returns the number of constraints.
        """
        return 2 if self.is_single_material else len(self.E_mul) + 1
    
    def is_independent(self):
        return False
    
    def constraint_map(self):
        """
        This function returns the mapping of constraints to design variables.
        :return: A list of lists, where each inner list contains the indices of design variables for each constraint.
        """
        if self.is_single_material:
            return 1
        else:
            mapping = np.zeros((self.n_material + 1, self.num_vars), dtype=self.dtype)
            
            for i in range(self.n_material):
                mapping[i, i*self.nel:(i+1)*self.nel] = 1
                
            mapping[-1, :] = 1
                
            return mapping
    
    def bounds(self):
        """
        This function returns the bounds for the design variables.
        :return: A tuple containing the lower and upper bounds for the design variables.
        """
        return (0, 1.0)
            
    def visualize_problem(self, **kwargs):
        self.FE.visualize_problem(**kwargs)
    
    def visualize_solution(self, **kwargs):
        rho = self.get_desvars()
        if self.n_material > 1:
            rho = rho.reshape(self.n_material, -1).T
        self.FE.visualize_density(rho, **kwargs)
    
    def init_desvars(self):
        np.random.seed(0)
        self.desvars = np.random.uniform(size=self.num_vars)
        self.iteration = 0
        self._compute()
    
    def set_desvars(self, desvars: np.ndarray):
        if desvars.shape[0] != self.num_vars:
            raise ValueError(f"Expected {self.num_vars} design variables, got {desvars.shape[0]}.")
        
        self.desvars = desvars
        self._compute()
        self.iteration += 1
    
    def get_desvars(self):
        return self.desvars
    
    def penalize(self, rho: np.ndarray):
        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, self.iteration)

        if self.is_single_material:
            if self.heavyside:
                _rho = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                _rho = rho
            _rho = _rho**pen
            _rho = np.clip(_rho, self.void, 1.0)
            _rho = _rho*self.E_mul
            _rho = np.clip(_rho, self.void, None)
            
            return _rho
            
        else:
            if self.heavyside:
                _rho = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                _rho = rho
            rho_ = _rho**pen
            rho__ = 1 - rho_
            
            rho_ *= (
                rho__[
                    :,
                    np.where(~np.eye(self.n_material, dtype=bool))[1].reshape(
                        self.n_material, -1
                    ),
                ]
                .transpose(1, 0, 2)
                .prod(axis=-1)
                .T
            )

            E = (rho_ * self.E_mul[np.newaxis, :]).sum(axis=1)
            E = np.clip(E, self.void, None)
            
            return E

    def penalize_grad(self, rho: np.ndarray):
        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, self.iteration)
            
        if self.is_single_material:
            if self.heavyside:
                rho_heavy = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                df = pen * rho_heavy ** (pen - 1) * self.beta * (1 - np.tanh(self.beta * (rho-self.eta))**2) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                df = pen * rho ** (pen - 1)

            return df*self.E_mul
        
        else:
            if self.heavyside:
                rho_heavy = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                
                rho_ = pen * rho_heavy ** (pen - 1)
                rho__ = 1 - rho_heavy**pen
                rho___ = rho_heavy**pen

                d = rho__[np.newaxis, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, np.arange(self.n_material)] = rho___.T
                d = d[np.newaxis, :, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, :, np.arange(self.n_material)] = 1
                d = d.prod(axis=-1).transpose(0, 2, 1)

                mul = -rho_.T[:, :, np.newaxis].repeat(self.n_material, -1)
                mul[np.arange(self.n_material), :, np.arange(self.n_material)] *= -1

                d *= mul
                d = d @ self.E_mul[:, np.newaxis]
                
                df = d.squeeze().T * self.beta * (1 - np.tanh(self.beta * (rho-self.eta))**2) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                
                return df
            else:
                rho_ = pen * rho ** (pen - 1)
                rho__ = 1 - rho**pen
                rho___ = rho**pen

                d = rho__[np.newaxis, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, np.arange(self.n_material)] = rho___.T
                d = d[np.newaxis, :, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, :, np.arange(self.n_material)] = 1
                d = d.prod(axis=-1).transpose(0, 2, 1)

                mul = -rho_.T[:, :, np.newaxis].repeat(self.n_material, -1)
                mul[np.arange(self.n_material), :, np.arange(self.n_material)] *= -1

                d *= mul
                d = d @ self.E_mul[:, np.newaxis]
                
                df = d.squeeze().T
                
                return df
    
    def _compute(self):
        
        if self.is_single_material:
            rho = self.filter.dot(self.desvars)
        else:
            rho = np.copy(self.desvars).reshape(self.n_material, -1).T
            for i in range(self.n_material):
                rho[:, i] = self.filter.dot(rho[:, i])
        
        rho_ = self.penalize(rho)
        
        U, residual = self.FE.solve(rho_)
        
        compliance = self.FE.rhs.dot(U)

        df = self.FE.kernel.process_grad(U)

        if rho.ndim > 1:
            df = df.reshape(-1,1)
        
        dr = self.penalize_grad(rho) * df

        dr = dr.reshape(dr.shape[0], -1)

        for i in range(dr.shape[1]):
            dr[:, i] = self.filter._rmatvec(dr[:, i])
        
        self._f = compliance
        if self.is_single_material:
            self._nabla_f = dr.reshape(-1)
        else:
            self._nabla_f = dr.T.reshape(-1)
        self._residual = residual
        
        vf = self._nabla_g[:-1].squeeze() @ self.desvars.reshape(-1, 1)
        V = vf * self.FE.mesh.volume
        
        if self.is_single_material:
            mul = self.FE.mesh.centroids * self._nabla_g[0][:, None] * self.FE.mesh.volume
            cmm = (mul * self.desvars.reshape(-1, 1)).sum(0)
        else:
            mul = (self.FE.mesh.centroids * self._nabla_g[0, :self.nel, None])[None] * self.FE.mesh.volume
            cmm = (mul * self.desvars.reshape(self.n_material, -1, 1)).sum(0).sum(0)

        cm = cmm / V.sum()
        distance = np.square(cm - self.target_centroid).sum()
        
        self._g = np.concatenate([
            vf.reshape(-1) - self.volume_fraction,
            np.array([distance - self.maximum_distance**2], dtype=self.dtype)
        ]).reshape(-1)

        nabla_centroid = (2 * (cm - self.target_centroid)[None] * (V.sum() * mul.squeeze() - cmm[None] * self._nabla_g[0][:self.nel, None] * self.FE.mesh.volume) / (V.sum()**2)).sum(1)
        if self.is_single_material:
            self._nabla_g[-1] = nabla_centroid
        else:
            for i in range(self.n_material):
                self._nabla_g[-1, i*self.nel:(i+1)*self.nel] = nabla_centroid
                
        self._cm = cm

    def f(self, rho: np.ndarray = None):
        if rho is None:
            return self._f
        else:
            return self._f + rho.T @ self._nabla_f

    def nabla_f(self, rho: np.ndarray = None):
        return self._nabla_f
    
    def g(self, rho=None):
        if rho is None:
            return self._g
        else:
            # local linear approximation
            return self._g + (self._nabla_g @ (rho - self.desvars).reshape(-1, 1)).squeeze()
        # (rho - self.desvars).dot(self._nabla_g) - self.comp_limit
        return self._g
        if rho is None:
            V = (self._nabla_g @ self.desvars.reshape(-1, 1))
            vf = V / self.FE.mesh.volume
            
            return vf.reshape(-1) - self.volume_fraction
        else:
            V = (self._nabla_g @ rho.reshape(-1, 1))
            vf = V / self.FE.mesh.volume
            
            return vf.reshape(-1) - self.volume_fraction
        
    def nabla_g(self):
        return self._nabla_g

    def ill_conditioned(self):
        if self._residual >= 1e-2:
            return True
        else:
            return False
        
    def is_terminal(self):
        if self.penalty_schedule is not None:
            if self.penalty_schedule(self.penalty, self.iteration) == self.penalty:
                return True
            else:
                return False
        else:
            return True
        
    def logs(self):
        
        return {
            'iteration': int(self.iteration),
            'residual': float(self._residual)
        }
        
    def FEA(self, thresshold: bool = True):
        if self.desvars is None:
            raise ValueError("Design variables are not initialized. Call init_desvars() or set_desvars() first.")

        if thresshold:
            rho = (self.get_desvars()>0.5).astype(self.dtype) + self.void
            
        if not self.is_single_material:
            rho = rho.reshape(self.n_material, -1).T
            rho = (rho * self.E_mul[np.newaxis, :]).sum(axis=1)
        else:
            rho = rho * self.E_mul
        
        if hasattr(self.FE.solver, 'maxiter'):
            maxiter = self.FE.solver.maxiter + 0
            self.FE.solver.maxiter = maxiter * 4
            
       
        U,residual = self.FE.solve(rho)
        compliance = self.FE.rhs.dot(U)

        if residual > 1e-5:
            print(f"Solver residual is above 1e-5 ({residual:.4e}). Consider higher iterations (rerun this function and more iteration from prior solve will be applied).")
            
        if isinstance(self.FE.mesh, StructuredMesh):
            strain, stress, strain_energy = FEA_locals_node_basis_parallel(self.FE.mesh.K_single,
                                                                            self.FE.mesh.locals[0],
                                                                            self.FE.mesh.locals[1],
                                                                            self.FE.kernel.elements_flat,
                                                                            rho.shape[0],
                                                                            rho,
                                                                            U,
                                                                            self.FE.mesh.dof,
                                                                            self.FE.mesh.elements_size,
                                                                            self.FE.mesh.locals[1].shape[0])
        elif self.FE.mesh.is_uniform:
            strain, stress, strain_energy = FEA_locals_node_basis_parallel_full(self.FE.mesh.Ks,
                                                                                    self.FE.mesh.locals[0],
                                                                                    self.FE.mesh.locals[1],
                                                                                    self.FE.kernel.elements_flat,
                                                                                    rho.shape[0],
                                                                                    rho,
                                                                                    U,
                                                                                    self.FE.mesh.dof,
                                                                                    self.FE.mesh.elements.shape[1],
                                                                                    self.FE.mesh.locals[1].shape[1])
        else:
            B_size = (self.FE.mesh.locals_ptr[1][1]-self.FE.mesh.locals_ptr[1][0])//((self.FE.mesh.elements_ptr[1]-self.FE.mesh.elements_ptr[0])*self.FE.mesh.dof)
            strain, stress, strain_energy = FEA_locals_node_basis_parallel_flat(self.FE.mesh.K_flat,
                                                                                    self.FE.mesh.locals_flat[0],
                                                                                    self.FE.mesh.locals_flat[1],
                                                                                    self.FE.kernel.elements_flat,
                                                                                    self.FE.mesh.elements_ptr,
                                                                                    self.FE.mesh.K_ptr,
                                                                                    self.FE.mesh.locals_ptr[1],
                                                                                    self.FE.mesh.locals_ptr[0],
                                                                                    rho.shape[0],
                                                                                    rho,
                                                                                    U,
                                                                                    self.FE.mesh.dof,
                                                                                    B_size)
            
        if self.FE.mesh.nodes.shape[1] == 2:
            von_mises = np.sqrt(stress[:, 0] ** 2 + stress[:, 1] ** 2 - stress[:, 0] * stress[:, 1] + 3 * stress[:, 2] ** 2)
        else:
            von_mises = np.sqrt(0.5 * ((stress[:, 0] - stress[:, 1]) ** 2 + (stress[:, 1] - stress[:, 2]) ** 2 + (stress[:, 2] - stress[:, 0]) ** 2 + 6 * (stress[:, 3] ** 2 + stress[:, 4] ** 2 + stress[:, 5] ** 2)))

        
        out = {
            'strain': strain,
            'stress': stress,
            'strain_energy': strain_energy,
            'von_mises': von_mises,
            'compliance': compliance,
            'Displacements': U
        }
        
        return out
    
    def visualize_field(self, field, ax=None, thresshold=True, **kwargs):
        if thresshold:
            rho = (self.desvars > 0.5).astype(self.dtype)
        else:
            rho = None
            
        if not self.is_single_material and rho is not None:
            rho = rho.reshape(self.n_material, -1).T
            rho = (rho).sum(axis=1)>0

        self.FE.visualize_field(field, ax=ax, rho=rho)