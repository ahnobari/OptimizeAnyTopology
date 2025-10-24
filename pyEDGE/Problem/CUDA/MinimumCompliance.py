from .._problem import Problem
from ...FiniteElement.CUDA.FiniteElement import FiniteElement
from ...geom.CUDA._filters import (
    CuStructuredFilter2D as StructuredFilter2D, 
    CuStructuredFilter3D as StructuredFilter3D, 
    CuGeneralFilter as GeneralFilter)
from typing import Union, Callable
import cupy as np

class MinimumCompliance(Problem):
    def __init__(self,
                 FE: FiniteElement,
                 filter: Union[StructuredFilter2D, StructuredFilter3D, GeneralFilter],
                 E_mul: list[float] = [1.0],
                 void: float = 1e-6,
                 penalty: float = 3.0,
                 volume_fraction: list[float] = [0.25],
                 penalty_schedule:  Callable[[float, int], float] = None,
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
            self._nabla_g = np.tile(self._nabla_g, self.num_vars)
        elif self._nabla_g.shape[0] == 1 and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As[0]/self.FE.mesh.volume
        elif self._nabla_g.shape[0] != self.num_vars and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As/self.FE.mesh.volume

        self.is_3D = self.FE.mesh.nodes.shape[1] == 3
            
    def N(self):
        """
        This function returns the number of design variables.
        """
        return self.num_vars

    def m(self):
        """
        This function returns the number of constraints.
        """
        return 1 if self.is_single_material else len(self.E_mul)
    
    def is_independent(self):
        return True
    
    def constraint_map(self):
        """
        This function returns the mapping of constraints to design variables.
        :return: A list of lists, where each inner list contains the indices of design variables for each constraint.
        """
        if self.is_single_material:
            return 1
        else:
            mapping = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            
            for i in range(self.n_material):
                mapping[i, i*self.nel:(i+1)*self.nel] = 1
                
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
        self.desvars = np.ones(self.num_vars, dtype=self.dtype)
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

    def f(self, rho: np.ndarray = None):
        if rho is None:
            return self._f
        else:
            return self._f + rho.T @ self._nabla_f

    def nabla_f(self, rho: np.ndarray = None):
        return self._nabla_f
    
    def g(self, rho=None):
        if rho is None:
            vf = (self._nabla_g @ self.desvars.reshape(-1, 1))
            
            return vf.reshape(-1) - self.volume_fraction
        else:
            vf = (self._nabla_g @ rho.reshape(-1, 1))
            
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
        
        out = {
            'compliance': compliance,
            'Displacements': U
        }
        
        return out
    
    def logs(self):
        
        return {
            'iteration': int(self.iteration),
            'residual': float(self._residual)
        }