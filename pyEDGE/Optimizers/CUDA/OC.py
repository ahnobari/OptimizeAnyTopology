import cupy as np
from .._optimizer import Optimizer
from ...Problem._problem import Problem
import time

class OC(Optimizer):
    def __init__(self, problem: Problem, move: float = 0.2, change_tol=1e-4, fun_tol=1e-6, timer=False):
        super().__init__(problem)

        if not problem.is_independent():
            raise ValueError("OC optimizer requires a problem with independent constraints.")

        self.last_desvars = np.copy(problem.get_desvars())
        self.last_f = problem.f()
                
        self.m = problem.m()
        self.ocP = None
        self.lambda_map = problem.constraint_map()
        self.move = move
        self.bounds = problem.bounds()
        self.change = np.inf
        self.change_f = np.inf
        self.change_tol = change_tol
        self.fun_tol = fun_tol
        self.timer = timer
        
        self.iteration = 0
        
        self.iteration = 0
            
    def _OCP(self):
        desvars = self.problem.get_desvars()
        dg = self.problem.nabla_g()
        df = self.problem.nabla_f()
        
        if self.m > 1:
            ocP = desvars * np.nan_to_num(np.sqrt(np.maximum(-df / dg.sum(axis=0), 0)), nan=0)
        else:
            ocP = desvars * np.nan_to_num(np.sqrt(np.maximum(-df / dg, 0)), nan=0)
            
        if np.abs(ocP).sum() == 0:
            ocP = np.ones_like(ocP) * 1e-3
        self.ocP = ocP
        
    
    def iter(self):
        if self.timer:
            start_time = time.time()
            
        self._OCP()
        
        desvars = self.problem.get_desvars()
        xU = np.clip(desvars + self.move, self.bounds[0], self.bounds[1])
        xL = np.clip(desvars - self.move, self.bounds[0], self.bounds[1])
        
        l1 = 1e-9 * np.ones(self.m, dtype=desvars.dtype)
        l2 = 1e9 * np.ones(self.m, dtype=desvars.dtype)
        
        while np.any((l2 - l1) / (l2 + l1) > 1e-8):
            l_mid = (l1 + l2) / 2
            
            if self.m > 1:
                l_mid_adjusted = (l_mid.reshape(1, -1) @ self.lambda_map).reshape(-1)
            else:
                l_mid_adjusted = l_mid
            
            desvars_new = np.maximum(
                self.bounds[0], np.maximum(xL, np.minimum(1.0, np.minimum(xU, self.ocP / l_mid_adjusted)))
            )

            valids = self.problem.g(desvars_new) <= 0.

            l2[valids] = l_mid[valids]
            l1[~valids] = l_mid[~valids]
        
        self.iteration += 1
        
        if self.timer:
            end_time = time.time()
        
        self.problem.set_desvars(desvars_new)
        self.change = np.linalg.norm(self.last_desvars - desvars_new)
        self.change_f = np.abs((self.problem.f()-self.last_f)/self.problem.f())
        
        self.last_f = self.problem.f()
        self.last_desvars = desvars_new.copy()
        
        if self.timer:
            return end_time - start_time
        
    def converged(self, *args, **kwargs):
        if not self.problem.is_terminal():
            return False
        elif self.change <= self.change_tol and self.change_f <= self.fun_tol:
            return True
        else:
            return False

    def logs(self):
        problem_logs = self.problem.logs()
        return{
            'objective': float(self.last_f),
            'variable change': float(self.change),
            'function change': float(self.change_f),
            **problem_logs
        }