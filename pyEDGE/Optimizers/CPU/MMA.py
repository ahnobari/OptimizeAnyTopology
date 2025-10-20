import numpy as np
from .._optimizer import Optimizer
from ...Problem._problem import Problem
from ...mma.CPU import mmasub
import time

class MMA(Optimizer):
    def __init__(self,
                 problem: Problem,
                 sub_tol=1e-7,
                 sub_maxiter=100,
                 change_tol=1e-4,
                 fun_tol=1e-6,
                 move=0.5,
                 timer=False):
        super().__init__(problem)
        
        self.last_desvars = np.copy(problem.get_desvars())
        self.last_f = problem.f()
        
        self.m = problem.m()
        self.ocP = None
        self.lambda_map = problem.constraint_map()
        self.bounds = problem.bounds()
        self.change = np.inf
        self.change_f = np.inf
        self.change_tol = change_tol
        self.fun_tol = fun_tol
        self.iteration = 0
        self.move = move
        self.sub_tol = sub_tol
        self.sub_maxiter = sub_maxiter
        
        self.x_1 = np.copy(problem.get_desvars())
        self.x_2 = np.copy(problem.get_desvars())
        self.low = np.ones([problem.get_desvars().shape[0], 1]) * self.bounds[0]
        self.upp = np.ones([problem.get_desvars().shape[0], 1]) * self.bounds[1]
        self.comp_base = self.problem.f()
        self.N = self.problem.N()
        
        self.timer = timer
     
    def iter(self):
        if self.timer:
            start_time = time.time()

        desvars = self.problem.get_desvars()
        dg = self.problem.nabla_g()
        df = self.problem.nabla_f()
        if self.iteration == 0:
            self.f0_val = self.problem.f()
        
        f_val = self.problem.g().reshape(-1,1)
        
        a = np.zeros([self.m,1])
        c = np.ones([self.m,1]) * 100000
        d = np.zeros([self.m,1])
        
        desvars_new, _, _, _, _, _, _, _, _, self.low, self.upp = mmasub(
            self.m,
            self.N,
            self.iteration+1,
            desvars.reshape(-1,1),
            np.ones_like(desvars).reshape(-1,1) * self.bounds[0],
            np.ones_like(desvars).reshape(-1,1) * self.bounds[1],
            self.x_1.reshape(-1,1),
            self.x_2.reshape(-1,1),
            0,
            df.reshape(-1,1),
            f_val.reshape(-1,1),
            dg,
            self.low,
            self.upp,
            1.0,
            a,
            c,
            d,
            move=self.move,
            sub_maxiter=self.sub_maxiter,
            sub_tol=self.sub_tol
        )
        self.x_2 = np.copy(self.x_1)
        self.x_1 = np.copy(desvars)
        
        self.iteration += 1
        
        if self.timer:
            end_time = time.time()
        
        # make small adjustments to avoid nan values
        if np.isnan(desvars_new).any():
            desvars_new = np.clip(desvars + self.sub_tol, self.bounds[0], self.bounds[1])

        self.problem.set_desvars(desvars_new.reshape(-1))
        self.change = np.linalg.norm(self.last_desvars - desvars_new.reshape(-1))
        self.change_f = np.abs((self.problem.f()-self.last_f)/self.problem.f())
        
        self.last_f = self.problem.f()
        self.last_desvars = self.problem.get_desvars().copy()
        
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
