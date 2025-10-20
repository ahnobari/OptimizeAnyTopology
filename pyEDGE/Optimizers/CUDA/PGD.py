import cupy as np
from .._optimizer import Optimizer
from ...Problem._problem import Problem
import time

class PGD(Optimizer):
    def __init__(self, 
                 problem: Problem,
                 change_tol=1e-4,
                 fun_tol=1e-6,
                 maxiter_N=50,
                 tol_B=1e-8,
                 tol_N=1e-6,
                 C=1e12,
                 fall_back_move=0.2,
                 alpha_max=1e2,
                 relaxation=1.0,
                 warmup_iter=50,
                 timer = False):
        super().__init__(problem)

        self.last_desvars = np.copy(problem.get_desvars())
        self.last_f = problem.f()
        self.last_nabla_f = self.problem.nabla_f().copy()
                
        self.m = problem.m()
        self.lambda_map = problem.constraint_map()
        self.bounds = problem.bounds()
        self.change = np.inf
        self.change_f = np.inf
        self.change_tol = change_tol
        self.fun_tol = fun_tol
        self.tol_B = tol_B
        self.tol_N = tol_N
        self.C = C
        self.fall_back_move = fall_back_move
        self.alpha_max = alpha_max
        self.maxiter_N = maxiter_N
        self.timer = timer
        self.relaxation = relaxation
        self.warmup_iter = warmup_iter
        
        self.iteration = 0
        
        self.is_independent = problem.is_independent()
        
        desvars = self.problem.get_desvars()
        desvars_new = desvars.copy()
        dg = self.problem.nabla_g()
        desvars_new = self.project_to_feasible(desvars_new, desvars, dg)
        self.problem.set_desvars(desvars_new)
    
    def alpha(self, desvars, df):
        
        ndf = max(np.linalg.norm(df)/np.sqrt(self.problem.N()), 1e-6)
        
        if self.iteration == 0:
            alpha = self.fall_back_move/np.linalg.norm(df,ord=np.inf)
            d = -df
        elif self.problem.g().max() > self.tol_N and self.iteration>self.warmup_iter:
            alpha = self.fall_back_move/np.linalg.norm(df,ord=np.inf)
            d = -df
        else:
            s = self.last_desvars - desvars
            y = self.last_nabla_f - df
            sy = np.dot(s, y)

            if sy > 1e-6:
                alpha = np.dot(s, s) / sy
                L = np.nan_to_num(np.linalg.norm(y) / np.linalg.norm(s), nan=0)
                alpha = np.clip(alpha, 1e-6, min(2 * self.relaxation/ (L + 1e-6), self.alpha_max))
            else:
                if np.allclose(y, 0):
                    # perfect linear behaviour
                    L = np.linalg.norm(df) * (np.linalg.norm(s) / np.sqrt(self.problem.N())) # Adjust step size based on projected change in variables
                    alpha = self.relaxation / (L + 1e-6)
                else:
                    L = np.nan_to_num(np.linalg.norm(y) / np.linalg.norm(s), nan=0)

                    alpha = min(self.relaxation / (L + 1e-6), self.alpha_max)

            d = -df
            
        return alpha, d
    
    def project_to_feasible(self, desvars_new, desvars, dg):
        
        if self.is_independent or self.m == 1:
            l1 = 0 * np.ones(self.m, dtype=desvars.dtype)
            l2 = -1e12 * np.ones(self.m, dtype=desvars.dtype)
            
            while np.any((l2 - l1) / (l2 + l1) > self.tol_B):
                l_mid = (l1 + l2) / 2
                
                if self.m > 1:
                    d_new = np.clip(desvars_new + (l_mid.reshape(1, -1) @ dg).reshape(-1), self.bounds[0], self.bounds[1])
                else:
                    d_new = np.clip(desvars_new + l_mid * dg, self.bounds[0], self.bounds[1])
                
                valids = self.problem.g(d_new) <= 0.
                l2[valids] = l_mid[valids]
                l1[~valids] = l_mid[~valids]

            desvars_new = d_new
        
        else:
            # First check if single active constraint solution exists
            l1 = 0 * np.ones(self.m, dtype=desvars.dtype)
            l2 = -1e12 * np.ones(self.m, dtype=desvars.dtype)
            G = self.problem.g(desvars)
            single_con_found = False
            
            while np.any((l2 - l1) / (l2 + l1) > self.tol_B):
                l_mid = (l1 + l2) / 2
                d_news = np.clip(desvars_new.reshape(1, -1) + l_mid.reshape(-1,1) * dg, self.bounds[0], self.bounds[1])
                diff = ((desvars.reshape(1, -1) - d_news) * dg).sum(axis=1)
                
                valids = G - diff <= 0.
                l2[valids] = l_mid[valids]
                l1[~valids] = l_mid[~valids]
                
            for i in range(self.m):
                if np.all(self.problem.g(d_news[i]) <= self.tol_N):
                    desvars_new = d_news[i]
                    single_con_found = True
                    break

            if not single_con_found:
                l = l_mid
                for _ in range(self.maxiter_N):
                    Phi, J_Phi, d_new = self._Phi(l, dg, desvars_new, J=True)
                    
                    if np.all(Phi <= self.tol_N):
                        break
                        
                    Delta = np.linalg.solve(J_Phi, Phi)
                    
                    alpha, l, Phi = self.wolfe_line_search(l, Delta, dg, desvars_new)
                    
                desvars_new = np.clip(desvars_new + (l.reshape(1, -1) @ dg).reshape(-1), self.bounds[0], self.bounds[1])
                
        return desvars_new
    
    def iter(self):
        if self.timer:
            start_time = time.time()
        desvars = self.problem.get_desvars()
        dg = self.problem.nabla_g()
        df = self.problem.nabla_f()
        f = self.problem.f()
        
        alpha, d = self.alpha(desvars, df)
            
        self.last_nabla_f = df.copy()
        self.d_last = d.copy()
        self.last_desvars = desvars.copy()
        self.last_f = f
        
        desvars_new = desvars + alpha * d
        self.iteration += 1
        
        desvars_new = self.project_to_feasible(desvars_new, desvars, dg)
        
        if self.timer:
            end_time = time.time()
        self.problem.set_desvars(desvars_new)
        self.change = np.linalg.norm(self.last_desvars - desvars_new)
        self.change_f = np.abs((self.problem.f()-self.last_f)/self.problem.f())
        
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
        
    
    def _Phi(self, l, dg, desvars_new, J=False):
        d_new = desvars_new + (l.reshape(1, -1) @ dg).reshape(-1)
        
        if J:
            non_saturated = np.logical_and(
                d_new > self.bounds[0],
                d_new < self.bounds[1]
            ).astype(desvars_new.dtype)
        
        d_new = np.clip(d_new, self.bounds[0], self.bounds[1])
            
        h = self.problem.g(d_new) + 2 * l / self.C
        
        if J:
            J_h = (dg * non_saturated.reshape(1,-1)) @ dg.T + 2 * np.eye(self.m, dtype=desvars_new.dtype)/self.C
            
        Phi = (h>0) * h + (h<=0) * (-l)
        
        if J:
            D_Phi = np.diag(h > 0).astype(desvars_new.dtype)
            J_Phi = D_Phi @ J_h - np.eye(self.m, dtype=desvars_new.dtype) + D_Phi
            
            return Phi, J_Phi, d_new
        
        return Phi, d_new
    
    def wolfe_line_search(self, l, delta, dg, desvars_new, c1=1e-4, c2=0.9, max_iter=10):
        """
        Wolfe conditions line search for Newton step
        """
        # Initial merit function and gradient
        Phi0, J_Phi, _ = self._Phi(l, dg, desvars_new, J=True)
        merit0 = 0.5 * np.dot(Phi0, Phi0)
        
        # Gradient of merit function w.r.t. l: grad_merit = J_Phi^T @ Phi
        grad_merit0 = J_Phi.T @ Phi0
        directional_deriv = -np.dot(grad_merit0, delta)
        
        alpha = 1.0
        
        for i in range(max_iter):
            l_new = l - alpha * delta
            
            # Compute new merit function
            Phi_new, J_Phi_new, _ = self._Phi(l_new, dg, desvars_new, J=True)
            merit_new = 0.5 * np.dot(Phi_new, Phi_new)
            
            # Armijo condition (sufficient decrease)
            if merit_new <= merit0 + c1 * alpha * directional_deriv:
                
                # Compute gradient for curvature condition
                grad_merit_new = J_Phi_new.T @ Phi_new
                new_directional_deriv = -np.dot(grad_merit_new, delta)
                
                # Wolfe curvature condition
                if new_directional_deriv >= c2 * directional_deriv:
                    return alpha, l_new, Phi_new
            
            alpha *= 0.5
        
        # Fallback: return last step
        return alpha, l_new, Phi_new