from sksparse.cholmod import cholesky
import numpy as np
from ..commons import Solver
from ...stiffness.CPU._FEA import StiffnessKernel
from ...stiffness.CPU import GeneralStiffnessKernel, UniformStiffnessKernel
from scipy.sparse.linalg import gmres as sp_gmres, cg as sp_cg, bicgstab as sp_bicgstab
from scipy.sparse.linalg import splu, spsolve
from scipy.linalg import get_lapack_funcs
from scipy.sparse.linalg import aslinearoperator
from ...core.CPU._mgm import apply_restriction, apply_prolongation, get_restricted_l0, get_restricted_l1p
from ...geom.CPU._mesh import StructuredMesh2D, StructuredMesh3D
import logging
from typing import Union
logger = logging.getLogger(__name__)

class CHOLMOD(Solver):
    def __init__(self, kernel: StiffnessKernel):
        super().__init__()
        
        if kernel.shape[0] > 3e6:
            raise ValueError("Currently we do not allow CHOLMOD for problem size bigger than 3M degrees of freedom. You can override this by passing a dummy kernel and overriding the kernel attribute.")
        
        self.kernel = kernel
        self.factor = None
        self.factorized = False
        self.n_desvars = len(kernel.elements)
        
    def reset(self):
        self.factor = None
        self.factorized = False
        
    def initialize(self):
        
        rho_temp = np.ones(self.n_desvars, dtype=self.kernel.nodes.dtype)
        K = self.kernel.construct(rho_temp)
        K = self.kernel.construct(rho_temp) # Twice to get correct pointers
        if self.kernel.has_cons:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:].tocsc()
            
        self.factor = cholesky(K)
        self.factorized = True
    
    def solve(self, rhs, rho=None, **kwargs):
        
        if not self.factorized:
            self.initialize()
        if not rho is None:
            K = self.kernel.construct(rho)
        else:
            if not self.kernel.has_rho:
                raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
            K = self.kernel.construct(self.kernel.rho)
        
        out = np.copy(rhs)
        
        if self.kernel.has_cons:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:].tocsc()
            rhs_ = rhs[self.kernel.non_con_map]
        
        self.factor.cholesky_inplace(K)
        
        out[self.kernel.non_con_map] = self.factor(rhs_)
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        return out, residual
    
def cg(A, b, x0=None, rtol=1e-5, maxiter=1000):
    tol = rtol
    normb = (b*b).sum()**0.5
    
    if x0 is None:
        x = b.copy()
    else:
        x = x0
    r = b - A.matvec(x)
    
    rho_prev, p = None, None
    
    for iteration in range(maxiter):
        if (r*r).sum()**0.5/normb < tol:
            return x, iteration
        z = r
        rho_cur = (r*z).sum()
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q = A.matvec(p)
        alpha = rho_cur / (p*q).sum()
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

    return x, iteration

def bicgstab(A, b, x0=None, rtol=1e-5, maxiter=1000):
    rhotol = np.finfo(b.dtype.char).eps**2
    omegatol = rhotol
    matvec = A.matvec
    
    tol = rtol
    normb = (b*b).sum()**0.5
    
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
        
    # Dummy values to initialize vars, silence linter warnings
    rho_prev, omega, alpha, p, v = None, None, None, None, None

    r = b - matvec(x) if x.any() else b.copy()
    rtilde = r.copy()

    for iteration in range(maxiter):
        if (r*r).sum()**0.5/normb < tol:  # Are we done?
            return x, 0

        rho = (rtilde*r).sum()
        if np.abs(rho) < rhotol:  # rho breakdown
            return x, -10

        if iteration > 0:
            if np.abs(omega) < omegatol:  # omega breakdown
                return x, -11

            beta = (rho / rho_prev) * (alpha / omega)
            p -= omega*v
            p *= beta
            p += r
        else:  # First spin
            s = np.empty_like(r)
            p = r.copy()

        phat = p
        v = matvec(phat)
        rv = (rtilde* v).sum()
        if rv == 0:
            return x, -11
        alpha = rho / rv
        r -= alpha*v
        s[:] = r[:]

        if (s*s).sum()**0.5/normb < tol:
            x += alpha*phat
            return x, 0

        shat = s
        t = matvec(shat)
        omega = (t*s).sum() / (t*t).sum()
        x += alpha*phat
        x += omega*shat
        r -= omega*t
        rho_prev = rho

    return x, maxiter

def gmres(A, b, x0=None, rtol=1e-5, maxiter=1000):
    
    Mb_nrm2 = (b*b).sum()**0.5
    bnrm2 = Mb_nrm2
    n = len(b)
    # ====================================================
    # =========== Tolerance control from gh-8400 =========
    # ====================================================
    # Tolerance passed to GMRESREVCOM applies to the inner
    # iteration and deals with the left-preconditioned
    # residual.
    ptol_max_factor = 1.
    ptol = Mb_nrm2 * min(ptol_max_factor, rtol / bnrm2)
    presid = 0.
    # ====================================================
    lartg = get_lapack_funcs('lartg', dtype=x.dtype)

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    eps = np.finfo(x.dtype.char).eps
    restart = 20
    # allocate internal variables
    v = np.empty([restart+1, n], dtype=x.dtype)
    h = np.zeros([restart, restart+1], dtype=x.dtype)
    givens = np.zeros([restart, 2], dtype=x.dtype)

    # legacy iteration count
    inner_iter = 0

    for iteration in range(maxiter):
        if iteration == 0:
            r = b - A.matvec(x)
            if (r*r).sum()**0.5/bnrm2 < rtol:
                return x, 0

        v[0, :] = r
        tmp = (v[0, :]*v[0, :]).sum()**0.5
        v[0, :] *= (1 / tmp)
        # RHS of the Hessenberg problem
        S = np.zeros(restart+1, dtype=x.dtype)
        S[0] = tmp

        breakdown = False
        for col in range(restart):
            av = A.matvec(v[col, :])
            w = av

            # Modified Gram-Schmidt
            h0 = (w*w).sum()**0.5
            for k in range(col+1):
                tmp = (v[k, :]* w).sum()
                h[col, k] = tmp
                w -= tmp*v[k, :]

            h1 = (w*w).sum()**0.5
            h[col, col + 1] = h1
            v[col + 1, :] = w[:]

            # Exact solution indicator
            if h1 <= eps*h0:
                h[col, col + 1] = 0
                breakdown = True
            else:
                v[col + 1, :] *= (1 / h1)

            # apply past Givens rotations to current h column
            for k in range(col):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = h[col, [k, k+1]]
                h[col, [k, k + 1]] = [c*n0 + s*n1, -s.conj()*n0 + c*n1]

            # get and apply current rotation to h and S
            c, s, mag = lartg(h[col, col], h[col, col+1])
            givens[col, :] = [c, s]
            h[col, [col, col+1]] = mag, 0

            # S[col+1] component is always 0
            tmp = -np.conjugate(s)*S[col]
            S[[col, col + 1]] = [c*S[col], tmp]
            presid = np.abs(tmp)
            inner_iter += 1

            if presid <= ptol or breakdown:
                break

        # Solve h(col, col) upper triangular system and allow pseudo-solve
        # singular cases as in (but without the f2py copies):
        # y = trsv(h[:col+1, :col+1].T, S[:col+1])

        if h[col, col] == 0:
            S[col] = 0

        y = np.zeros([col+1], dtype=x.dtype)
        y[:] = S[:col+1]
        for k in range(col, 0, -1):
            if y[k] != 0:
                y[k] /= h[k, k]
                tmp = y[k]
                y[:k] -= tmp*h[k, :k]
        if y[0] != 0:
            y[0] /= h[0, 0]

        x += y @ v[:col+1, :]

        r = b - A.matvec(x)
        rnorm = (r*r).sum()**0.5

        if rnorm/bnrm2 <= rtol:
            break
        elif breakdown:
            # Reached breakdown (= exact solution), but the external
            # tolerance check failed. Bail out with failure.
            break
        elif presid <= ptol:
            # Inner loop passed but outer didn't
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)

        ptol = presid * min(ptol_max_factor, rtol)

    info = 0 if (rnorm/bnrm2 <= rtol) else maxiter
    return x, info

class CG(Solver):
    def __init__(self, kernel: StiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, GeneralStiffnessKernel) or isinstance(kernel, UniformStiffnessKernel):
            if matrix_free is None:
                matrix_free = False
                logger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                logger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
        elif matrix_free is None:
            matrix_free = True
        
        self.matrix_free = matrix_free
        
        if not self.matrix_free:
            self.K = None
    
    def solve(self, rhs, rho=None, use_last=True):
        
        if self.kernel.has_rho or (not rho is None):
            if rho is not None:
                self.kernel.set_rho(rho)
            if use_last:
                if self.matrix_free:
                    out = cg(self.kernel, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = aslinearoperator(self.kernel.construct(self.kernel.rho))
                    out = cg(self.K, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = cg(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = aslinearoperator(self.kernel.construct(self.kernel.rho))
                    out = cg(self.K, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        r = rhs - self.kernel@out
        residual = (r*r).sum()**0.5/(rhs*rhs).sum()**0.5
        
        return out, residual
    
class BiCGSTAB(Solver):
    def __init__(self, kernel: StiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, GeneralStiffnessKernel) or isinstance(kernel, UniformStiffnessKernel):
            if matrix_free is None:
                matrix_free = False
                logger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                logger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
        elif matrix_free is None:
            matrix_free = True
        
        self.matrix_free = matrix_free
        
        if not self.matrix_free:
            self.K = None
    
    def solve(self, rhs, rho=None, use_last=True):
        
        if self.kernel.has_rho or (not rho is None):
            if rho is not None:
                self.kernel.set_rho(rho)
            if use_last:
                if self.matrix_free:
                    out = bicgstab(self.kernel, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_bicgstab(self.K, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = bicgstab(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_bicgstab(self.K, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        r = rhs - self.kernel@out
        residual = (r*r).sum()**0.5/(rhs*rhs).sum()**0.5
        
        return out, residual

class GMRES(Solver):
    def __init__(self, kernel: StiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, GeneralStiffnessKernel) or isinstance(kernel, UniformStiffnessKernel):
            if matrix_free is None:
                matrix_free = False
                logger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                logger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
        elif matrix_free is None:
            matrix_free = True
        
        self.matrix_free = matrix_free
        
        if not self.matrix_free:
            self.K = None
    
    def solve(self, rhs, rho=None, use_last=True):
        
        if self.kernel.has_rho or (not rho is None):
            if rho is not None:
                self.kernel.set_rho(rho)
            if use_last:
                if self.matrix_free:
                    out = gmres(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_gmres(self.K, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = gmres(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_gmres(self.K, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        self.last_x0 = out
        
        return out, residual
    
class SPLU(Solver):
    def __init__(self, kernel: StiffnessKernel):
        super().__init__()
        
        if kernel.shape[0] > 3e6:
            raise ValueError("Currently we do not allow SuperLU for problem size bigger than 3M degrees of freedom. You can override this by passing a dummy kernel and overriding the kernel attribute.")
        
        self.kernel = kernel
    
    def solve(self, rhs, rho=None, **kwargs):
        
        if not rho is None:
            K = self.kernel.construct(rho)
        else:
            if not self.kernel.has_rho:
                raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
            K = self.kernel.construct(self.kernel.rho)
        
        out = np.zeros_like(rhs)
        
        if self.kernel.has_cons:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:]
            rhs_ = rhs[self.kernel.non_con_map]
        
        LU = splu(K)
        
        out[self.kernel.non_con_map] = LU.solve(rhs_)
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        return out, residual
    
    
class SPSOLVE(Solver):
    def __init__(self, kernel: StiffnessKernel):
        super().__init__()
        
        if kernel.shape[0] > 3e6:
            raise ValueError("Currently we do not allow spsolve for problem size bigger than 3M degrees of freedom. You can override this by passing a dummy kernel and overriding the kernel attribute.")
        
        self.kernel = kernel
    
    def solve(self, rhs, rho=None, **kwargs):
        
        if not rho is None:
            K = self.kernel.construct(rho)
        else:
            if not self.kernel.has_rho:
                raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
            K = self.kernel.construct(self.kernel.rho)
        
        out = np.copy(rhs)
        
        if self.kernel.has_cons:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:]
            rhs_ = rhs[self.kernel.non_con_map]
        
        
        out[self.kernel.non_con_map] = spsolve(K, rhs_)
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        return out, residual

def opt_coef(k):
    # Quadrature nodes (x is a vector of length k)
    x = np.cos(np.arange(1, k+1) * np.pi / (k + 0.5))
    # Quadrature weights
    w = (1 - x) / (k + 0.5)
    
    # 4th-kind Chebyshev polynomials evaluated at x.
    # W is a (k x k) matrix.
    W = np.zeros((k, k))
    W[:, 0] = 1
    if k >= 2:
        W[:, 1] = 2 * x + 1
    for i in range(2, k):
        W[:, i] = 2 * x * W[:, i-1] - W[:, i-2]
    
    # Compute the roots of the optimal polynomial.
    r = opt_roots(k)
    
    # Transform nodes to [0, 1]
    lam = (1 - x) / 2
    # Compute p as the product over the k entries.
    # In MATLAB: p = prod(1 - lambda'./r, 1)' gives a k-by-1 vector.
    # Here, lam is length k and r is length k, so we form a (k x k) array and take the product over axis 0.
    p = np.prod(1 - (lam[None, :] / r[:, None]), axis=0)
    
    # Compute alpha = W' * (w .* p)
    alpha = np.dot(W.T, w * p)
    
    # Compute beta = 1 - cumsum((2*[0:k-1]'+1) .* alpha)
    beta = 1 - np.cumsum((2 * np.arange(k) + 1) * alpha)
    return beta

def opt_roots(k):
    def vars_func(r, x):
        # Here, r is a vector of length k and x is a vector (length n).
        # Compute p: for each entry in x, p[j] = prod(1 - x[j]/r[i]) for i=1,...,k.
        p = np.prod(1 - (x[None, :] / r[:, None]), axis=0)
        # Compute w = x/(1 - p^2)
        w_val = x / (1 - p**2)
        # f = sqrt(w) * p
        f_val = np.sqrt(w_val) * p
        # Compute q = sum_{i=1}^k 1/(x[j] - r[i]) for each j.
        q = np.sum(1 / (x[None, :] - r[:, None]), axis=0)
        # g = x * (1/(2*w) + q)
        g_val = x * (1/(2 * w_val) + q)
        # Compute ngp:
        # For each j: ngp[j] = sum_{i=1}^k ( p[j]^2 + r[i]/(x[j]-r[i]) )/(x[j]-r[i])
        ngp_val = np.sum(((p**2)[None, :] + r[:, None] / (x[None, :] - r[:, None])) / (x[None, :] - r[:, None]), axis=0)
        return w_val, f_val, g_val, ngp_val

    # Initial guesses:
    r = 0.5 - 0.5 * np.cos(np.arange(1, k+1) * np.pi / (k + 0.5))
    x = 0.5 - 0.5 * np.cos((0.5 + np.arange(1, k)) * np.pi / (k + 0.5))
    
    tol = 128 * 1e-12  # Tolerance for convergence
    dr = r.copy()
    drsize = 1.0
    
    # Outer loop: adjust r until convergence.
    while drsize > tol:
        dx = x.copy()
        dxsize = 1.0
        # Inner loop: adjust x until convergence.
        while dxsize > tol:
            dxsize = np.linalg.norm(dx, ord=np.inf)
            _, _, g, ngp = vars_func(r, x)
            dx = g / ngp
            x = x + dx
        
        # Append 1 to x to form x1.
        x1 = np.concatenate([x, [1]])
        w_val, f, _, _ = vars_func(r, x1)
        f0 = np.sqrt(0.5 / np.sum(1 / r))
        
        # Compute J elementwise.
        J = (f0**3 / (r**2)) + (w_val * np.abs(f))[:,None] / (r * (x1[:, None] - r))
        # Solve for dr elementwise.
        dr = -np.linalg.solve(J, f0 - np.abs(f))
        drsize = np.linalg.norm(dr, ord=np.inf)
        r = r + dr
    return r

class MultiGrid(Solver):
    def __init__(self, mesh: Union[StructuredMesh2D,StructuredMesh3D],
                 kernel: StiffnessKernel, maxiter=1000, tol=1e-5, n_smooth=3,
                 omega=0.5 , n_level = 3, cycle='W', w_level=1, coarse_solver='splu',
                 matrix_free=False, low_level_tol = 1e-8, low_level_maxiter=5000, min_omega=0.4, omega_boost=1.06):
        super().__init__()
        self.kernel = kernel
        self.mesh = mesh
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        self.n_smooth = n_smooth
        self.omega = omega
        self.max_omega = omega
        self.min_omega = max(omega/2, min_omega)
        self.d_omega = (self.max_omega - self.min_omega)
        self.omega_boost = omega_boost
        self.n_level = n_level
        self.cycle = cycle
        self.w_level = w_level
        self.matrix_free = matrix_free
        self.dof = self.mesh.dof
        self.coarse_solver = coarse_solver
        self.low_level_tol = low_level_tol
        self.low_level_maxiter = low_level_maxiter
        self.factor = None
        
        self.beta = opt_coef(self.n_smooth)
        
        if isinstance(self.w_level, int):
            self.w_level = [self.w_level]
        
        if self.coarse_solver in ['splu', 'spsolve'] and self.matrix_free:
            raise ValueError("Matrix free is not supported with splu solver use cg instead.")
        
        if self.coarse_solver not in ['cg', 'bicgstab', 'gmres', 'splu', 'spsolve', 'cholmod']:
            raise ValueError("Coarse solver not recognized.")
        
        if not self.matrix_free:
            self.ptr = None
            self.PRs = []
        else:
            raise NotImplementedError("Matrix free is not implemented yet.")

    def reset(self):
        self.ptr = None
    
    def _estimate_rho(self, A, D_inv, num_iterations=10):
        return 0
        """
        Estimate the spectral radius (largest eigenvalue) of D_inv @ A using the power method.
        
        Parameters:
            A              : The system matrix.
            D_inv          : The inverse of the diagonal of A (assumed to be provided as a vector or as a diagonal matrix).
            num_iterations : Number of iterations for the power method (default is 10).
        
        Returns:
            rho_est : An estimate of the spectral radius of D_inv @ A.
        """
        # Start with a random vector.
        x = np.random.rand(A.shape[0])
        x /= np.linalg.norm(x)

        for _ in range(num_iterations):
            # Apply the operator: D_inv * (A @ x)
            # If D_inv is a vector, use elementwise multiplication.
            x = D_inv * (A @ x)
            # Normalize the vector to avoid overflow/underflow issues.
            x /= np.linalg.norm(x)
        
        # After convergence, the Rayleigh quotient gives the dominant eigenvalue.
        Ax = A @ x
        rho_est = np.dot(x, D_inv * Ax)
        return rho_est
    
    def _chebyshev_smoother(self, x, b, A, D_inv, n_steps, rho):
        """
        Applies a Chebyshev polynomial smoother based on Chebyshev polynomials of the fourth kind.
        
        Instead of applying a fixed damped Jacobi update repeatedly, this smoother uses a
        three-term recurrence to compute a correction that more effectively reduces the
        high-frequency error components.

        Parameters:
            x       : Current solution vector.
            b       : Right-hand side vector.
            A       : System matrix.
            D_inv   : Inverse of the diagonal of A (or a suitable preconditioner).
            n_steps : Number of Chebyshev smoothing steps to perform.
            rho     : Estimate of the spectral radius of (D_inv @ A). This scales the spectrum to [0,1].

        Returns:
            x       : Updated solution vector after applying the Chebyshev smoother.
        """
        # Initialize the correction vector (z) to zero.
        z = np.zeros_like(x)
        
        # Loop through each Chebyshev smoothing step.
        for k in range(1, n_steps + 1):
            residual = b - A @ x
            # Compute the scaling factors based on the current step k.
            # factor2 scales the new correction term.
            factor2 = (8 * k - 4) / (2 * k + 1)

            if k == 1:
                # For the first step, z is simply the scaled correction.
                z = factor2 * (1 / rho) * (D_inv * residual)
            else:
                # For subsequent steps, combine the previous correction (scaled by factor1)
                # with the new term.
                factor1 = (2 * k - 3) / (2 * k + 1)
                z = factor1 * z + factor2 * (1 / rho) * (D_inv * residual)
            
            # Update the current solution with the computed correction.
            x = x + z

        return x
    
    def _jacobi_smoother(self, x, b, A, D_inv, n_step):
        for i in range(n_step):
            x += self.omega * D_inv * (b - A @ x)
        return x
    
    def _setup(self):
        self.levels = []
        
        D = 1/self.kernel.diagonal()
        rho = self._estimate_rho(self.kernel, D)
        self.levels.append((self.kernel, D, rho))
        
        for i in range(self.n_level):
            if i == 0 and not self.ptr is None:
                op = get_restricted_l0(self.mesh, self.kernel, Cp = self.ptr[0])
                D = 1/op.diagonal()
                rho = self._estimate_rho(op, D)
                self.levels.append((op, D, rho))
                
            elif i == 0:
                self.ptr = []
                if self.mesh.nel.shape[0] == 2:
                    nnz = np.ones((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*self.dof, dtype=np.int32) * 18
                    Cp = np.zeros((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                else:
                    nnz = np.ones((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*(self.mesh.nel[2]//2+1)*self.dof, dtype=np.int32) * 81
                    Cp = np.zeros((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*(self.mesh.nel[2]//2+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                
                self.ptr.append(np.copy(Cp))
                op = get_restricted_l0(self.mesh, self.kernel, Cp = Cp)
                D = 1/op.diagonal()
                rho = self._estimate_rho(op, D)
                self.levels.append((op, D, rho))
                
            elif len(self.ptr) <= i:
                
                if self.mesh.nel.shape[0] == 2:
                    nnz = np.ones((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*self.dof, dtype=np.int32) * 18
                    Cp = np.zeros((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                else:
                    nnz = np.ones((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*(self.mesh.nel[2]//(2**(i+1))+1)*self.dof, dtype=np.int32) * 81
                    Cp = np.zeros((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*(self.mesh.nel[2]//(2**(i+1))+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                
                self.ptr.append(np.copy(Cp))
                op = get_restricted_l1p(self.levels[-1][0], self.mesh.nel//(2**i), self.dof, Cp = Cp)
                D = 1/op.diagonal()
                rho = self._estimate_rho(op, D)
                self.levels.append((op, D, rho))
            else:
                op = get_restricted_l1p(self.levels[-1][0], self.mesh.nel//(2**i), self.dof, Cp = self.ptr[i])
                D = 1/op.diagonal()
                rho = self._estimate_rho(op, D)
                self.levels.append((op, D, rho))

    def _coarse_solver(self, K):
        if self.coarse_solver == 'cg':
            return lambda rhs: cg(K, rhs, rtol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'bicgstab':
            return lambda rhs: bicgstab(K, rhs, rtol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'gmres':
            return lambda rhs: gmres(K, rhs, rtol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'splu':
            SOLVER = splu(K)
            return lambda rhs: SOLVER.solve(rhs)
        elif self.coarse_solver == 'cholmod':
            K = K.tocsc()
            if self.factor is None:
                self.factor = cholesky(K, beta=1e-6)
            else:
                self.factor.cholesky_inplace(K)
            return lambda rhs: self.factor(rhs, beta=1e-6)
        elif self.coarse_solver == 'spsolve':
            return lambda rhs: spsolve(K, rhs)
        else:
            raise ValueError("Coarse solver not recognized.")
    
    def _multi_grid(self, x, b, level):
        if level == self.n_level:
            return self.coarse_solve(b)
        
        # presmooth
        A, D, rho = self.levels[level]
        
        self.omega = self.omega * (self.omega_boost)**(level)
        x = self._jacobi_smoother(x, b, A, D, self.n_smooth)
        # x = self._chebyshev_smoother(x, b, A, D, self.n_smooth, rho)
        self.omega = self.omega / (self.omega_boost)**(level)
        # residual
        r = b - A@x
        
        nel = (self.mesh.nel // 2**level).astype(np.int32)
        
        # restrict
        coarse_residual = apply_restriction(r,nel,self.dof)
        # coarse_residual = self.PRs[level][1] @ r
        
        # go to next level
        coarse_u = coarse_residual*0.0
        e = self._multi_grid(coarse_u, coarse_residual, level+1)
        
        # prolongate
        e = apply_prolongation(e,nel,self.dof)
        # e = self.PRs[level][0] @ e
        
        self.omega = self.omega * (self.omega_boost)**(level)
        e = self._jacobi_smoother(e, r, A, D, self.n_smooth)
        # e = self._chebyshev_smoother(e, r, A, D, self.n_smooth, rho)
        self.omega = self.omega / (self.omega_boost)**(level)
        
        x += e
        
        if self.cycle == 'w' and level in self.w_level:
            r = b - A@x
        
            # restrict
            coarse_residual = apply_restriction(r,nel,self.dof)
            # coarse_residual = self.PRs[level][1] @ r
            
            # go to next level
            coarse_u = coarse_residual*0.0
            e = self._multi_grid(coarse_u, coarse_residual, level+1)
            
            # prolongate
            e = apply_prolongation(e,nel,self.dof)
            # e = self.PRs[level][0] @ e
            
            self.omega = self.omega * (self.omega_boost)**(level)
            e = self._jacobi_smoother(e, r, A, D, self.n_smooth)
            # e = self._chebyshev_smoother(e, r, A, D, self.n_smooth, rho)
            self.omega = self.omega / (self.omega_boost)**(level)
            
            x += e
        
        return x
    
    def solve(self, rhs, rho=None, use_last=True):
        
        if not (self.kernel.has_rho or (not rho is None)):
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        if not rho is None:
            rho = self.kernel.set_rho(rho)
        
        if use_last and self.last_x0 is not None:
            x = self.last_x0
        else:
            x = np.copy(rhs)
            
        if self.matrix_free:
            self._mat_free_setup()
        else:
            self._setup()
        
        v_cycle = self._mat_free_multi_grid if self.matrix_free else self._multi_grid
        
        self.coarse_solve = self._coarse_solver(self.levels[-1][0])

        self.omega = self.min_omega
        r = rhs - self.kernel.dot(x)
        z = v_cycle(np.zeros_like(r), r, 0)
        p = z.copy()
        rho_old = (r*z).sum()
        norm_b = (rhs*rhs).sum()**0.5
        
        for i in range(self.maxiter):
            q = self.kernel.dot(p)
            alpha = rho_old / (p*q).sum()
            x += alpha * p
            r -= alpha * q
            
            norm_r = (r*r).sum()**0.5
            R = norm_r / norm_b
            if R < self.tol:
                break
            self.omega = self.min_omega + self.d_omega/2*np.exp((-np.clip(R,self.tol,1e-1)+self.tol)*500)
            z = v_cycle(np.copy(r), r,0)
            rho_new = (r*z).sum()
            beta = rho_new / rho_old
            p = z + beta * p
            rho_old = rho_new
        
        residual = norm_r / norm_b
        
        self.last_x0 = x
        
        del self.levels, self.coarse_solve
        
        
        return x, residual