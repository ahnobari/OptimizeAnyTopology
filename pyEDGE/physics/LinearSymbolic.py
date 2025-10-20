from ._physx import Physx
import numpy as np
from sympy import symbols, Matrix, diff, lambdify, Symbol, Expr, sqrt
import sympy
from typing import Optional, List, Dict, Tuple
from .LinearElasticity import _triangle_element_area, _quadrilateral_element_area

class LinearSymbolic(Physx):
    def __init__(self, 
                 polygon=4,
                 dofs_per_node=2,
                 bilinear_form: Optional[Expr] = None,
                 linear_form: Optional[Expr] = None,
                 integration_order: int = 2,
                 Symbols: Optional[Tuple[List[Symbol], List[float]]] = None
                 ):
        '''
        Quick Docs:
        -----------
        This class sets up the symbolic framework for 2D linear finite elements.
        It currently supports 3-node triangles and 4-node quadrilaterals.
        
        Parameters:
        -----------
        polygon : int
            Number of nodes in the polygonal element (3 for triangle, 4 for quadrilateral).
        dofs_per_node : int
            Degrees of freedom per node (default is 2 for 2D problems). No assumptions on number of fields.
        bilinear_form : sympy expression, optional
            A user-defined bilinear form expression. This is assumed to be a sympy expression including terms specifically inline with:
                - u[0], v[0], u[1], v[1], ... : test and trial functions
                - du[0]_dx, dv[0]_dx, du[0]_dy, dv[0]_dy, ... : derivatives of test and trial functions w.r.t. x and y
                
            NOTE: Any symbols in these expressions will be kept as symbolic parameters in the final stiffness matrix lambdafied function.
        linear_form : sympy expression, optional
            A user-defined linear form expression. This is assumed to be a sympy expression including terms specifically inline with:
                - v[0], v[1], ... : test functions
                - dv[0]_dx, dv[0]_dy, ... : derivatives of test functions w.r.t. x and y
            Currently Neumann BCs are not supported in this framework so this ignored.
        
        integration_order : int
            Order of the Gaussian quadrature for numerical integration (default is 2).

        symbols : (List[Symbol], List[float])
            A list of symbols to include when lambdifying and their corresponding values. If your experssion includes constants such as E, nu etc, include these and their values.
        '''
        
        super().__init__()
        self.polygon = polygon
        if polygon not in [3, 4]:
            raise ValueError("Only 3-node triangles and 4-node quadrilaterals are currently supported.")
            
        self.polygon = polygon
        self.dofs_per_node = dofs_per_node
        self.nodes = list(range(1, polygon + 1))
        self.integration_order = integration_order
        
        # Natural coordinates
        self.xi, self.eta = symbols('xi eta')
        self.N, self.dN_dxi = self._shape_functions()
        
        # Nodal coordinates symbols
        self.coords = Matrix([Symbol(f'{c}{i}') for i in self.nodes for c in ['x', 'y']])
        
        x_coords = Matrix(self.coords[0::2])
        y_coords = Matrix(self.coords[1::2])

        # Compute physical coordinates as a function of natural coordinates
        x = self.N.T * x_coords
        y = self.N.T * y_coords
        
        # The coordinate vector in the physical domain
        X = Matrix([x[0], y[0]])
        
        # The Jacobian of the transformation
        J = X.jacobian([self.xi, self.eta])
        self.J = J
        self.detJ = J.det()
        
        self.dN_dx = self.J.inv() * self.dN_dxi.T
        
        # weak form substitutes
        self.d = Matrix([Symbol(f'd{i}_{j}') for i in self.nodes for j in range(self.dofs_per_node)])
        self.c = Matrix([Symbol(f'c{i}_{j}') for i in self.nodes for j in range(self.dofs_per_node)])
        self.subs_dict = {}
        for i in range(self.dofs_per_node):
            u_nodal = self.d[i::self.dofs_per_node]
            v_nodal = self.c[i::self.dofs_per_node]

            self.subs_dict[Symbol(f'u{i}')] = self.N.dot(u_nodal)
            self.subs_dict[Symbol(f'v{i}')] = self.N.dot(v_nodal)

            du_dx = self.dN_dx[0, :].dot(u_nodal)
            du_dy = self.dN_dx[1, :].dot(u_nodal)
            dv_dx = self.dN_dx[0, :].dot(v_nodal)
            dv_dy = self.dN_dx[1, :].dot(v_nodal)

            self.subs_dict[Symbol(f'du{i}_dx')] = du_dx
            self.subs_dict[Symbol(f'du{i}_dy')] = du_dy
            self.subs_dict[Symbol(f'dv{i}_dx')] = dv_dx
            self.subs_dict[Symbol(f'dv{i}_dy')] = dv_dy

        # hold the weak form expressions
        self.bilinear_form = bilinear_form
        self.linear_form = linear_form
        self.symbols = Symbols
        
        if bilinear_form is not None:
            self._lambdify()
        else:
            print("No bilinear form provided, stiffness matrix not generated. Use set_bilinear_form() to set one later.")
    
    def _shape_functions(self):
        if self.polygon == 3: # Linear Triangle
            N1 = 1 - self.xi - self.eta
            N2 = self.xi
            N3 = self.eta
            N = Matrix([N1, N2, N3])
        elif self.polygon == 4: # Bilinear Quadrilateral
            N1 = (1 - self.xi) * (1 - self.eta) / 4
            N2 = (1 + self.xi) * (1 - self.eta) / 4
            N3 = (1 + self.xi) * (1 + self.eta) / 4
            N4 = (1 - self.xi) * (1 + self.eta) / 4
            N = Matrix([N1, N2, N3, N4])
        
        dN_dxi = N.jacobian([self.xi, self.eta])
        return N, dN_dxi
    
    def set_bilinear_form(self, bilinear_form, symbols=None):
        self.bilinear_form = bilinear_form
        self.symbols = symbols
        self._lambdify()
    
    @staticmethod
    def _get_gauss_points(order):
        """
        Provides Gauss points and weights for a given order (for quads).
        """
        if order == 1:
            return [0], [2]
        if order == 2:
            p = 1 / sqrt(3)
            return [-p, p], [1, 1]
        if order == 3:
            p = sqrt(3/5)
            w1 = 5/9
            w2 = 8/9
            return [-p, 0, p], [w1, w2, w1]
        raise ValueError("Only Gauss orders 1, 2, and 3 are supported.")
    
    def _lambdify(self):
        print("Lambdifying the stiffness matrix, This may take a while...")
        integrand = self.bilinear_form.subs(self.subs_dict)
        integral_expr = sympy.S.Zero
        
        if self.polygon == 4:
            points, weights = self._get_gauss_points(self.integration_order)
            for i in range(self.integration_order):
                for j in range(self.integration_order):
                    xi_p, eta_p = points[i], points[j]
                    W = weights[i] * weights[j]
                    
                    subs_gauss = {self.xi: xi_p, self.eta: eta_p}
                    term = integrand.subs(subs_gauss) * self.detJ.subs(subs_gauss) * W
                    integral_expr += term
                    
        
        elif self.polygon == 3:
            xi_p, eta_p = sympy.S(1)/3, sympy.S(1)/3
            W = sympy.S(1)/2
            subs_gauss = {self.xi: xi_p, self.eta: eta_p}
            integral_expr = (integrand.subs(subs_gauss) * self.detJ.subs(subs_gauss)) * W
            
        Ke = sympy.zeros(self.polygon * self.dofs_per_node)
        for i, ci in enumerate(self.c):
            for j, dj in enumerate(self.d):
                Ke[i, j] = diff(integral_expr, ci, dj)
                
        all_symbols = list(self.coords) + self.symbols[0]
        Ke_numerical_func = lambdify(all_symbols, Ke, 'numpy')
        
        self.Ke = Ke
        self.Ke_lambdified = Ke_numerical_func
        
    def K(self, x0s : np.ndarray, *args):
        single=False
        if x0s.ndim == 2:
            x0s = x0s[None]
            single = True
            
        K_out = []
        for x_e in x0s:
            if len(args) > 0:
                K_out.append(self.Ke_lambdified(*x_e.flatten().tolist(), *args))
            elif self.symbols is not None:
                K_out.append(self.Ke_lambdified(*x_e.flatten().tolist(), *self.symbols[1]))
            else:
                K_out.append(self.Ke_lambdified(*x_e.flatten().tolist()))

        if single:
            return K_out[0]
        else:
            return np.array(K_out)
        
    def locals(self, *args, **kwargs):
        return []
    
    def volume(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 3 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 3 and x0s.shape[2] == 2):
            return _triangle_element_area(x0s)
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return _quadrilateral_element_area(x0s)
        else:
            raise ValueError("Invalid input shape")