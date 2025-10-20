from ._physx import Physx
import numpy as np

class LinearElasticity(Physx):
    def __init__(self, E=1.0, nu=1./3., thickness=1.0, type='PlaneStress'):
        super().__init__()
        self.E = E
        self.nu = nu
        self.thickness = thickness
        if type == 'PlaneStress':
            self.type = 0
        else:
            self.type = 1

    def K(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 3 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 3 and x0s.shape[2] == 2):
            return _triangle_element_stiffness(x0s, E=self.E, NU=self.nu, e_type=self.type, t=self.thickness)[0]
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return _quadrilateral_element_stiffness(x0s, E=self.E, NU=self.nu, e_type=self.type, t=self.thickness)[0]
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 3):
            return _tetrahedron_element_stiffness(x0s, E=self.E, NU=self.nu)[0]
        elif (x0s.ndim == 2 and x0s.shape[0] == 8 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 8 and x0s.shape[2] == 3):
            return _hexahedron_element_stiffness(x0s, E=self.E, NU=self.nu)[0]
        else:
            raise ValueError("Invalid input shape")

    def locals(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 3 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 3 and x0s.shape[2] == 2):
            return list(_triangle_element_stiffness(x0s, E=self.E, NU=self.nu, e_type=self.type, t=self.thickness)[1:])
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return list(_quadrilateral_element_stiffness(x0s, E=self.E, NU=self.nu, e_type=self.type, t=self.thickness)[1:])
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 3):
            return list(_tetrahedron_element_stiffness(x0s, E=self.E, NU=self.nu)[1:])
        elif (x0s.ndim == 2 and x0s.shape[0] == 8 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 8 and x0s.shape[2] == 3):
            return list(_hexahedron_element_stiffness(x0s, E=self.E, NU=self.nu)[1:])
        else:
            raise ValueError("Invalid input shape")
    
    def volume(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 3 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 3 and x0s.shape[2] == 2):
            return _triangle_element_area(x0s)
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return _quadrilateral_element_area(x0s)
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 3):
            return _tetrahedron_element_volume(x0s)
        elif (x0s.ndim == 2 and x0s.shape[0] == 8 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 8 and x0s.shape[2] == 3):
            return _hexahedron_element_volume(x0s)
        else:
            raise ValueError("Invalid input shape")
    
    def neumann(self, x0s):
        pass


def _triangle_element_stiffness(x0s, E=1.0, NU=0.33, e_type=0, t=1.0):
    """
    This function computes the stiffness matrix for a batch of triangle elements given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (n_elements, 3, 2) or (3, 2) for single element
        E (float or np.array): Young's modulus. Can be scalar or array of shape (n_elements,)
        NU (float or np.array): Poisson's ratio. Can be scalar or array of shape (n_elements,)
        e_type (int): 0 for Plane Stress, 1 for Plane Strain
        t (float or np.array): Thickness of the element. Can be scalar or array of shape (n_elements,)

    Returns:
        K (np.array): Stiffness matrices. Shape (n_elements, 6, 6) or (6, 6) for single element
        D (np.array): Constitutive matrices. Shape (n_elements, 3, 3) or (3, 3) for single element
        B (np.array): B-Operators. Shape (n_elements, 3, 6) or (3, 6) for single element
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    E = np.atleast_1d(E)
    NU = np.atleast_1d(NU)
    t = np.atleast_1d(t)
    
    # Broadcast to match number of elements
    if E.size == 1:
        E = np.full(n_elements, E[0])
    if NU.size == 1:
        NU = np.full(n_elements, NU[0])
    if t.size == 1:
        t = np.full(n_elements, t[0])
    
    # Compute areas for all elements
    A = (
        x0s[:, 0, 0] * (x0s[:, 1, 1] - x0s[:, 2, 1])
        + x0s[:, 1, 0] * (x0s[:, 2, 1] - x0s[:, 0, 1])
        + x0s[:, 2, 0] * (x0s[:, 0, 1] - x0s[:, 1, 1])
    ) / 2
    
    # Check for negative areas
    if np.any(A < 0):
        negative_indices = np.where(A < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")
    
    # Compute beta and gamma coefficients for all elements
    betai = x0s[:, 1, 1] - x0s[:, 2, 1]  # shape: (n_elements,)
    betaj = x0s[:, 2, 1] - x0s[:, 0, 1]
    betam = x0s[:, 0, 1] - x0s[:, 1, 1]
    gammai = x0s[:, 2, 0] - x0s[:, 1, 0]
    gammaj = x0s[:, 0, 0] - x0s[:, 2, 0]
    gammam = x0s[:, 1, 0] - x0s[:, 0, 0]
    
    # Build B matrices for all elements
    B = np.zeros((n_elements, 3, 6))
    B[:, 0, 0] = betai
    B[:, 0, 2] = betaj
    B[:, 0, 4] = betam
    B[:, 1, 1] = gammai
    B[:, 1, 3] = gammaj
    B[:, 1, 5] = gammam
    B[:, 2, 0] = gammai
    B[:, 2, 1] = betai
    B[:, 2, 2] = gammaj
    B[:, 2, 3] = betaj
    B[:, 2, 4] = gammam
    B[:, 2, 5] = betam
    
    # Normalize by 2*A
    B = B / (2 * A[:, np.newaxis, np.newaxis])
    
    # Compute constitutive matrices
    D = np.zeros((n_elements, 3, 3))
    
    if e_type == 0:  # Plane Stress
        factor = E / (1 - NU * NU)
        D[:, 0, 0] = factor
        D[:, 0, 1] = factor * NU
        D[:, 1, 0] = factor * NU
        D[:, 1, 1] = factor
        D[:, 2, 2] = factor * (1 - NU) / 2
    else:  # Plane Strain
        factor = E / ((1 + NU) * (1 - 2 * NU))
        D[:, 0, 0] = factor * (1 - NU)
        D[:, 0, 1] = factor * NU
        D[:, 1, 0] = factor * NU
        D[:, 1, 1] = factor * (1 - NU)
        D[:, 2, 2] = factor * (1 - 2 * NU) / 2
    
    # Compute stiffness matrices: K = t * A * B^T @ D @ B
    # Using einsum for efficient batch matrix multiplication
    K = np.einsum('i,i,ijk,ikl,ilm->ijm', t, A, 
                  np.transpose(B, (0, 2, 1)),  # B^T
                  D, 
                  B)
    
    # Return single element results if input was single element
    if single_element:
        return K[0], D[0], B[0]
    else:
        return K, D, B
    
def _triangle_element_area(x0s):
    """
    This function computes the area of triangle elements given the nodal positions.
    Vectorized version that can handle single triangles or batches.

    Parameters:
        x0s (np.array): Array with the` nodal positions. 
                       Shape (3,2) for single triangle or (n_elements, 3, 2) for batch

    Returns:
        A (float or np.array): Area(s) of the triangle element(s).
                              Float for single triangle, array of shape (n_elements,) for batch
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    # Vectorized area calculation for all elements
    A = (
        x0s[:, 0, 0] * (x0s[:, 1, 1] - x0s[:, 2, 1])
        + x0s[:, 1, 0] * (x0s[:, 2, 1] - x0s[:, 0, 1])
        + x0s[:, 2, 0] * (x0s[:, 0, 1] - x0s[:, 1, 1])
    ) / 2
    
    # Return single element result if input was single element
    if single_element:
        return A[0]
    else:
        return A
    
def _quadrilateral_element_area(x0s_in):
    """
    Vectorized function to compute the area of quadrilateral elements.

    Parameters:
        x0s_in (np.array): Array with the nodal positions. 
                          Shape (4,2) for single quad or (n_elements, 4, 2) for batch

    Returns:
        A (float or np.array): Area(s) of the quadrilateral element(s).
    """
    # Handle single element case
    if x0s_in.ndim == 2:
        x0s_in = x0s_in[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    # First triangle: nodes 0, 1, 2
    x0s_tri1 = x0s_in[:, 0:3, :]
    A1 = (
        x0s_tri1[:, 0, 0] * (x0s_tri1[:, 1, 1] - x0s_tri1[:, 2, 1])
        + x0s_tri1[:, 1, 0] * (x0s_tri1[:, 2, 1] - x0s_tri1[:, 0, 1])
        + x0s_tri1[:, 2, 0] * (x0s_tri1[:, 0, 1] - x0s_tri1[:, 1, 1])
    ) / 2
    
    # Second triangle: nodes 0, 2, 3
    x0s_tri2 = x0s_in[:, [0, 2, 3], :]
    A2 = (
        x0s_tri2[:, 0, 0] * (x0s_tri2[:, 1, 1] - x0s_tri2[:, 2, 1])
        + x0s_tri2[:, 1, 0] * (x0s_tri2[:, 2, 1] - x0s_tri2[:, 0, 1])
        + x0s_tri2[:, 2, 0] * (x0s_tri2[:, 0, 1] - x0s_tri2[:, 1, 1])
    ) / 2
    
    A = A1 + A2
    
    if single_element:
        return A[0]
    else:
        return A
    
def _quadrilateral_element_stiffness(x0s, E=1.0, NU=0.33, e_type=0, t=1.0):
    """
    Vectorized function to compute the stiffness matrix for quadrilateral elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (4,2) for single quad or (n_elements, 4, 2) for batch
        E (float or np.array): Young's modulus
        NU (float or np.array): Poisson's ratio  
        e_type (int): 0 for Plane Stress, 1 for Plane Strain
        t (float or np.array): Thickness of the element

    Returns:
        k (np.array): Stiffness matrices. Shape (8,8) or (n_elements, 8, 8)
        D (np.array): Constitutive matrices. Shape (3,3) or (n_elements, 3, 3)
        B_global (np.array): B-Operators. Shape (3,8) or (n_elements, 3, 8)
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    E = np.atleast_1d(E)
    NU = np.atleast_1d(NU)
    t = np.atleast_1d(t)
    
    # Broadcast to match number of elements
    if E.size == 1:
        E = np.full(n_elements, E[0])
    if NU.size == 1:
        NU = np.full(n_elements, NU[0])
    if t.size == 1:
        t = np.full(n_elements, t[0])
    
    # Check areas
    A = _quadrilateral_element_area(x0s)
    if single_element:
        A = np.array([A])
    
    if np.any(A < 0):
        negative_indices = np.where(A < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")

    # Gaussian quadrature points and weights
    gauss_points = np.array([-0.577350269189626, 0.577350269189626])
    weights = np.ones(2)

    # Create constitutive matrices for all elements
    D = np.zeros((n_elements, 3, 3))
    
    if e_type == 0:  # Plane Stress
        coeff = E / (1 - NU * NU)
        D[:, 0, 0] = coeff
        D[:, 1, 1] = coeff
        D[:, 0, 1] = coeff * NU
        D[:, 1, 0] = coeff * NU
        D[:, 2, 2] = coeff * (1 - NU) / 2
    else:  # Plane Strain
        coeff = E / ((1 + NU) * (1 - 2 * NU))
        D[:, 0, 0] = coeff * (1 - NU)
        D[:, 1, 1] = coeff * (1 - NU)
        D[:, 0, 1] = coeff * NU
        D[:, 1, 0] = coeff * NU
        D[:, 2, 2] = coeff * (1 - 2 * NU) / 2

    # Initialize stiffness matrices and global B matrices
    k = np.zeros((n_elements, 8, 8))
    B_global = np.zeros((n_elements, 3, 8))

    # Gaussian quadrature integration
    for i in range(2):
        r = gauss_points[i]
        wi = weights[i]
        
        for j in range(2):
            s = gauss_points[j]
            wj = weights[j]

            # Shape function derivatives in natural coordinates
            dNdr = np.array([
                -(1 - s), (1 - s), (1 + s), -(1 + s)
            ]) * 0.25
            
            dNds = np.array([
                -(1 - r), -(1 + r), (1 + r), (1 - r)
            ]) * 0.25

            # Compute Jacobian matrices for all elements
            J = np.zeros((n_elements, 2, 2))
            
            for m in range(4):
                J[:, 0, 0] += dNdr[m] * x0s[:, m, 0]
                J[:, 0, 1] += dNdr[m] * x0s[:, m, 1]
                J[:, 1, 0] += dNds[m] * x0s[:, m, 0]
                J[:, 1, 1] += dNds[m] * x0s[:, m, 1]

            # Determinant and inverse of Jacobian
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Shape function derivatives in physical coordinates
            dNdx = np.zeros((n_elements, 4))
            dNdy = np.zeros((n_elements, 4))
            
            for m in range(4):
                dNdx[:, m] = invJ[:, 0, 0] * dNdr[m] + invJ[:, 0, 1] * dNds[m]
                dNdy[:, m] = invJ[:, 1, 0] * dNdr[m] + invJ[:, 1, 1] * dNds[m]

            # Construct B matrices for all elements
            B = np.zeros((n_elements, 3, 8))
            for m in range(4):
                B[:, 0, m*2] = dNdx[:, m]      # dN/dx for u
                B[:, 1, m*2+1] = dNdy[:, m]    # dN/dy for v
                B[:, 2, m*2] = dNdy[:, m]      # dN/dy for u (shear)
                B[:, 2, m*2+1] = dNdx[:, m]    # dN/dx for v (shear)

            # Integration coefficient
            coeff = t * wi * wj * detJ

            # Add contribution to stiffness matrix: k += B^T @ D @ B * coeff
            # Using einsum for efficient batch computation
            k += np.einsum('i,ijk,ikl,ilm->ijm', coeff,
                          np.transpose(B, (0, 2, 1)),  # B^T
                          D,
                          B)
            
            # Accumulate B for global B matrix (averaged)
            B_global += B / 4

    # Return single element results if input was single element
    if single_element:
        return k[0], D[0], B_global[0]
    else:
        return k, D, B_global

def _tetrahedron_element_volume(x0s):
    """
    Vectorized function to compute the volume of tetrahedron elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (4,3) for single tetrahedron or (n_elements, 4, 3) for batch

    Returns:
        V (float or np.array): Volume(s) of the tetrahedron element(s).
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Create the xyz matrices for all elements: [ones, x0s]
    ones_column = np.ones((n_elements, 4, 1))
    xyz = np.concatenate([ones_column, x0s], axis=-1)
    
    # Compute determinants for all elements
    V = np.linalg.det(xyz) / 6
    
    if single_element:
        return V[0]
    else:
        return V
    
def _tetrahedron_element_stiffness(x0s, E=1.0, NU=0.33):
    """
    Vectorized function to compute the stiffness matrix for tetrahedron elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (4,3) for single tetrahedron or (n_elements, 4, 3) for batch
        E (float or np.array): Young's modulus
        NU (float or np.array): Poisson's ratio

    Returns:
        K (np.array): Stiffness matrices. Shape (12,12) or (n_elements, 12, 12)
        D (np.array): Constitutive matrices. Shape (6,6) or (n_elements, 6, 6)
        B (np.array): B-Operators. Shape (6,12) or (n_elements, 6, 12)
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    E = np.atleast_1d(E)
    NU = np.atleast_1d(NU)
    
    # Broadcast to match number of elements
    if E.size == 1:
        E = np.full(n_elements, E[0])
    if NU.size == 1:
        NU = np.full(n_elements, NU[0])
    
    # Compute volumes for all elements
    ones_column = np.ones((n_elements, 4, 1))
    xyz = np.concatenate([ones_column, x0s], axis=-1)
    V = np.linalg.det(xyz) / 6
    
    # Check for negative volumes
    if np.any(V < 0):
        negative_indices = np.where(V < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")
    
    # Compute beta coefficients for all elements
    # mbeta matrices: [ones, y, z] columns for different node combinations
    ones_3x1 = np.ones((n_elements, 3, 1))
    
    # Beta coefficients (derivatives w.r.t. x)
    mbeta1 = np.concatenate([ones_3x1, x0s[:, [1, 2, 3]][:, :, 1:]], axis=-1)  # nodes 1,2,3, cols y,z
    mbeta2 = np.concatenate([ones_3x1, x0s[:, [0, 2, 3]][:, :, 1:]], axis=-1)  # nodes 0,2,3, cols y,z
    mbeta3 = np.concatenate([ones_3x1, x0s[:, [0, 1, 3]][:, :, 1:]], axis=-1)  # nodes 0,1,3, cols y,z
    mbeta4 = np.concatenate([ones_3x1, x0s[:, [0, 1, 2]][:, :, 1:]], axis=-1)  # nodes 0,1,2, cols y,z
    
    beta1 = -np.linalg.det(mbeta1)
    beta2 = np.linalg.det(mbeta2)
    beta3 = -np.linalg.det(mbeta3)
    beta4 = np.linalg.det(mbeta4)
    
    # Gamma coefficients (derivatives w.r.t. y)
    mgamma1 = np.concatenate([ones_3x1, x0s[:, [1, 2, 3]][:, :, [0, 2]]], axis=-1)  # nodes 1,2,3, cols x,z
    mgamma2 = np.concatenate([ones_3x1, x0s[:, [0, 2, 3]][:, :, [0, 2]]], axis=-1)  # nodes 0,2,3, cols x,z
    mgamma3 = np.concatenate([ones_3x1, x0s[:, [0, 1, 3]][:, :, [0, 2]]], axis=-1)  # nodes 0,1,3, cols x,z
    mgamma4 = np.concatenate([ones_3x1, x0s[:, [0, 1, 2]][:, :, [0, 2]]], axis=-1)  # nodes 0,1,2, cols x,z
    
    gamma1 = np.linalg.det(mgamma1)
    gamma2 = -np.linalg.det(mgamma2)
    gamma3 = np.linalg.det(mgamma3)
    gamma4 = -np.linalg.det(mgamma4)
    
    # Delta coefficients (derivatives w.r.t. z)
    mdelta1 = np.concatenate([ones_3x1, x0s[:, [1, 2, 3]][:, :, 0:2]], axis=-1)  # nodes 1,2,3, cols x,y
    mdelta2 = np.concatenate([ones_3x1, x0s[:, [0, 2, 3]][:, :, 0:2]], axis=-1)  # nodes 0,2,3, cols x,y
    mdelta3 = np.concatenate([ones_3x1, x0s[:, [0, 1, 3]][:, :, 0:2]], axis=-1)  # nodes 0,1,3, cols x,y
    mdelta4 = np.concatenate([ones_3x1, x0s[:, [0, 1, 2]][:, :, 0:2]], axis=-1)  # nodes 0,1,2, cols x,y
    
    delta1 = -np.linalg.det(mdelta1)
    delta2 = np.linalg.det(mdelta2)
    delta3 = -np.linalg.det(mdelta3)
    delta4 = np.linalg.det(mdelta4)
    
    # Construct B matrices for all elements
    B = np.zeros((n_elements, 6, 12))
    
    # Row 0: du/dx terms
    B[:, 0, 0] = beta1
    B[:, 0, 3] = beta2
    B[:, 0, 6] = beta3
    B[:, 0, 9] = beta4
    
    # Row 1: dv/dy terms
    B[:, 1, 1] = gamma1
    B[:, 1, 4] = gamma2
    B[:, 1, 7] = gamma3
    B[:, 1, 10] = gamma4
    
    # Row 2: dw/dz terms
    B[:, 2, 2] = delta1
    B[:, 2, 5] = delta2
    B[:, 2, 8] = delta3
    B[:, 2, 11] = delta4
    
    # Row 3: du/dy + dv/dx terms (shear xy)
    B[:, 3, 0] = gamma1
    B[:, 3, 1] = beta1
    B[:, 3, 3] = gamma2
    B[:, 3, 4] = beta2
    B[:, 3, 6] = gamma3
    B[:, 3, 7] = beta3
    B[:, 3, 9] = gamma4
    B[:, 3, 10] = beta4
    
    # Row 4: dv/dz + dw/dy terms (shear yz)
    B[:, 4, 1] = delta1
    B[:, 4, 2] = gamma1
    B[:, 4, 4] = delta2
    B[:, 4, 5] = gamma2
    B[:, 4, 7] = delta3
    B[:, 4, 8] = gamma3
    B[:, 4, 10] = delta4
    B[:, 4, 11] = gamma4
    
    # Row 5: du/dz + dw/dx terms (shear xz)
    B[:, 5, 0] = delta1
    B[:, 5, 2] = beta1
    B[:, 5, 3] = delta2
    B[:, 5, 5] = beta2
    B[:, 5, 6] = delta3
    B[:, 5, 8] = beta3
    B[:, 5, 9] = delta4
    B[:, 5, 11] = beta4
    
    # Normalize B matrices
    B = B / (6 * V[:, np.newaxis, np.newaxis])
    
    # Create constitutive matrices for all elements
    D = np.zeros((n_elements, 6, 6))
    
    # Fill upper 3x3 block
    D[:, 0, 0] = 1 - NU
    D[:, 1, 1] = 1 - NU
    D[:, 2, 2] = 1 - NU
    D[:, 0, 1] = NU
    D[:, 0, 2] = NU
    D[:, 1, 0] = NU
    D[:, 1, 2] = NU
    D[:, 2, 0] = NU
    D[:, 2, 1] = NU
    
    # Fill lower 3x3 block (shear terms)
    shear_coeff = (1 - 2 * NU) / 2
    D[:, 3, 3] = shear_coeff
    D[:, 4, 4] = shear_coeff
    D[:, 5, 5] = shear_coeff
    
    # Apply material constant
    material_coeff = E / ((1 + NU) * (1 - 2 * NU))
    D = material_coeff[:, np.newaxis, np.newaxis] * D
    
    # Compute stiffness matrices: K = V * B^T @ D @ B
    K = np.einsum('i,ijk,ikl,ilm->ijm', 
                  V,
                  np.transpose(B, (0, 2, 1)),  # B^T
                  D,
                  B)
    
    # Return single element results if input was single element
    if single_element:
        return K[0], D[0], B[0]
    else:
        return K, D, B
    
def _hexahedron_element_volume(x0s_in):
    """
    Vectorized function to compute the volume of hexahedron elements.

    Parameters:
        x0s_in (np.array): Array with the nodal positions. 
                          Shape (8,3) for single hex or (n_elements, 8, 3) for batch

    Returns:
        V (float or np.array): Volume(s) of the hexahedron element(s).
    """
    # Handle single element case
    if x0s_in.ndim == 2:
        x0s_in = x0s_in[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    # Define the 6 tetrahedra that decompose each hexahedron
    # Each row contains the 4 node indices for one tetrahedron
    tet_nodes = np.array([
        [0, 1, 3, 7],
        [1, 2, 3, 7], 
        [0, 1, 7, 4],
        [1, 2, 7, 6],
        [1, 5, 7, 4],
        [1, 5, 6, 7]
    ])
    
    # Extract all tetrahedra for all elements
    # Shape: (n_elements, 6, 4, 3)
    all_tets = x0s_in[:, tet_nodes, :]
    
    # Reshape to process all tetrahedra at once
    # Shape: (n_elements * 6, 4, 3)
    n_elements = x0s_in.shape[0]
    all_tets_flat = all_tets.reshape(-1, 4, 3)
    
    # Compute volumes of all tetrahedra
    tet_volumes = _tetrahedron_element_volume(all_tets_flat)
    
    # Reshape back and sum over the 6 tetrahedra for each element
    # Shape: (n_elements, 6) -> (n_elements,)
    tet_volumes_reshaped = tet_volumes.reshape(n_elements, 6)
    V = np.sum(tet_volumes_reshaped, axis=1)
    
    if single_element:
        return V[0]
    else:
        return V
    
def _hexahedron_element_stiffness(x0s, E=1, NU=0.33):
    """
    Vectorized function to compute the stiffness matrix for hexahedron elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (8,3) for single hex or (n_elements, 8, 3) for batch
        E (float or np.array): Young's modulus
        nu (float or np.array): Poisson's ratio

    Returns:
        K (np.array): Stiffness matrices. Shape (24,24) or (n_elements, 24, 24)
        C (np.array): Constitutive matrices. Shape (6,6) or (n_elements, 6, 6)
        B_global (np.array): B-Operators. Shape (6,24) or (n_elements, 6, 24)
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    E = np.atleast_1d(E)
    nu = np.atleast_1d(NU)
    
    # Broadcast to match number of elements
    if E.size == 1:
        E = np.full(n_elements, E[0])
    if nu.size == 1:
        nu = np.full(n_elements, nu[0])
    
    # Check volumes
    V = _hexahedron_element_volume(x0s)
    if single_element:
        V = np.array([V])
    
    if np.any(V < 0):
        negative_indices = np.where(V < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")
    
    # Compute 3D constitutive matrices for all elements
    C = np.zeros((n_elements, 6, 6))
    
    # Material factor for each element
    factor = E / ((1 + nu) * (1 - 2 * nu))
    
    # Fill upper 3x3 block (normal stress terms)
    C[:, 0, 0] = factor * (1 - nu)
    C[:, 1, 1] = factor * (1 - nu)
    C[:, 2, 2] = factor * (1 - nu)
    C[:, 0, 1] = factor * nu
    C[:, 0, 2] = factor * nu
    C[:, 1, 0] = factor * nu
    C[:, 1, 2] = factor * nu
    C[:, 2, 0] = factor * nu
    C[:, 2, 1] = factor * nu
    
    # Fill lower 3x3 block (shear stress terms)
    shear_factor = factor * (1 - 2 * nu) / 2
    C[:, 3, 3] = shear_factor
    C[:, 4, 4] = shear_factor
    C[:, 5, 5] = shear_factor

    # Gauss points coordinates (2x2x2 integration)
    gauss_coord = 1 / np.sqrt(3)
    GaussPoints = [-gauss_coord, gauss_coord]

    # Initialize stiffness matrices and global B matrices
    K = np.zeros((n_elements, 24, 24))
    B_global = np.zeros((n_elements, 6, 24))

    # Loop over each Gauss point (2x2x2 = 8 integration points)
    for xi1 in GaussPoints:
        for xi2 in GaussPoints:
            for xi3 in GaussPoints:
                # Compute shape function derivatives for all elements
                # These are the same for all elements (only depend on xi1, xi2, xi3)
                dShape = (1 / 8) * np.array([
                    [-(1 - xi2) * (1 - xi3), (1 - xi2) * (1 - xi3), (1 + xi2) * (1 - xi3), -(1 + xi2) * (1 - xi3),
                     -(1 - xi2) * (1 + xi3), (1 - xi2) * (1 + xi3), (1 + xi2) * (1 + xi3), -(1 + xi2) * (1 + xi3)],
                    [-(1 - xi1) * (1 - xi3), -(1 + xi1) * (1 - xi3), (1 + xi1) * (1 - xi3), (1 - xi1) * (1 - xi3),
                     -(1 - xi1) * (1 + xi3), -(1 + xi1) * (1 + xi3), (1 + xi1) * (1 + xi3), (1 - xi1) * (1 + xi3)],
                    [-(1 - xi1) * (1 - xi2), -(1 + xi1) * (1 - xi2), -(1 + xi1) * (1 + xi2), -(1 - xi1) * (1 + xi2),
                     (1 - xi1) * (1 - xi2), (1 + xi1) * (1 - xi2), (1 + xi1) * (1 + xi2), (1 - xi1) * (1 + xi2)]
                ])

                # Compute Jacobian matrices for all elements
                # JacobianMatrix[i] = dShape @ x0s[i]
                JacobianMatrix = np.einsum('jk,ikl->ijl', dShape, x0s)
                
                # Compute determinants and inverse Jacobians
                detJ = np.linalg.det(JacobianMatrix)
                invJ = np.linalg.inv(JacobianMatrix)
                
                # Compute auxiliary matrices (inv(J) @ dShape) for all elements
                auxiliar = np.einsum('ijk,kl->ijl', invJ, dShape)

                # Construct B-operators for all elements
                B = np.zeros((n_elements, 6, 24))
                
                # Fill normal strain terms (du/dx, dv/dy, dw/dz)
                for i in range(3):
                    for j in range(8):
                        B[:, i, 3 * j + i] = auxiliar[:, i, j]
                
                # Fill shear strain terms
                # gamma_xy = du/dy + dv/dx
                B[:, 3, 0::3] = auxiliar[:, 1, :]  # du/dy terms
                B[:, 3, 1::3] = auxiliar[:, 0, :]  # dv/dx terms
                
                # gamma_yz = dv/dz + dw/dy
                B[:, 4, 2::3] = auxiliar[:, 1, :]  # dw/dy terms  
                B[:, 4, 1::3] = auxiliar[:, 2, :]  # dv/dz terms
                
                # gamma_xz = du/dz + dw/dx
                B[:, 5, 0::3] = auxiliar[:, 2, :]  # du/dz terms
                B[:, 5, 2::3] = auxiliar[:, 0, :]  # dw/dx terms

                # Add contribution to stiffness matrix: K += B^T @ C @ B * detJ
                K += np.einsum('i,ijk,ikl,ilm->ijm', detJ,
                              np.transpose(B, (0, 2, 1)),  # B^T
                              C,
                              B)
                
                # Accumulate B for global B matrix (averaged over integration points)
                B_global += B / 8

    # Return single element results if input was single element
    if single_element:
        return K[0], C[0], B_global[0]
    else:
        return K, C, B_global