from ._physx import Physx
import numpy as np

class SteadyHeatTransfer(Physx):
    def __init__(self, k=1.0, thickness=1.0):
        super().__init__()
        self.k = k  # thermal conductivity
        self.thickness = thickness

    def K(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 3 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 3 and x0s.shape[2] == 2):
            return _triangle_element_conductivity(x0s, k=self.k, t=self.thickness)[0]
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return _quadrilateral_element_conductivity(x0s, k=self.k, t=self.thickness)[0]
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 3):
            return _tetrahedron_element_conductivity(x0s, k=self.k, t=self.thickness)[0]
        elif (x0s.ndim == 2 and x0s.shape[0] == 8 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 8 and x0s.shape[2] == 3):
            return _hexahedron_element_conductivity(x0s, k=self.k, t=self.thickness)[0]
        else:
            raise ValueError("Invalid input shape")

    def locals(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 3 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 3 and x0s.shape[2] == 2):
            return list(_triangle_element_conductivity(x0s, k=self.k, t=self.thickness)[1:])
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return list(_quadrilateral_element_conductivity(x0s, k=self.k, t=self.thickness)[1:])
        elif (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 3):
            return list(_tetrahedron_element_conductivity(x0s, k=self.k, t=self.thickness)[1:])
        elif (x0s.ndim == 2 and x0s.shape[0] == 8 and x0s.shape[1] == 3) or (x0s.ndim == 3 and x0s.shape[1] == 8 and x0s.shape[2] == 3):
            return list(_hexahedron_element_conductivity(x0s, k=self.k, t=self.thickness)[1:])
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


def _triangle_element_conductivity(x0s, k=1.0, t=1.0):
    """
    This function computes the conductivity matrix for a batch of triangle elements given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (n_elements, 3, 2) or (3, 2) for single element
        k (float or np.array): Thermal conductivity. Can be scalar or array of shape (n_elements,)
        t (float or np.array): Thickness of the element. Can be scalar or array of shape (n_elements,)

    Returns:
        K (np.array): Conductivity matrices. Shape (n_elements, 3, 3) or (3, 3) for single element
        k_mat (np.array): Conductivity matrices. Shape (n_elements, 2, 2) or (2, 2) for single element
        B (np.array): B-Operators. Shape (n_elements, 2, 3) or (2, 3) for single element
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    k = np.atleast_1d(k)
    t = np.atleast_1d(t)
    
    # Broadcast to match number of elements
    if k.size == 1:
        k = np.full(n_elements, k[0])
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
    
    # Build B matrices for all elements (gradient operator)
    B = np.zeros((n_elements, 2, 3))
    B[:, 0, 0] = betai  # dN1/dx
    B[:, 0, 1] = betaj  # dN2/dx
    B[:, 0, 2] = betam  # dN3/dx
    B[:, 1, 0] = gammai # dN1/dy
    B[:, 1, 1] = gammaj # dN2/dy
    B[:, 1, 2] = gammam # dN3/dy
    
    # Normalize by 2*A
    B = B / (2 * A[:, np.newaxis, np.newaxis])
    
    # Compute conductivity matrices (2x2 identity scaled by k)
    k_mat = np.zeros((n_elements, 2, 2))
    k_mat[:, 0, 0] = k
    k_mat[:, 1, 1] = k
    
    # Compute conductivity matrices: K = t * A * B^T @ k_mat @ B
    # Using einsum for efficient batch matrix multiplication
    K = np.einsum('i,i,ijk,ikl,ilm->ijm', t, A, 
                  np.transpose(B, (0, 2, 1)),  # B^T
                  k_mat, 
                  B)
    
    # Return single element results if input was single element
    if single_element:
        return K[0], k_mat[0], B[0]
    else:
        return K, k_mat, B


def _quadrilateral_element_conductivity(x0s, k=1.0, t=1.0):
    """
    Vectorized function to compute the conductivity matrix for quadrilateral elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (4,2) for single quad or (n_elements, 4, 2) for batch
        k (float or np.array): Thermal conductivity
        t (float or np.array): Thickness of the element

    Returns:
        K (np.array): Conductivity matrices. Shape (4,4) or (n_elements, 4, 4)
        k_mat (np.array): Conductivity matrices. Shape (2,2) or (n_elements, 2, 2)
        B_global (np.array): B-Operators. Shape (2,4) or (n_elements, 2, 4)
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    k = np.atleast_1d(k)
    t = np.atleast_1d(t)
    
    # Broadcast to match number of elements
    if k.size == 1:
        k = np.full(n_elements, k[0])
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

    # Create conductivity matrices for all elements
    k_mat = np.zeros((n_elements, 2, 2))
    k_mat[:, 0, 0] = k
    k_mat[:, 1, 1] = k

    # Initialize conductivity matrices and global B matrices
    K = np.zeros((n_elements, 4, 4))
    B_global = np.zeros((n_elements, 2, 4))

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

            # Construct B matrices for all elements (gradient operator)
            B = np.zeros((n_elements, 2, 4))
            for m in range(4):
                B[:, 0, m] = dNdx[:, m]  # dN/dx
                B[:, 1, m] = dNdy[:, m]  # dN/dy

            # Integration coefficient
            coeff = t * wi * wj * detJ

            # Add contribution to conductivity matrix: K += B^T @ k_mat @ B * coeff
            # Using einsum for efficient batch computation
            K += np.einsum('i,ijk,ikl,ilm->ijm', coeff,
                          np.transpose(B, (0, 2, 1)),  # B^T
                          k_mat,
                          B)
            
            # Accumulate B for global B matrix (averaged)
            B_global += B / 4

    # Return single element results if input was single element
    if single_element:
        return K[0], k_mat[0], B_global[0]
    else:
        return K, k_mat, B_global


def _tetrahedron_element_conductivity(x0s, k=1.0, t=1.0):
    """
    Vectorized function to compute the conductivity matrix for tetrahedron elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (4,3) for single tetrahedron or (n_elements, 4, 3) for batch
        k (float or np.array): Thermal conductivity
        t (float or np.array): Not used for 3D elements (kept for consistency)

    Returns:
        K (np.array): Conductivity matrices. Shape (4,4) or (n_elements, 4, 4)
        k_mat (np.array): Conductivity matrices. Shape (3,3) or (n_elements, 3, 3)
        B (np.array): B-Operators. Shape (3,4) or (n_elements, 3, 4)
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    k = np.atleast_1d(k)
    
    # Broadcast to match number of elements
    if k.size == 1:
        k = np.full(n_elements, k[0])
    
    # Compute volumes for all elements
    ones_column = np.ones((n_elements, 4, 1))
    xyz = np.concatenate([ones_column, x0s], axis=-1)
    V = np.linalg.det(xyz) / 6
    
    # Check for negative volumes
    if np.any(V < 0):
        negative_indices = np.where(V < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")
    
    # Compute beta, gamma, and delta coefficients for all elements
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
    
    # Construct B matrices for all elements (gradient operator)
    B = np.zeros((n_elements, 3, 4))
    
    # Row 0: dT/dx terms
    B[:, 0, 0] = beta1
    B[:, 0, 1] = beta2
    B[:, 0, 2] = beta3
    B[:, 0, 3] = beta4
    
    # Row 1: dT/dy terms
    B[:, 1, 0] = gamma1
    B[:, 1, 1] = gamma2
    B[:, 1, 2] = gamma3
    B[:, 1, 3] = gamma4
    
    # Row 2: dT/dz terms
    B[:, 2, 0] = delta1
    B[:, 2, 1] = delta2
    B[:, 2, 2] = delta3
    B[:, 2, 3] = delta4
    
    # Normalize B matrices
    B = B / (6 * V[:, np.newaxis, np.newaxis])
    
    # Create conductivity matrices for all elements (3x3 identity scaled by k)
    k_mat = np.zeros((n_elements, 3, 3))
    k_mat[:, 0, 0] = k
    k_mat[:, 1, 1] = k
    k_mat[:, 2, 2] = k
    
    # Compute conductivity matrices: K = V * B^T @ k_mat @ B
    K = np.einsum('i,ijk,ikl,ilm->ijm', 
                  V,
                  np.transpose(B, (0, 2, 1)),  # B^T
                  k_mat,
                  B)
    
    # Return single element results if input was single element
    if single_element:
        return K[0], k_mat[0], B[0]
    else:
        return K, k_mat, B


def _hexahedron_element_conductivity(x0s, k=1.0, t=1.0):
    """
    Vectorized function to compute the conductivity matrix for hexahedron elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (8,3) for single hex or (n_elements, 8, 3) for batch
        k (float or np.array): Thermal conductivity
        t (float or np.array): Not used for 3D elements (kept for consistency)

    Returns:
        K (np.array): Conductivity matrices. Shape (8,8) or (n_elements, 8, 8)
        k_mat (np.array): Conductivity matrices. Shape (3,3) or (n_elements, 3, 3)
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
    k = np.atleast_1d(k)
    
    # Broadcast to match number of elements
    if k.size == 1:
        k = np.full(n_elements, k[0])
    
    # Check volumes
    V = _hexahedron_element_volume(x0s)
    if single_element:
        V = np.array([V])
    
    if np.any(V < 0):
        negative_indices = np.where(V < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")
    
    # Create 3D conductivity matrices for all elements
    k_mat = np.zeros((n_elements, 3, 3))
    k_mat[:, 0, 0] = k
    k_mat[:, 1, 1] = k
    k_mat[:, 2, 2] = k

    # Gauss points coordinates (2x2x2 integration)
    gauss_coord = 1 / np.sqrt(3)
    GaussPoints = [-gauss_coord, gauss_coord]

    # Initialize conductivity matrices and global B matrices
    K = np.zeros((n_elements, 8, 8))
    B_global = np.zeros((n_elements, 3, 8))

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

                # Construct B-operators for all elements (gradient operator)
                B = np.zeros((n_elements, 3, 8))
                
                # Fill gradient terms (dT/dx, dT/dy, dT/dz)
                for i in range(3):
                    for j in range(8):
                        B[:, i, j] = auxiliar[:, i, j]

                # Add contribution to conductivity matrix: K += B^T @ k_mat @ B * detJ
                K += np.einsum('i,ijk,ikl,ilm->ijm', detJ,
                              np.transpose(B, (0, 2, 1)),  # B^T
                              k_mat,
                              B)
                
                # Accumulate B for global B matrix (averaged over integration points)
                B_global += B / 8

    # Return single element results if input was single element
    if single_element:
        return K[0], k_mat[0], B_global[0]
    else:
        return K, k_mat, B_global


# Reuse geometry functions from the original elasticity code
def _triangle_element_area(x0s):
    """
    This function computes the area of triangle elements given the nodal positions.
    Vectorized version that can handle single triangles or batches.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
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