import numpy as np

def triangle_element_stiffness(x0s, E=1.0, NU=0.33, e_type=0, t=1.0):
    """
    This function computes the stiffness matrix for a triangle element given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (3,2)
        E (float): Young's modulus
        NU (float): Poisson's ratio
        e_type (int): 0 for Plane Stress, 1 for Plane Strain
        t (float): Thickness of the element

    Returns:
        K (np.array): Stiffness matrix. Shape (6,6)
        D (np.array): Constitutive matrix. Shape (3,3)
        B (np.array): B-Operator. Shape (3,6)
    """
    A = (
        x0s[0, 0] * (x0s[1, 1] - x0s[2, 1])
        + x0s[1, 0] * (x0s[2, 1] - x0s[0, 1])
        + x0s[2, 0] * (x0s[0, 1] - x0s[1, 1])
    ) / 2
    if A < 0:
        raise Exception("Node Order Is Not Correct")
    betai = x0s[1, 1] - x0s[2, 1]
    betaj = x0s[2, 1] - x0s[0, 1]
    betam = x0s[0, 1] - x0s[1, 1]
    gammai = x0s[2, 0] - x0s[1, 0]
    gammaj = x0s[0, 0] - x0s[2, 0]
    gammam = x0s[1, 0] - x0s[0, 0]
    B = np.array(
        [
            [betai, 0, betaj, 0, betam, 0],
            [0, gammai, 0, gammaj, 0, gammam],
            [gammai, betai, gammaj, betaj, gammam, betam],
        ]
    ) / (2 * A)
    if e_type == 0:
        D = (E / (1 - NU * NU)) * np.array(
            [[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]]
        )
    else:
        D = (E / (1 + NU) / (1 - 2 * NU)) * np.array(
            [[1 - NU, NU, 0], [NU, 1 - NU, 0], [0, 0, (1 - 2 * NU) / 2]]
        )

    K = t * A * B.T @ D @ B

    return K, D, B


def triangle_element_area(x0s):
    """
    This function computes the area of a triangle element given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (3,2)

    Returns:
        A (float): Area of the triangle element.
    """
    return (
        x0s[0, 0] * (x0s[1, 1] - x0s[2, 1])
        + x0s[1, 0] * (x0s[2, 1] - x0s[0, 1])
        + x0s[2, 0] * (x0s[0, 1] - x0s[1, 1])
    ) / 2


def tetrahedron_element_stiffness(x0s, E=1.0, NU=0.33):
    """
    This function computes the stiffness matrix for a tetrahedron element given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (4,3)
        E (float): Young's modulus
        NU (float): Poisson's ratio

    Returns:
        K (np.array): Stiffness matrix. Shape (12,12)
        D (np.array): Constitutive matrix. Shape (6,6)
        B (np.array): B-Operator. Shape (6,12)
    """
    xyz = np.concatenate([np.ones([4, 1]), x0s], -1)
    V = np.linalg.det(xyz) / 6

    if V < 0:
        raise Exception("Node Order Is Not Correct")

    mbeta1 = np.concatenate([np.ones([3, 1]), x0s[[1, 2, 3]][:, 1:]], -1)
    mbeta2 = np.concatenate([np.ones([3, 1]), x0s[[0, 2, 3]][:, 1:]], -1)
    mbeta3 = np.concatenate([np.ones([3, 1]), x0s[[0, 1, 3]][:, 1:]], -1)
    mbeta4 = np.concatenate([np.ones([3, 1]), x0s[[0, 1, 2]][:, 1:]], -1)
    mgamma1 = np.concatenate([np.ones([3, 1]), x0s[[1, 2, 3]][:, [0, 2]]], -1)
    mgamma2 = np.concatenate([np.ones([3, 1]), x0s[[0, 2, 3]][:, [0, 2]]], -1)
    mgamma3 = np.concatenate([np.ones([3, 1]), x0s[[0, 1, 3]][:, [0, 2]]], -1)
    mgamma4 = np.concatenate([np.ones([3, 1]), x0s[[0, 1, 2]][:, [0, 2]]], -1)
    mdelta1 = np.concatenate([np.ones([3, 1]), x0s[[1, 2, 3]][:, 0:2]], -1)
    mdelta2 = np.concatenate([np.ones([3, 1]), x0s[[0, 2, 3]][:, 0:2]], -1)
    mdelta3 = np.concatenate([np.ones([3, 1]), x0s[[0, 1, 3]][:, 0:2]], -1)
    mdelta4 = np.concatenate([np.ones([3, 1]), x0s[[0, 1, 2]][:, 0:2]], -1)
    beta1 = -1 * np.linalg.det(mbeta1)
    beta2 = np.linalg.det(mbeta2)
    beta3 = -1 * np.linalg.det(mbeta3)
    beta4 = np.linalg.det(mbeta4)
    gamma1 = np.linalg.det(mgamma1)
    gamma2 = -1 * np.linalg.det(mgamma2)
    gamma3 = np.linalg.det(mgamma3)
    gamma4 = -1 * np.linalg.det(mgamma4)
    delta1 = -1 * np.linalg.det(mdelta1)
    delta2 = np.linalg.det(mdelta2)
    delta3 = -1 * np.linalg.det(mdelta3)
    delta4 = np.linalg.det(mdelta4)

    B = (
        np.array(
            [
                [beta1, 0, 0, beta2, 0, 0, beta3, 0, 0, beta4, 0, 0],
                [0, gamma1, 0, 0, gamma2, 0, 0, gamma3, 0, 0, gamma4, 0],
                [0, 0, delta1, 0, 0, delta2, 0, 0, delta3, 0, 0, delta4],
                [
                    gamma1,
                    beta1,
                    0,
                    gamma2,
                    beta2,
                    0,
                    gamma3,
                    beta3,
                    0,
                    gamma4,
                    beta4,
                    0,
                ],
                [
                    0,
                    delta1,
                    gamma1,
                    0,
                    delta2,
                    gamma2,
                    0,
                    delta3,
                    gamma3,
                    0,
                    delta4,
                    gamma4,
                ],
                [
                    delta1,
                    0,
                    beta1,
                    delta2,
                    0,
                    beta2,
                    delta3,
                    0,
                    beta3,
                    delta4,
                    0,
                    beta4,
                ],
            ]
        )
        / 6
        / V
    )
    D = np.zeros([6, 6])
    D[0:3, 0:3] = np.array([[1 - NU, NU, NU], [NU, 1 - NU, NU], [NU, NU, 1 - NU]])
    D[3:, 3:] = np.array(
        [[(1 - 2 * NU) / 2, 0, 0], [0, (1 - 2 * NU) / 2, 0], [0, 0, (1 - 2 * NU) / 2]]
    )
    D = E / (1 + NU) / (1 - 2 * NU) * D
    K = V * B.T @ D @ B

    return K, D, B


def tetrahedron_element_volume(x0s):
    """
    This function computes the volume of a tetrahedron element given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (4,3)

    Returns:
        V (float): Volume of the tetrahedron element.
    """
    xyz = np.concatenate([np.ones([4, 1]), x0s], -1)
    V = np.linalg.det(xyz) / 6

    return V


def quadrilateral_element_stiffness(x0s, E=1.0, NU=0.33, e_type=0, t=1.0):
    """
    This function computes the stiffness matrix for a quadrilateral element given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (4,2)
        E (float): Young's modulus
        NU (float): Poisson's ratio
        e_type (int): 0 for Plane Stress, 1 for Plane Strain
        t (float): Thickness of the element

    Returns:
        k (np.array): Stiffness matrix. Shape (8,8)
        D (np.array): Constitutive matrix. Shape (3,3)
        B (np.array): B-Operator. Shape (3,8)
    """

    A = quadrilateral_element_area(x0s)

    if A < 0:
        raise Exception("Node Order Is Not Correct")

    # Guass Points For Integration
    guass_point = np.zeros([2, 2])
    weight = np.ones([2, 2])
    guass_point[0, :] = -0.577350269189626
    guass_point[1, :] = 0.577350269189626

    # Create D
    D = np.zeros([3, 3])
    B_global = np.zeros([3, 8])
    if e_type == 0:
        coeff = E / (1 - (NU * NU))
        D[0, 0] = coeff
        D[1, 1] = D[0, 0]
        D[0, 1] = coeff * NU
        D[1, 0] = D[0, 1]
        D[2, 2] = coeff * (1 - NU) / 2

    elif e_type == 1:
        coeff = E / ((1 + NU) * (1 - 2 * NU))
        D[0, 0] = coeff * (1 - NU)
        D[1, 1] = D[0, 0]
        D[0, 1] = coeff * NU
        D[1, 0] = D[0, 1]
        D[2, 2] = coeff * (1 - 2 * NU) / 2

    # Creat stiffness matrix
    k = np.zeros([8, 8])
    coeff = 0

    for i in range(2):
        r = guass_point[i, 0]
        wi = weight[i, 0]

        for j in range(2):
            s = guass_point[j, 1]
            wj = weight[j, 1]

            dNdx = np.zeros([4, 1])
            dNdy = np.zeros([4, 1])

            sN = np.zeros([4, 1])
            dNdr = np.zeros([4, 1])
            dNds = np.zeros([4, 1])

            # Shape functions
            sN[0] = (1 - r) * (1 - s)
            sN[1] = (1 + r) * (1 - s)
            sN[2] = (1 + r) * (1 + s)
            sN[3] = (1 - r) * (1 + s)
            sN *= 0.25

            # Derivatives
            dNdr[0] = -(1 - s)
            dNdr[1] = 1 - s
            dNdr[2] = 1 + s
            dNdr[3] = -(1 + s)
            dNdr *= 0.25

            dNds[0] = -(1 - r)
            dNds[1] = -(1 + r)
            dNds[2] = 1 + r
            dNds[3] = 1 - r
            dNds *= 0.25

            J = np.zeros([2, 2])

            for m in range(4):
                J[0, 0] = J[0, 0] + dNdr[m] * x0s[m, 0]
                J[0, 1] = J[0, 1] + dNdr[m] * x0s[m, 1]
                J[1, 0] = J[1, 0] + dNds[m] * x0s[m, 0]
                J[1, 1] = J[1, 1] + dNds[m] * x0s[m, 1]

            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            for m in range(4):
                dNdx[m] = invJ[0, 0] * dNdr[m] + invJ[0, 1] * dNds[m]
                dNdy[m] = invJ[1, 0] * dNdr[m] + invJ[1, 1] * dNds[m]

            B = np.zeros([3, 8])

            # Compute B
            for m in range(4):
                B[:, m * 2 : 2 * (m + 1)] = np.array(
                    [[dNdx[m, 0], 0.0], [0.0, dNdy[m, 0]], [dNdy[m, 0], dNdx[m, 0]]]
                )

            coeff = t * wi * wj * detJ

            k = k + (B.T) @ D @ B * coeff
            B_global = B_global + B / 4

    return k, D, B_global


def quadrilateral_element_area(x0s_in):
    """
    This function computes the area of a quadrilateral element given the nodal positions.

    Parameters:
        x0s_in (np.array): Array with the nodal positions. Shape (4,3)

    Returns:
        A (float): Area of the quadrilateral element.
    """
    x0s = x0s_in[0:3, :]
    A = (
        x0s[0, 0] * (x0s[1, 1] - x0s[2, 1])
        + x0s[1, 0] * (x0s[2, 1] - x0s[0, 1])
        + x0s[2, 0] * (x0s[0, 1] - x0s[1, 1])
    ) / 2
    x0s = x0s_in[[0, 2, 3], :]
    A += (
        x0s[0, 0] * (x0s[1, 1] - x0s[2, 1])
        + x0s[1, 0] * (x0s[2, 1] - x0s[0, 1])
        + x0s[2, 0] * (x0s[0, 1] - x0s[1, 1])
    ) / 2

    return A


def hexahedron_element_stiffness(x0s, E=1, nu=0.33):
    """
    This function computes the stiffness matrix for a hexahedron element given the nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (8,3)
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        K (np.array): Stiffness matrix. Shape (24,24)
        D (np.array): Constitutive matrix. Shape (6,6)
        B (np.array): B-Operator. Shape (6,24)
    """

    V = hexahedron_element_volume(x0s)

    if V < 0:
        raise Exception("Node Order Is Not Correct")

    # Compute 3D constitutive matrix (linear continuum mechanics)
    C = (
        E
        / ((1 + nu) * (1 - 2 * nu))
        * np.array(
            [
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
            ]
        )
    )

    # Gauss points coordinates on each direction
    GaussPoint = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

    # Matrix of vertices coordinates.
    coordinates = x0s

    # Preallocate memory for stiffness matrix
    K = np.zeros((24, 24))
    B_global = np.zeros((6, 24))
    # Loop over each Gauss point
    for xi1 in GaussPoint:
        for xi2 in GaussPoint:
            for xi3 in GaussPoint:
                # Compute shape functions derivatives
                dShape = (1 / 8) * np.array(
                    [
                        [
                            -(1 - xi2) * (1 - xi3),
                            (1 - xi2) * (1 - xi3),
                            (1 + xi2) * (1 - xi3),
                            -(1 + xi2) * (1 - xi3),
                            -(1 - xi2) * (1 + xi3),
                            (1 - xi2) * (1 + xi3),
                            (1 + xi2) * (1 + xi3),
                            -(1 + xi2) * (1 + xi3),
                        ],
                        [
                            -(1 - xi1) * (1 - xi3),
                            -(1 + xi1) * (1 - xi3),
                            (1 + xi1) * (1 - xi3),
                            (1 - xi1) * (1 - xi3),
                            -(1 - xi1) * (1 + xi3),
                            -(1 + xi1) * (1 + xi3),
                            (1 + xi1) * (1 + xi3),
                            (1 - xi1) * (1 + xi3),
                        ],
                        [
                            -(1 - xi1) * (1 - xi2),
                            -(1 + xi1) * (1 - xi2),
                            -(1 + xi1) * (1 + xi2),
                            -(1 - xi1) * (1 + xi2),
                            (1 - xi1) * (1 - xi2),
                            (1 + xi1) * (1 - xi2),
                            (1 + xi1) * (1 + xi2),
                            (1 - xi1) * (1 + xi2),
                        ],
                    ]
                )

                # Compute Jacobian matrix
                JacobianMatrix = dShape @ coordinates

                # Compute auxiliar matrix for construction of B-Operator
                auxiliar = np.linalg.inv(JacobianMatrix) @ dShape

                # Preallocate memory for B-Operator
                B = np.zeros((6, 24))
                # Construct B-Operator
                for i in range(3):
                    for j in range(8):
                        B[i, 3 * j + i] = auxiliar[i, j]
                B[3, 0::3] = auxiliar[1, :]
                B[3, 1::3] = auxiliar[0, :]
                B[4, 2::3] = auxiliar[1, :]
                B[4, 1::3] = auxiliar[2, :]
                B[5, 0::3] = auxiliar[2, :]
                B[5, 2::3] = auxiliar[0, :]

                # Add to stiffness matrix
                K += B.T @ C @ B * np.linalg.det(JacobianMatrix)

                B_global += B / 8

    return K, C, B_global


def hexahedron_element_volume(x0s_in):
    """
    This function computes the volume of a hexahedron element given the nodal positions.

    Parameters:
        x0s_in (np.array): Array with the nodal positions. Shape (8,3)

    Returns:
        V (float): Volume of the hexahedron element.
    """
    V = 0.0
    x0s = x0s_in[[0, 1, 3, 7], :]
    V += tetrahedron_element_volume(x0s)
    x0s = x0s_in[[1, 2, 3, 7], :]
    V += tetrahedron_element_volume(x0s)
    x0s = x0s_in[[0, 1, 7, 4], :]
    V += tetrahedron_element_volume(x0s)
    x0s = x0s_in[[1, 2, 7, 6], :]
    V += tetrahedron_element_volume(x0s)
    x0s = x0s_in[[1, 5, 7, 4], :]
    V += tetrahedron_element_volume(x0s)
    x0s = x0s_in[[1, 5, 6, 7], :]
    V += tetrahedron_element_volume(x0s)

    return V


def auto_stiffness(x0s, E, Nu):
    """
    This function computes the stiffness matrix for a given element type and nodal positions.

    Parameters:
        x0s (np.array): Array with the nodal positions. Shape (n_nodes, dim)
        E (float): Young's modulus
        Nu (float): Poisson's ratio

    Returns:
        K (np.array): Stiffness matrix. Shape (n_dofs, n_dofs)
        D (np.array): Constitutive matrix. Shape (3,3) in 2D and (6,6) in 3D
        B (np.array): B-Operator. shape (D.shape[0], n_dofs)
        A/V (float): Area/Volume of the element
    """
    if x0s.shape[0] == 3:
        A = triangle_element_area(x0s)
        if A < 0:
            A = triangle_element_area(x0s[::-1, :])
            K, D, B = triangle_element_stiffness(x0s[::-1, :], E, Nu)
            reorder_idx = np.array([2, 1, 0]) * 2
            reorder_idx = np.vstack([reorder_idx, reorder_idx + 1]).T.flatten()
            K = K[reorder_idx, :][:, reorder_idx]
            B = B[:, reorder_idx]
        else:
            K, D, B = triangle_element_stiffness(x0s, E, Nu)
    elif x0s.shape[0] == 4 and x0s.shape[1] == 3:
        A = tetrahedron_element_volume(x0s)
        if A < 0:
            A = tetrahedron_element_volume(x0s[[0, 1, 3, 2], :])
            K, D, B = tetrahedron_element_stiffness(x0s[[0, 1, 3, 2], :], E, Nu)
            reorder_idx = np.array([0, 1, 3, 2]) * 3
            reorder_idx = np.vstack(
                [reorder_idx, reorder_idx + 1, reorder_idx + 2]
            ).T.flatten()
            K = K[reorder_idx, :][:, reorder_idx]
            B = B[:, reorder_idx]
        else:
            K, D, B = tetrahedron_element_stiffness(x0s, E, Nu)
    elif x0s.shape[0] == 4:
        A = quadrilateral_element_area(x0s)
        if A < 0:
            A = quadrilateral_element_area(x0s[::-1, :])
            K, D, B = quadrilateral_element_stiffness(x0s[::-1, :], E, Nu)
            reorder_idx = np.array([3, 2, 1, 0]) * 2
            reorder_idx = np.vstack([reorder_idx, reorder_idx + 1]).T.flatten()
            K = K[reorder_idx, :][:, reorder_idx]
            B = B[:, reorder_idx]
        else:
            K, D, B = quadrilateral_element_stiffness(x0s, E, Nu)
        K, D, B = quadrilateral_element_stiffness(x0s, E, Nu)
    elif x0s.shape[0] == 8:
        A = hexahedron_element_volume(x0s)
        if A < 0:
            A = hexahedron_element_volume(x0s[[4, 5, 6, 7, 0, 1, 2, 3], :])
            K, D, B = hexahedron_element_stiffness(
                x0s[[4, 5, 6, 7, 0, 1, 2, 3], :], E, Nu
            )
            reorder_idx = np.array([4, 5, 6, 7, 0, 1, 2, 3]) * 3
            reorder_idx = np.vstack(
                [reorder_idx, reorder_idx + 1, reorder_idx + 2]
            ).T.flatten()
            K = K[reorder_idx, :][:, reorder_idx]
            B = B[:, reorder_idx]
        else:
            K, D, B = hexahedron_element_stiffness(x0s, E, Nu)
        K, D, B = hexahedron_element_stiffness(x0s, E, Nu)
    else:
        raise Exception("Element Type Not Supported")

    return K, D, B, A
