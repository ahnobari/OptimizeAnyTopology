import numpy as np
from numba import int32, njit, prange

@njit(int32[:,:](int32, int32), cache=True, parallel=True)
def generate_elements_2d(nx, ny):
    """
    Generates 2D element connectivity in parallel with original traversal order.
    """
    nel_x, nel_y = nx-1, ny-1
    num_elem = nel_x * nel_y
    elements = np.zeros((num_elem, 4), dtype=np.int32)
    
    for counter in prange(num_elem):
        # Match original nested loop order:
        # for i in range(nx - 1):
        #    for j in range(ny - 1):
        i = counter // (ny - 1)  #  varies slowest
        j = counter % (ny - 1)   # j varies fastest
        
        elements[counter] = [
            j * nx + i,
            j * nx + i + 1,
            (j + 1) * nx + i + 1,
            (j + 1) * nx + i,
        ]
            
    return elements

@njit(int32[:,:](int32, int32, int32), cache=True, parallel=True)
def generate_elements_3d(nx, ny, nz):
    """
    Generates 3D element connectivity in parallel with original traversal order.
    """
    nel_x, nel_y, nel_z = nx-1, ny-1, nz-1
    num_elem = nel_x * nel_y * nel_z
    elements = np.zeros((num_elem, 8), dtype=np.int32)
    
    for counter in prange(num_elem):
        # Match original nested loop order:
        # for i in range(nx - 1):
        #    for j in range(ny - 1):
        #        for k in range(nz - 1):
        i = counter // ((ny - 1) * (nz - 1))  # i varies slowest
        tmp = counter % ((ny - 1) * (nz - 1))
        j = tmp // (nz - 1)                   # j varies middle
        k = tmp % (nz - 1)                    # k varies fastest
        
        elements[counter] = [
            nz * i + nz * nx * j + k,
            nz * (i + 1) + nz * nx * j + k,
            nz * (i + 1) + nz * nx * (j + 1) + k,
            nz * i + nz * nx * (j + 1) + k,
            nz * i + nz * nx * j + k + 1,
            nz * (i + 1) + nz * nx * j + k + 1,
            nz * (i + 1) + nz * nx * (j + 1) + k + 1,
            nz * i + nz * nx * (j + 1) + k + 1,
        ]
                
    return elements

def generate_structured_mesh(dim, nel, dtype=np.float64):
    """
    Wrapper function for structured mesh generation.
    """
    if len(dim) != len(nel):
        raise ValueError("Dimensions of dim and nel must match")
        
    if len(dim) == 2:
        nx, ny = nel[0] + 1, nel[1] + 1
        L, H = dim[0], dim[1]
        
        # Generate node positions using numpy
        x = np.linspace(0, L, nx, dtype=dtype)
        y = np.linspace(0, H, ny, dtype=dtype)
        xx, yy = np.meshgrid(x, y)
        node_positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        
        # Generate elements in parallel
        elements = generate_elements_2d(nx, ny)
        
    elif len(dim) == 3:
        nx, ny, nz = nel[0] + 1, nel[1] + 1, nel[2] + 1
        L, H, W = dim[0], dim[1], dim[2]
        
        # Generate node positions using numpy
        x = np.linspace(0, L, nx, dtype=dtype)
        y = np.linspace(0, H, ny, dtype=dtype)
        z = np.linspace(0, W, nz, dtype=dtype)
        xx, yy, zz = np.meshgrid(x, y, z)
        node_positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
        
        # Generate elements in parallel
        elements = generate_elements_3d(nx, ny, nz)
        
    else:
        raise ValueError("Only 2D and 3D meshes are supported")
        
    return elements, node_positions