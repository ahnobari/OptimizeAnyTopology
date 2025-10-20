import cupy as cp

element_2d_kernel_code = '''
extern "C" __global__
void generate_elements_2d(int* elements, int nx, int ny) {
    int counter = blockDim.x * blockIdx.x + threadIdx.x;
    int nel_x = nx - 1;
    int nel_y = ny - 1;
    int num_elem = nel_x * nel_y;
    
    if (counter < num_elem) {
        int i = counter / nel_y;  // i varies slowest
        int j = counter % nel_y;  // j varies fastest
        
        elements[counter * 4] = j * nx + i;
        elements[counter * 4 + 1] = j * nx + i + 1;
        elements[counter * 4 + 2] = (j + 1) * nx + i + 1;
        elements[counter * 4 + 3] = (j + 1) * nx + i;
    }
}
'''

element_3d_kernel_code = '''
extern "C" __global__
void generate_elements_3d(int* elements, int nx, int ny, int nz) {
    int counter = blockDim.x * blockIdx.x + threadIdx.x;
    int nel_x = nx - 1;
    int nel_y = ny - 1;
    int nel_z = nz - 1;
    int num_elem = nel_x * nel_y * nel_z;
    
    if (counter < num_elem) {
        int i = counter / (nel_y * nel_z);  // i varies slowest
        int tmp = counter % (nel_y * nel_z);
        int j = tmp / nel_z;                // j varies middle
        int k = tmp % nel_z;                // k varies fastest
        
        int idx = counter * 8;
        elements[idx] = nz * i + nz * nx * j + k;
        elements[idx + 1] = nz * (i + 1) + nz * nx * j + k;
        elements[idx + 2] = nz * (i + 1) + nz * nx * (j + 1) + k;
        elements[idx + 3] = nz * i + nz * nx * (j + 1) + k;
        elements[idx + 4] = nz * i + nz * nx * j + k + 1;
        elements[idx + 5] = nz * (i + 1) + nz * nx * j + k + 1;
        elements[idx + 6] = nz * (i + 1) + nz * nx * (j + 1) + k + 1;
        elements[idx + 7] = nz * i + nz * nx * (j + 1) + k + 1;
    }
}
'''

generate_elements_2d_kernel = cp.RawKernel(element_2d_kernel_code, 'generate_elements_2d')
generate_elements_3d_kernel = cp.RawKernel(element_3d_kernel_code, 'generate_elements_3d')

def generate_structured_mesh_cuda(dim, nel, dtype=cp.float64):
    """
    Generate structured mesh entirely on GPU using CUDA.
    
    Parameters:
        dim: Array with domain dimensions
        nel: Array with number of elements in each direction
        
    Returns:
        elements: Array of element connectivity (on GPU)
        node_positions: Array of nodal coordinates (on GPU)
    """
    # Convert inputs to GPU if needed
    dim = cp.asarray(dim)
    nel = cp.asarray(nel)
    
    if len(dim) != len(nel):
        raise ValueError("Dimensions of dim and nel must match")
        
    if len(dim) == 2:
        nx, ny = nel[0] + 1, nel[1] + 1
        L, H = dim[0], dim[1]
        
        # Generate elements
        num_elem = int((nx - 1) * (ny - 1))
        elements = cp.zeros((num_elem, 4), dtype=cp.int32)
        
        threads_per_block = 256
        blocks_per_grid = (num_elem + threads_per_block - 1) // threads_per_block
        generate_elements_2d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (elements, int(nx), int(ny))
        )
        
        # Generate node positions using CuPy
        x = cp.linspace(0, L, int(nx), dtype=dtype)
        y = cp.linspace(0, H, int(ny), dtype=dtype)
        xx, yy = cp.meshgrid(x, y, copy=False)
        node_positions = cp.stack([xx.flatten(), yy.flatten()], axis=-1, dtype=dtype)
        
    elif len(dim) == 3:
        nx, ny, nz = nel[0] + 1, nel[1] + 1, nel[2] + 1
        L, H, W = dim[0], dim[1], dim[2]
        
        # Generate elements
        num_elem = int((nx - 1) * (ny - 1) * (nz - 1))
        elements = cp.zeros((num_elem, 8), dtype=cp.int32)
        
        threads_per_block = 256
        blocks_per_grid = (num_elem + threads_per_block - 1) // threads_per_block
        generate_elements_3d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (elements, int(nx), int(ny), int(nz))
        )
        
        # Generate node positions using CuPy
        x = cp.linspace(0, L, int(nx), dtype=dtype)
        y = cp.linspace(0, H, int(ny), dtype=dtype)
        z = cp.linspace(0, W, int(nz), dtype=dtype)
        xx, yy, zz = cp.meshgrid(x, y, z, copy=False)
        node_positions = cp.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1, dtype=dtype)
        
    else:
        raise ValueError("Only 2D and 3D meshes are supported")
        
    return elements, node_positions