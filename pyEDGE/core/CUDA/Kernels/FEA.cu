template<typename T> __global__
void FEA_locals_node_basis_parallel_cuda_kernel(T* K_single, T* D_single, T* B_single, int* elements_flat, 
    int nel, T* weights, T* U, int dof, int elements_size, int B_size, T* strain, T* stress, T* strain_energy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nel) return;

    int start = i * elements_size;
    int end = start + elements_size;
    T weight = weights[i];
    
    // Calculate strain
    for (int j = 0; j < B_size; j++) {
        strain[i * B_size + j] = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            strain[i * B_size + j] += B_single[j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
    }
    
    // Calculate stress
    for (int j = 0; j < B_size; j++) {
        stress[i * B_size + j] = 0;
        for (int k = 0; k < B_size; k++) {
            stress[i * B_size + j] += D_single[j * B_size + k] * strain[i * B_size + k];
        }
    }
    
    // Calculate strain energy
    strain_energy[i] = 0;
    for (int j = 0; j < elements_size * dof; j++) {
        T val = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            val += weight * K_single[j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
        strain_energy[i] += 0.5 * U[elements_flat[start + j/dof] * dof + j%dof] * val;
    }
}

template<typename T> __global__
void FEA_locals_node_basis_parallel_full_cuda_kernel(T* Ks, T* Ds, T* Bs, int* elements_flat,
    int nel, T* weights, T* U, int dof, int elements_size, int B_size, T* strain, T* stress, T* strain_energy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nel) return;

    int start = i * elements_size;
    int end = start + elements_size;
    T weight = weights[i];
    
    // Calculate strain
    for (int j = 0; j < B_size; j++) {
        strain[i * B_size + j] = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            strain[i * B_size + j] += Bs[i * B_size * elements_size * dof + j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
    }
    
    // Calculate stress
    for (int j = 0; j < B_size; j++) {
        stress[i * B_size + j] = 0;
        for (int k = 0; k < B_size; k++) {
            stress[i * B_size + j] += Ds[i * B_size * B_size + j * B_size + k] * strain[i * B_size + k];
        }
    }
    
    // Calculate strain energy
    strain_energy[i] = 0;
    for (int j = 0; j < elements_size * dof; j++) {
        T val = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            val += weight * Ks[i * elements_size * dof * elements_size * dof + j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
        strain_energy[i] += 0.5 * U[elements_flat[start + j/dof] * dof + j%dof] * val;
    }
}

template<typename T> __global__
void FEA_locals_node_basis_parallel_flat_cuda_kernel(T* K_flat, T* D_flat, T* B_flat, int* elements_flat,
    int* elements_ptr, int* K_ptr, int* B_ptr, int* D_ptr, int nel, T* weights, T* U, int dof, int B_size,
    T* strain, T* stress, T* strain_energy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nel) return;

    int start = elements_ptr[i];
    int end = elements_ptr[i + 1];
    int size = end - start;
    T weight = weights[i];
    
    // Calculate strain
    for (int j = 0; j < B_size; j++) {
        strain[i * B_size + j] = 0;
        for (int k = 0; k < size * dof; k++) {
            strain[i * B_size + j] += B_flat[B_ptr[i] + j * size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
    }
    
    // Calculate stress
    for (int j = 0; j < B_size; j++) {
        stress[i * B_size + j] = 0;
        for (int k = 0; k < B_size; k++) {
            stress[i * B_size + j] += D_flat[D_ptr[i] + j * B_size + k] * strain[i * B_size + k];
        }
    }
    
    // Calculate strain energy
    strain_energy[i] = 0;
    for (int j = 0; j < size * dof; j++) {
        T val = 0;
        for (int k = 0; k < size * dof; k++) {
            val += weight * K_flat[K_ptr[i] + j * size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
        strain_energy[i] += 0.5 * U[elements_flat[start + j/dof] * dof + j%dof] * val;
    }
}