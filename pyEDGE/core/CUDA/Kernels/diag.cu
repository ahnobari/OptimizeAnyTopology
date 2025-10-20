// Fully Structured Matrix Free diagonal extraction (Arbitrary Order) CUDA Kernel (Works for any mesh with repeated elements)
template<typename T> __global__
void get_diagonal_node_basis_cuda_kernel(T* K_single, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, bool* cons_map, T* diag, int max_elem_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    for (int k = 0; k < dof; k++) {
        if (cons_map[i*dof+k]) {
            diag[i*dof+k] = 1;
        } else {
            diag[i*dof+k] = 0;
        }
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int k = 0; k < dof; k++) {
            if (cons_map[i*dof+k]) {
                continue;
            }
            for (int l = 0; l < elements_size*dof; l++) {
                if (elements_flat[start+l/dof]*dof+l%dof == i*dof+k) {
                    atomicAdd(&diag[i*dof+k], K_single[relative_dof*dof+k + (elements_size*dof)*l] * weight);
                }
            }
        }
    }
}

// Unstructred Matrix Free diagonal extraction (Arbitrary Order) CUDA Kernels (full for same element type and flat for mixed element types)
template<typename T> __global__
void get_diagonal_node_basis_full_cuda_kernel(T* Ks, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, bool* cons_map, T* diag, int max_elem_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    for (int k = 0; k < dof; k++) {
        if (cons_map[i*dof+k]) {
            diag[i*dof+k] = 1;
        } else {
            diag[i*dof+k] = 0;
        }
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int k = 0; k < dof; k++) {
            if (cons_map[i*dof+k]) {
                continue;
            }
            for (int l = 0; l < elements_size*dof; l++) {
                if (elements_flat[start+l/dof]*dof+l%dof == i*dof+k) {
                    atomicAdd(&diag[i*dof+k], Ks[elements_ids_j * (elements_size*dof*elements_size*dof) + 
                        (relative_dof*dof+k) * (elements_size*dof) + l] * weight);
                }
            }
        }
    }
}

template<typename T> __global__
void get_diagonal_node_basis_flat_cuda_kernel(T* K_flat, int* elements_flat, int* K_ptr, 
    int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, 
    int n_nodes, int dof, bool* cons_map, T* diag, int max_elem_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }
    
    int n_elements = en-st;

    for (int k = 0; k < dof; k++) {
        if (cons_map[i*dof+k]) {
            diag[i*dof+k] = 1;
        } else {
            diag[i*dof+k] = 0;
        }
    }
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        T weight = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
        
        for (int k = 0; k < dof; k++) {
            if (cons_map[i*dof+k]) {
                continue;
            }
            for (int l = 0; l < size*dof; l++) {
                if (elements_flat[start+l/dof]*dof+l%dof == i*dof+k) {
                    atomicAdd(&diag[i*dof+k], K_flat[k_start + k*size*dof + l] * weight);
                }
            }
        }
    }
}

// general csr diagonal extraction
template<typename T> __global__
void csr_diagonal(const T* __restrict__ Ax, const int*  Ap, const int* __restrict__ Aj, T* out, const int n_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>=n_rows) return;

    for (int j=Ap[i]; j<Ap[i+1]; j++) {
        if (Aj[j] == i) {
            out[i] += Ax[j];
        }
    }
}