// Fully Structured Matrix Free Matvec (Arbitrary Order) CUDA Kernel (Works for any mesh with repeated elements)
template<typename T> __global__
void mat_vec_node_basis_parallel_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            out[i*dof+k] = 0;
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof] * weight);
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof];
                }
                out[i*dof+k] += val * weight;
            }
        }
    }
}

// When Dirichlet BCs are present
template<typename T> __global__
void mat_vec_node_basis_parallel_wcon_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            if (con_map[i*dof+k] == true) {
                out[i*dof+k] = vec[i*dof+k];
            }
            else{
                out[i*dof+k] = 0;
            }
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    int kk = elements_flat[start+l/dof]*dof + l % dof;
                    if (con_map[kk] == true) {
                        continue;
                    }
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[kk];
                }
                out[i*dof+k] += val * weight;
            }
        }
    }
}

// Unstructred Matrix Free Matvec (Arbitrary Order) CUDA Kernels (full for same element type and flat for mixed element types)
template<typename T> __global__
void mat_vec_node_basis_parallel_flat_cuda_kernel(T* K_flat, int* elements_flat, int* K_ptr, int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elem_flat_size, T* out) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }

        for (int k=0; k<dof; k++) {
            out[i*dof+k] = 0;
        }
        
        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ptr[elements_ids_j];
            int end = elements_ptr[elements_ids_j+1];
            int size = end - start;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
            
            for (int k=0; k<dof; k++) {
                val = 0.0;
                for (int l=0; l<size*dof; l++) {
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof] * weight);
                    val += K_flat[k_start + k*size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof];
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

template<typename T> __global__
void mat_vec_node_basis_parallel_full_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            out[i*dof+k] = 0;
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = elements_ids_j * elements_size * dof * elements_size * dof + relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof] * weight);
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof];
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

// When Dirichlet BCs are present
template<typename T> __global__
void mat_vec_node_basis_parallel_flat_wcon_cuda_kernel(T* K_flat, int* elements_flat, int* K_ptr, int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elem_flat_size, T* out, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }

        for (int k=0; k<dof; k++) {
            if (con_map[i*dof+k] == true) {
                out[i*dof+k] = vec[i*dof+k];
            }
            else{
                out[i*dof+k] = 0;
            }
        }
        
        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ptr[elements_ids_j];
            int end = elements_ptr[elements_ids_j+1];
            int size = end - start;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
            
            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                val = 0.0;
                for (int l=0; l<size*dof; l++) {
                    int kk = elements_flat[start+l/dof]*dof + l % dof;
                    if (con_map[kk] == true) {
                        continue;
                    }
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*size*dof + l] * vec[kk] * weight);
                    val += K_flat[k_start + k*size*dof + l] * vec[kk];
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

template<typename T> __global__
void mat_vec_node_basis_parallel_full_wcon_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            if (con_map[i*dof+k] == true) {
                out[i*dof+k] = vec[i*dof+k];
            }
            else{
                out[i*dof+k] = 0;
            }
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = elements_ids_j * elements_size * dof * elements_size * dof + relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    int kk = elements_flat[start+l/dof]*dof + l % dof;
                    if (con_map[kk] == true) {
                        continue;
                    }
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[kk];
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*elements_size*dof + l] * vec[kk] * weight);
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}