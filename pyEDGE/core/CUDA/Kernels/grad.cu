// Sensativity analysis kernels
template<typename T> __global__
void process_dk_kernel_cuda(T* K_flat, int* elements_flat, T* U, int dof, int elements_size, int nel, T* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nel){
        out[i] = 0;
        T val = 0;
        for(int j=0; j<dof*elements_size; j++){
            val = 0;
            for(int k=0; k<dof*elements_size; k++){
                val += K_flat[j*dof*elements_size + k] * U[elements_flat[i*elements_size + k/dof]*dof + k % dof];
            }
            out[i] -= val * U[elements_flat[i*elements_size + j/dof]*dof + j % dof];
        }
    }
}

template<typename T> __global__
void process_dk_full_kernel_cuda(T* K_flat, int* elements_flat, T* U, int dof, int elements_size, int nel, T* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nel){
        out[i] = 0;
        T val = 0;
        int k_start = dof*elements_size*dof*elements_size * i;
        for(int j=0; j<dof*elements_size; j++){
            val = 0;
            for(int k=0; k<dof*elements_size; k++){
                val += K_flat[k_start + j*dof*elements_size + k] * U[elements_flat[i*elements_size + k/dof]*dof + k % dof];
            }
            out[i] -= val * U[elements_flat[i*elements_size + j/dof]*dof + j % dof];
        }
    }
}

template<typename T> __global__
void process_dk_flat_kernel_cuda(T* K_flat, int* elements_flat, int* K_ptr, int* elements_ptr, T* U, int dof, int nel, T* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nel){
        out[i] = 0;
        T val = 0;
        int k_start = K_ptr[i];
        int size = elements_ptr[i+1] - elements_ptr[i];
        
        for(int j=0; j<dof*size; j++){
            val = 0;
            for(int k=0; k<dof*size; k++){
                val += K_flat[k_start + j*dof*size + k] * U[elements_flat[elements_ptr[i] + k/dof]*dof + k % dof];
            }
            out[i] -= val * U[elements_flat[elements_ptr[i] + j/dof]*dof + j % dof];
        }
    }
}