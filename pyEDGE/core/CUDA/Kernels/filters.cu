template<typename T> __global__
void apply_filter_2D_kernel(T* v_in, T* v_out, int nelx, int nely, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelx * nely) {
        int ix = i / nely;
        int iy = i % nely;

        T sum_weighted_values = 0.0;
        T sum_weights = 0.0;
        
        for (int k = 0; k < offsets_len; k++) {
            int dx = offsets[2 * k];
            int dy = offsets[2 * k + 1];
            int ix_n = ix + dx;
            int iy_n = iy + dy;
            
            if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely) {
                int neighbor_idx = ix_n * nely + iy_n;
                T weight = weights[k];
                sum_weighted_values += weight * v_in[neighbor_idx];
                sum_weights += weight;
            }
        }
        
        v_out[i] = sum_weights > 0 ? sum_weighted_values / sum_weights : 0.0f;
    }
}

template<typename T> __global__
void apply_filter_3D_kernel(T* v_in, T* v_out, int nelx, int nely, int nelz, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelx * nely * nelz) {
        int ix = i / (nely * nelz);
        int iy = (i / nelz) % nely;
        int iz = i % nelz;

        T sum_weighted_values = 0.0;
        T sum_weights = 0.0;
        
        for (int k = 0; k < offsets_len; k++) {
            int dx = offsets[3 * k];
            int dy = offsets[3 * k + 1];
            int dz = offsets[3 * k + 2];
            int ix_n = ix + dx;
            int iy_n = iy + dy;
            int iz_n = iz + dz;
            
            if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely && iz_n >= 0 && iz_n < nelz) {
                int neighbor_idx = ix_n * nely * nelz + iy_n * nelz + iz_n;
                T weight = weights[k];
                sum_weighted_values += weight * v_in[neighbor_idx];
                sum_weights += weight;
            }
        }
        
        v_out[i] = sum_weights > 0 ? sum_weighted_values / sum_weights : 0.0f;
    }
}

template<typename T> __global__
void get_filter_2D_weights_kernel(T* normalization, int nelx, int nely, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely) return;
    
    int ix = i / nely;
    int iy = i % nely;
    
    T sum_weights = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[2 * k];
        int dy = offsets[2 * k + 1];
        int ix_n = ix + dx;
        int iy_n = iy + dy;
        
        if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely) {
            sum_weights += weights[k];
        }
    }
    normalization[i] = sum_weights;
}

template<typename T> __global__
void get_filter_3D_weights_kernel(T* normalization, int nelx, int nely, int nelz, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely * nelz) return;
    
    int ix = i / (nely * nelz);
    int iy = (i / nelz) % nely;
    int iz = i % nelz;
    
    T sum_weights = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[3 * k];
        int dy = offsets[3 * k + 1];
        int dz = offsets[3 * k + 2];
        int ix_n = ix + dx;
        int iy_n = iy + dy;
        int iz_n = iz + dz;
        
        if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely && iz_n >= 0 && iz_n < nelz) {
            sum_weights += weights[k];
        }
    }
    normalization[i] = sum_weights;
}

template<typename T> __global__
void apply_filter_2D_transpose_kernel(T* v_in, T* v_out, int nelx, int nely, int* offsets, T* weights, T* normalization, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely) return;
    
    int ix = i / nely;
    int iy = i % nely;
    
    T sum_value = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[2 * k];
        int dy = offsets[2 * k + 1];
        int jx = ix + dx;
        int jy = iy + dy;
        
        if (jx >= 0 && jx < nelx && jy >= 0 && jy < nely) {
            int j_idx = jx * nely + jy;
            sum_value += (weights[k] / normalization[j_idx]) * v_in[j_idx];
        }
    }
    v_out[i] = sum_value;
}

template<typename T> __global__
void apply_filter_3D_transpose_kernel(T* v_in, T* v_out, int nelx, int nely, int nelz, int* offsets, T* weights, T* normalization, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely * nelz) return;
    
    int ix = i / (nely * nelz);
    int iy = (i / nelz) % nely;
    int iz = i % nelz;
    
    T sum_value = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[3 * k];
        int dy = offsets[3 * k + 1];
        int dz = offsets[3 * k + 2];
        int jx = ix + dx;
        int jy = iy + dy;
        int jz = iz + dz;
        
        if (jx >= 0 && jx < nelx && jy >= 0 && jy < nely && jz >= 0 && jz < nelz) {
            int j_idx = jx * nely * nelz + jy * nelz + jz;
            sum_value += (weights[k] / normalization[j_idx]) * v_in[j_idx];
        }
    }
    v_out[i] = sum_value;
}