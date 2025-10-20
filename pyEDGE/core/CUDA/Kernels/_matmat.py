import os
import cupy as cp
from functools import lru_cache

with open(os.path.join(os.path.dirname(__file__), "matmat.cu"), "r") as f:
    main_code = f.read()
    
_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=[
                                "matmat_node_basis_parallel_kernel<double>",
                                "matmat_node_basis_parallel_kernel<float>",
                                "matmat_node_basis_full_parallel_kernel<double>",
                                "matmat_node_basis_full_parallel_kernel<float>",
                                "matmat_node_basis_flat_parallel_kernel<double>",
                                "matmat_node_basis_flat_parallel_kernel<float>",
                                "matmat_node_basis_full_parallel_wcon_kernel<double>",
                                "matmat_node_basis_full_parallel_wcon_kernel<float>",
                                "matmat_node_basis_flat_parallel_wcon_kernel<double>",
                                "matmat_node_basis_flat_parallel_wcon_kernel<float>",
                                "matmat_node_basis_parallel_wcon_kernel<double>",
                                "matmat_node_basis_parallel_wcon_kernel<float>",
                            ])


matmat_node_basis_parallel_kernel_double = _cuda_module.get_function("matmat_node_basis_parallel_kernel<double>")
matmat_node_basis_parallel_kernel_float = _cuda_module.get_function("matmat_node_basis_parallel_kernel<float>")

def matmat_node_basis_parallel_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_parallel_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_parallel_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_full_parallel_kernel_double = _cuda_module.get_function("matmat_node_basis_full_parallel_kernel<double>")
matmat_node_basis_full_parallel_kernel_float = _cuda_module.get_function("matmat_node_basis_full_parallel_kernel<float>")

def matmat_node_basis_full_parallel_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_full_parallel_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_full_parallel_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_flat_parallel_kernel_double = _cuda_module.get_function("matmat_node_basis_flat_parallel_kernel<double>")
matmat_node_basis_flat_parallel_kernel_float = _cuda_module.get_function("matmat_node_basis_flat_parallel_kernel<float>")

def matmat_node_basis_flat_parallel_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_flat_parallel_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_flat_parallel_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")
    
matmat_node_basis_full_parallel_wcon_kernel_double = _cuda_module.get_function("matmat_node_basis_full_parallel_wcon_kernel<double>")
matmat_node_basis_full_parallel_wcon_kernel_float = _cuda_module.get_function("matmat_node_basis_full_parallel_wcon_kernel<float>")

def matmat_node_basis_full_parallel_wcon_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_full_parallel_wcon_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_full_parallel_wcon_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_flat_parallel_wcon_kernel_double = _cuda_module.get_function("matmat_node_basis_flat_parallel_wcon_kernel<double>")
matmat_node_basis_flat_parallel_wcon_kernel_float = _cuda_module.get_function("matmat_node_basis_flat_parallel_wcon_kernel<float>")

def matmat_node_basis_flat_parallel_wcon_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_flat_parallel_wcon_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_flat_parallel_wcon_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_parallel_wcon_kernel_double = _cuda_module.get_function("matmat_node_basis_parallel_wcon_kernel<double>")
matmat_node_basis_parallel_wcon_kernel_float = _cuda_module.get_function("matmat_node_basis_parallel_wcon_kernel<float>")

def matmat_node_basis_parallel_wcon_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_parallel_wcon_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_parallel_wcon_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")
    
    
matmat_node_basis_nnz_per_row_kernel_code = '''
extern "C" __global__
void matmat_node_basis_nnz_per_row_kernel(int* elements_flat, int* el_ids, int* sorter, 
    int* node_ids, int n_nodes, int dof, int elements_size, int n_col, 
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                        if (ll == max_nnz-1){
                            printf("Error: Exceeded maximum number of nonzeros per row");
                            return;
                        }
                    }
                }
            }
        }
    }
}
'''

matmat_node_basis_flat_nnz_per_row_kernel_code = '''
extern "C" __global__
void matmat_node_basis_flat_nnz_per_row_kernel(int* elements_flat, int* elements_ptr, 
    int* el_ids, int* sorter, int* node_ids, int n_nodes, int dof, int n_col,
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                    }
                    if (ll == max_nnz-1){
                        printf("Error: Exceeded maximum number of nonzeros per row");
                        return;
                    }
                }
            }
        }
    }
}
'''

@lru_cache(maxsize=None)
def get_matmat_node_basis_nnz_per_row_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_nnz_per_row_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_nnz_per_row_kernel'
    )

def matmat_node_basis_nnz_per_row_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_nnz_per_row_kernel(max_nnz)
    kernel(B, T, A)


@lru_cache(maxsize=None)
def get_matmat_node_basis_flat_nnz_per_row_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_flat_nnz_per_row_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_flat_nnz_per_row_kernel'
    )

def matmat_node_basis_flat_nnz_per_row_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_flat_nnz_per_row_kernel(max_nnz)
    kernel(B, T, A)



matmat_node_basis_nnz_per_row_wcon_kernel_code = '''
extern "C" __global__
void matmat_node_basis_nnz_per_row_wcon_kernel(int* elements_flat, int* el_ids, int* sorter, 
    int* node_ids, int n_nodes, int dof, int elements_size, int n_col, 
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            if (con_map[jj] == true) continue;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                        if (ll == max_nnz-1){
                            printf("Error: Exceeded maximum number of nonzeros per row");
                            return;
                        }
                    }
                }
            }
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            nnz_per_row[i*dof+d] = Bp[i*dof+d+1] - Bp[i*dof+d];
        }
    }
}
'''

matmat_node_basis_flat_nnz_per_row_wcon_kernel_code = '''
extern "C" __global__
void matmat_node_basis_flat_nnz_per_row_kernel(int* elements_flat, int* elements_ptr, 
    int* el_ids, int* sorter, int* node_ids, int n_nodes, int dof, int n_col,
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            if (con_map[jj] == true) continue;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                    }
                    if (ll == max_nnz-1){
                        printf("Error: Exceeded maximum number of nonzeros per row");
                        return;
                    }
                }
            }
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            nnz_per_row[i*dof+d] = Bp[i*dof+d+1] - Bp[i*dof+d];
        }
    }
}
'''

@lru_cache(maxsize=None)
def get_matmat_node_basis_nnz_per_row_wcon_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_nnz_per_row_wcon_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_nnz_per_row_wcon_kernel'
    )

def matmat_node_basis_nnz_per_row_wcon_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_nnz_per_row_wcon_kernel(max_nnz)
    kernel(B, T, A)

@lru_cache(maxsize=None)
def get_matmat_node_basis_flat_nnz_per_row_wcon_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_flat_nnz_per_row_wcon_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_flat_nnz_per_row_kernel'
    )

def matmat_node_basis_flat_nnz_per_row_wcon_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_flat_nnz_per_row_wcon_kernel(max_nnz)
    kernel(B, T, A)