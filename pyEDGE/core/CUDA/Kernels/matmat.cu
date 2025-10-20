// Fully Structured Matrix Free MatMat (Arbitrary Order) CUDA Kernel (Works for any mesh with repeated elements)
// Used with multiplication with identity to assemble the global matrix when needed
template<typename T> __global__
void matmat_node_basis_parallel_kernel(T* K_single, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, int dof, int elements_size,int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_nodes) return;

	int start_ = Cp[i*dof];
	int end_ = Cp[i*dof+1];
	int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
		int e_ind_j = sorter[st+j];
		int elements_ids_j = el_ids[e_ind_j];
		int start = elements_ids_j * elements_size;
		int end = start + elements_size;
		int relative_dof = e_ind_j - start;
        
		for (int l = 0; l < elements_size*dof; l++) {
			int jj = elements_flat[start + l/dof]*dof + l % dof;
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
				for (int ll = start_; ll < end_; ll++) {
					if (Cj[ll] == k) {
						for (int d = 0; d < dof; d++) {
                            Cx[ll + length * d] += K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j];
						}
						break;
					} else if (Cj[ll] == -1) {
						for (int d = 0; d < dof; d++) {
							Cj[ll + length * d] = k;
							Cx[ll + length * d] += K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j];
						}
						break;
					}
				}
			}
		}
	}
}

// When Dirichlet BCs are applied (will set constrained dofs to identity values should be adjusted post solve if not set on RHS)
template<typename T> __global__
void matmat_node_basis_parallel_wcon_kernel(T* K_single, int* elements_flat, int* el_ids, T* weights,
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size,
    int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size, bool* con_map) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_nodes) return;

	// int start_ = Cp[i*dof];
	// int end_ = Cp[i*dof+1];
	// int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
		int e_ind_j = sorter[st+j];
		int elements_ids_j = el_ids[e_ind_j];
		int start = elements_ids_j * elements_size;
		int end = start + elements_size;
		int relative_dof = e_ind_j - start;
        
		for (int l = 0; l < elements_size*dof; l++) {
			int jj = elements_flat[start + l/dof]*dof + l % dof;
            if (con_map[jj] == true) {
                continue;
            }
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
                for (int d = 0; d < dof; d++) {
                    if (con_map[i*dof+d] == true) {
                        continue;
                    }
                    int start_ = Cp[i*dof+d];
                    int end_ = Cp[i*dof+d+1];
                    for (int ll = start_; ll < end_; ll++) {
                        if (Cj[ll] == k) {
                            atomicAdd(&Cx[ll], K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j]);
                            break;
                        } else if (Cj[ll] == -1) {
                            Cj[ll] = k;
                            atomicAdd(&Cx[ll], K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j]);
                            break;
                        }
                    }
                }
			}
		}
	}
 
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            int jj = i*dof+d;
            int count = 0;
            int start_ = Cp[i*dof+d];
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++){
                int k = Bj[kk];
                Cj[start_ + count] = k;
                Cx[start_ + count] = Bx[kk];
                count++;
            }
        }
    }
}

// Unstructred Matrix Free MatMat (Arbitrary Order) CUDA Kernels (full for same element type and flat for mixed element types)
// Used with multiplication with identity to assemble the global matrix when needed
template<typename T> __global__
void matmat_node_basis_full_parallel_kernel(T* Ks, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, 
    int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int start_ = Cp[i*dof];
    int end_ = Cp[i*dof+1];
    int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = start_; ll < end_; ll++) {
                    if (Cj[ll] == k) {
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&Cx[ll + length * d], 
                                Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                (relative_dof*dof+d) * elements_size * dof + l] * 
                                Bx[kk] * weight);
                        }
                        break;
                    } else if (Cj[ll] == -1) {
                        for (int d = 0; d < dof; d++) {
                            Cj[ll + length * d] = k;
                            atomicAdd(&Cx[ll + length * d], 
                                Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                (relative_dof*dof+d) * elements_size * dof + l] * 
                                Bx[kk] * weight);
                        }
                        break;
                    }
                }
            }
        }
    }
}

template<typename T> __global__
void matmat_node_basis_flat_parallel_kernel(T* K_flat, int* elements_flat, int* K_ptr, 
    int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, 
    int n_nodes, int dof, int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, 
    int max_elem_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int start_ = Cp[i*dof];
    int end_ = Cp[i*dof+1];
    int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        T weight = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = start_; ll < end_; ll++) {
                    if (Cj[ll] == k) {
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&Cx[ll + length * d], 
                                K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                        }
                        break;
                    } else if (Cj[ll] == -1) {
                        for (int d = 0; d < dof; d++) {
                            Cj[ll + length * d] = k;
                            atomicAdd(&Cx[ll + length * d], 
                                K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                        }
                        break;
                    }
                }
            }
        }
    }
}

// When Dirichlet BCs are applied (will set constrained dofs to identity values should be adjusted post solve if not set on RHS)
template<typename T> __global__
void matmat_node_basis_full_parallel_wcon_kernel(T* Ks, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, 
    int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size, bool* con_map) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    // int start_ = Cp[i*dof];
    // int end_ = Cp[i*dof+1];
    // int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            if (con_map[jj] == true) {
                continue;
            }
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
                for (int d = 0; d < dof; d++) {
                    if (con_map[i*dof+d] == true) {
                        continue;
                    }

                    int start_ = Cp[i*dof+d];
                    int end_ = Cp[i*dof+d+1];

                    for (int ll = start_; ll < end_; ll++) {
                        if (Cj[ll] == k) {
                            atomicAdd(&Cx[ll], 
                                    Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                    (relative_dof*dof+d) * elements_size * dof + l] * 
                                    Bx[kk] * weight);
                            break;
                        } else if (Cj[ll] == -1) {
                            Cj[ll] = k;
                            atomicAdd(&Cx[ll], 
                                    Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                    (relative_dof*dof+d) * elements_size * dof + l] * 
                                    Bx[kk] * weight);
                            break;
                        }
                    }
                }
			}
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            int jj = i*dof+d;
            int count = 0;
            int start_ = Cp[i*dof+d];
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++){
                int k = Bj[kk];
                Cj[start_ + count] = k;
                Cx[start_ + count] = Bx[kk];
                count++;
            }
        }
    }
}

template<typename T> __global__
void matmat_node_basis_flat_parallel_wcon_kernel(T* K_flat, int* elements_flat, int* K_ptr, 
    int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, 
    int n_nodes, int dof, int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, 
    int max_elem_size, bool* con_map) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    // int start_ = Cp[i*dof];
    // int end_ = Cp[i*dof+1];
    // int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        T weight = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            if (con_map[jj] == true) {
                continue;
            }
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
                for (int d = 0; d < dof; d++) {
                    if (con_map[i*dof+d] == true) {
                        continue;
                    }
                    int start_ = Cp[i*dof+d];
                    int end_ = Cp[i*dof+d+1];
                    for (int ll = start_; ll < end_; ll++) {
                        if (Cj[ll] == k) {
                            atomicAdd(&Cx[ll], 
                                    K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                            break;
                        } else if (Cj[ll] == -1) {
                            Cj[ll] = k;
                            atomicAdd(&Cx[ll], 
                                    K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                            break;
                        }
                    }
                }
			}
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            int jj = i*dof+d;
            int count = 0;
            int start_ = Cp[i*dof+d];
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++){
                int k = Bj[kk];
                Cj[start_ + count] = k;
                Cx[start_ + count] = Bx[kk];
                count++;
            }
        }
    }
}