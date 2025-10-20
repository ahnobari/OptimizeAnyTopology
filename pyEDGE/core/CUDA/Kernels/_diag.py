import os
import cupy as cp

with open(os.path.join(os.path.dirname(__file__), "diag.cu"), "r") as f:
    main_code = f.read()
    
_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=[
                                "get_diagonal_node_basis_cuda_kernel<double>",
                                "get_diagonal_node_basis_cuda_kernel<float>",
                                "get_diagonal_node_basis_full_cuda_kernel<double>",
                                "get_diagonal_node_basis_full_cuda_kernel<float>",
                                "get_diagonal_node_basis_flat_cuda_kernel<double>",
                                "get_diagonal_node_basis_flat_cuda_kernel<float>",
                                'csr_diagonal<double>',
                                'csr_diagonal<float>'
                            ])

csr_diagonal_double = _cuda_module.get_function("csr_diagonal<double>")
csr_diagonal_float = _cuda_module.get_function("csr_diagonal<float>")

def csr_diagonal(*args):
    if args[-1][0].dtype == cp.float64:
        return csr_diagonal_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return csr_diagonal_float(*args)
    else:
        raise ValueError("Unsupported dtype")
    
get_diagonal_node_basis_cuda_kernel_double = _cuda_module.get_function("get_diagonal_node_basis_cuda_kernel<double>")
get_diagonal_node_basis_cuda_kernel_float = _cuda_module.get_function("get_diagonal_node_basis_cuda_kernel<float>")

def get_diagonal_node_basis_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_diagonal_node_basis_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_diagonal_node_basis_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_diagonal_node_basis_full_cuda_kernel_double = _cuda_module.get_function("get_diagonal_node_basis_full_cuda_kernel<double>")
get_diagonal_node_basis_full_cuda_kernel_float = _cuda_module.get_function("get_diagonal_node_basis_full_cuda_kernel<float>")

def get_diagonal_node_basis_full_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_diagonal_node_basis_full_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_diagonal_node_basis_full_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_diagonal_node_basis_flat_cuda_kernel_double = _cuda_module.get_function("get_diagonal_node_basis_flat_cuda_kernel<double>")
get_diagonal_node_basis_flat_cuda_kernel_float = _cuda_module.get_function("get_diagonal_node_basis_flat_cuda_kernel<float>")

def get_diagonal_node_basis_flat_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_diagonal_node_basis_flat_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_diagonal_node_basis_flat_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")