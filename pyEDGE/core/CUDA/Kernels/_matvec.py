import os
import cupy as cp

with open(os.path.join(os.path.dirname(__file__), "matvec.cu"), "r") as f:
    main_code = f.read()
    
_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=[
                                "mat_vec_node_basis_parallel_flat_cuda_kernel<double>",
                                "mat_vec_node_basis_parallel_flat_cuda_kernel<float>",
                                "mat_vec_node_basis_parallel_full_cuda_kernel<double>",
                                "mat_vec_node_basis_parallel_full_cuda_kernel<float>",
                                "mat_vec_node_basis_parallel_cuda_kernel<double>",
                                "mat_vec_node_basis_parallel_cuda_kernel<float>",
                                "mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<double>",
                                "mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<float>",
                                "mat_vec_node_basis_parallel_full_wcon_cuda_kernel<double>",
                                "mat_vec_node_basis_parallel_full_wcon_cuda_kernel<float>",
                                "mat_vec_node_basis_parallel_wcon_cuda_kernel<double>",
                                "mat_vec_node_basis_parallel_wcon_cuda_kernel<float>",
                            ])

mat_vec_node_basis_parallel_flat_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_cuda_kernel<double>")
mat_vec_node_basis_parallel_flat_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_cuda_kernel<float>")

def mat_vec_node_basis_parallel_flat_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_flat_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_flat_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_full_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_full_cuda_kernel<double>")
mat_vec_node_basis_parallel_full_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_full_cuda_kernel<float>")

def mat_vec_node_basis_parallel_full_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_full_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_full_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_cuda_kernel<double>")
mat_vec_node_basis_parallel_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_cuda_kernel<float>")

def mat_vec_node_basis_parallel_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")
    
mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<double>")
mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<float>")

def mat_vec_node_basis_parallel_flat_wcon_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_full_wcon_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_full_wcon_cuda_kernel<double>")
mat_vec_node_basis_parallel_full_wcon_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_full_wcon_cuda_kernel<float>")

def mat_vec_node_basis_parallel_full_wcon_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_full_wcon_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_full_wcon_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_wcon_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_wcon_cuda_kernel<double>")
mat_vec_node_basis_parallel_wcon_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_wcon_cuda_kernel<float>")

def mat_vec_node_basis_parallel_wcon_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_wcon_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_wcon_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")