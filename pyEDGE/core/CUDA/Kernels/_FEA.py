import os
import cupy as cp

with open(os.path.join(os.path.dirname(__file__), "FEA.cu"), "r") as f:
    main_code = f.read()
    
_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=[
                                "FEA_locals_node_basis_parallel_cuda_kernel<double>",
                                "FEA_locals_node_basis_parallel_cuda_kernel<float>",
                                "FEA_locals_node_basis_parallel_full_cuda_kernel<double>",
                                "FEA_locals_node_basis_parallel_full_cuda_kernel<float>",
                                "FEA_locals_node_basis_parallel_flat_cuda_kernel<double>",
                                "FEA_locals_node_basis_parallel_flat_cuda_kernel<float>",
                            ])

FEA_locals_node_basis_parallel_cuda_kernel_double = _cuda_module.get_function("FEA_locals_node_basis_parallel_cuda_kernel<double>")
FEA_locals_node_basis_parallel_cuda_kernel_float = _cuda_module.get_function("FEA_locals_node_basis_parallel_cuda_kernel<float>")

def FEA_locals_node_basis_parallel_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return FEA_locals_node_basis_parallel_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return FEA_locals_node_basis_parallel_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")

FEA_locals_node_basis_parallel_full_cuda_kernel_double = _cuda_module.get_function("FEA_locals_node_basis_parallel_full_cuda_kernel<double>")
FEA_locals_node_basis_parallel_full_cuda_kernel_float = _cuda_module.get_function("FEA_locals_node_basis_parallel_full_cuda_kernel<float>")

def FEA_locals_node_basis_parallel_full_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return FEA_locals_node_basis_parallel_full_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return FEA_locals_node_basis_parallel_full_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")

FEA_locals_node_basis_parallel_flat_cuda_kernel_double = _cuda_module.get_function("FEA_locals_node_basis_parallel_flat_cuda_kernel<double>")
FEA_locals_node_basis_parallel_flat_cuda_kernel_float = _cuda_module.get_function("FEA_locals_node_basis_parallel_flat_cuda_kernel<float>")

def FEA_locals_node_basis_parallel_flat_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return FEA_locals_node_basis_parallel_flat_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return FEA_locals_node_basis_parallel_flat_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")