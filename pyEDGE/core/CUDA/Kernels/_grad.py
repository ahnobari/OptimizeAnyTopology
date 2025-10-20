import os
import cupy as cp

with open(os.path.join(os.path.dirname(__file__), "grad.cu"), "r") as f:
    main_code = f.read()
    
_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=[
                                "process_dk_kernel_cuda<double>",
                                "process_dk_kernel_cuda<float>",
                                "process_dk_full_kernel_cuda<double>",
                                "process_dk_full_kernel_cuda<float>",
                                "process_dk_flat_kernel_cuda<double>",
                                "process_dk_flat_kernel_cuda<float>",
                            ])

process_dk_kernel_cuda_double = _cuda_module.get_function("process_dk_kernel_cuda<double>")
process_dk_kernel_cuda_float = _cuda_module.get_function("process_dk_kernel_cuda<float>")

def process_dk_kernel_cuda(*args):
    if args[-1][0].dtype == cp.float64:
        return process_dk_kernel_cuda_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return process_dk_kernel_cuda_float(*args)
    else:
        raise ValueError("Unsupported dtype")



process_dk_full_kernel_cuda_double = _cuda_module.get_function("process_dk_full_kernel_cuda<double>")
process_dk_full_kernel_cuda_float = _cuda_module.get_function("process_dk_full_kernel_cuda<float>")

def process_dk_full_kernel_cuda(*args):
    if args[-1][0].dtype == cp.float64:
        return process_dk_full_kernel_cuda_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return process_dk_full_kernel_cuda_float(*args)
    else:
        raise ValueError("Unsupported dtype")



process_dk_flat_kernel_cuda_double = _cuda_module.get_function("process_dk_flat_kernel_cuda<double>")
process_dk_flat_kernel_cuda_float = _cuda_module.get_function("process_dk_flat_kernel_cuda<float>")

def process_dk_flat_kernel_cuda(*args):
    if args[-1][0].dtype == cp.float64:
        return process_dk_flat_kernel_cuda_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return process_dk_flat_kernel_cuda_float(*args)
    else:
        raise ValueError("Unsupported dtype")