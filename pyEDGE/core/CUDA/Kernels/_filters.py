import os
import cupy as cp

with open(os.path.join(os.path.dirname(__file__), "filters.cu"), "r") as f:
    main_code = f.read()
    
_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=[
                                "apply_filter_2D_kernel<double>",
                                "apply_filter_2D_kernel<float>",
                                "apply_filter_3D_kernel<double>",
                                "apply_filter_3D_kernel<float>",
                                "get_filter_2D_weights_kernel<double>",
                                "get_filter_2D_weights_kernel<float>",
                                "get_filter_3D_weights_kernel<double>",
                                "get_filter_3D_weights_kernel<float>",
                                "apply_filter_2D_transpose_kernel<double>",
                                "apply_filter_2D_transpose_kernel<float>",
                                "apply_filter_3D_transpose_kernel<double>",
                                "apply_filter_3D_transpose_kernel<float>",
                            ])

apply_filter_2D_kernel_double = _cuda_module.get_function("apply_filter_2D_kernel<double>")
apply_filter_2D_kernel_float = _cuda_module.get_function("apply_filter_2D_kernel<float>")

def apply_filter_2D_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_2D_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_2D_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



apply_filter_3D_kernel_double = _cuda_module.get_function("apply_filter_3D_kernel<double>")
apply_filter_3D_kernel_float = _cuda_module.get_function("apply_filter_3D_kernel<float>")

def apply_filter_3D_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_3D_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_3D_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_filter_2D_weights_kernel_double = _cuda_module.get_function("get_filter_2D_weights_kernel<double>")
get_filter_2D_weights_kernel_float = _cuda_module.get_function("get_filter_2D_weights_kernel<float>")

def get_filter_2D_weights_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_filter_2D_weights_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_filter_2D_weights_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_filter_3D_weights_kernel_double = _cuda_module.get_function("get_filter_3D_weights_kernel<double>")
get_filter_3D_weights_kernel_float = _cuda_module.get_function("get_filter_3D_weights_kernel<float>")

def get_filter_3D_weights_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_filter_3D_weights_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_filter_3D_weights_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")


apply_filter_2D_transpose_kernel_double = _cuda_module.get_function("apply_filter_2D_transpose_kernel<double>")
apply_filter_2D_transpose_kernel_float = _cuda_module.get_function("apply_filter_2D_transpose_kernel<float>")

def apply_filter_2D_transpose_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_2D_transpose_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_2D_transpose_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")


apply_filter_3D_transpose_kernel_double = _cuda_module.get_function("apply_filter_3D_transpose_kernel<double>")
apply_filter_3D_transpose_kernel_float = _cuda_module.get_function("apply_filter_3D_transpose_kernel<float>")

def apply_filter_3D_transpose_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_3D_transpose_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_3D_transpose_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")