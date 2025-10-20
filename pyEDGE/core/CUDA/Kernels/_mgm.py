import os
import cupy as cp

with open(os.path.join(os.path.dirname(__file__), "mgm.cu"), "r") as f:
    main_code = f.read()
    
_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=[
                                'get_restricted_3d_l0_nnz_based<double>',
                                'get_restricted_3d_l0_nnz_based<float>',
                                'get_restricted_3d_l1p_nnz_based<double>',
                                'get_restricted_3d_l1p_nnz_based<float>',
                                'get_restricted_2d_l0_nnz_based<double>',
                                'get_restricted_2d_l0_nnz_based<float>',
                                'get_restricted_2d_l1p_nnz_based<double>',
                                'get_restricted_2d_l1p_nnz_based<float>',
                                "restriction_3d_kernel<double>",
                                "restriction_3d_kernel<float>",
                                "restriction_2d_kernel<double>",
                                "restriction_2d_kernel<float>",
                                "prolongation_3d_kernel<double>",
                                "prolongation_3d_kernel<float>",
                                "prolongation_2d_kernel<double>",
                                "prolongation_2d_kernel<float>"
                            ])


get_restricted_2d_l0_nnz_based_double = _cuda_module.get_function("get_restricted_2d_l0_nnz_based<double>")
get_restricted_2d_l0_nnz_based_float = _cuda_module.get_function("get_restricted_2d_l0_nnz_based<float>")

def get_restricted_2d_l0_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_2d_l0_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_2d_l0_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")
    
get_restricted_2d_l1p_nnz_based_double = _cuda_module.get_function("get_restricted_2d_l1p_nnz_based<double>")
get_restricted_2d_l1p_nnz_based_float = _cuda_module.get_function("get_restricted_2d_l1p_nnz_based<float>")

def get_restricted_2d_l1p_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_2d_l1p_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_2d_l1p_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")

get_restricted_3d_l1p_nnz_based_double = _cuda_module.get_function("get_restricted_3d_l1p_nnz_based<double>")
get_restricted_3d_l1p_nnz_based_float = _cuda_module.get_function("get_restricted_3d_l1p_nnz_based<float>")

def get_restricted_3d_l1p_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_3d_l1p_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_3d_l1p_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")

get_restricted_3d_l0_nnz_based_double = _cuda_module.get_function("get_restricted_3d_l0_nnz_based<double>")
get_restricted_3d_l0_nnz_based_float = _cuda_module.get_function("get_restricted_3d_l0_nnz_based<float>")

def get_restricted_3d_l0_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_3d_l0_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_3d_l0_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")
    
restriction_3d_kernel_double = _cuda_module.get_function("restriction_3d_kernel<double>")
restriction_3d_kernel_float = _cuda_module.get_function("restriction_3d_kernel<float>")

def restriction_3d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return restriction_3d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return restriction_3d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



restriction_2d_kernel_double = _cuda_module.get_function("restriction_2d_kernel<double>")
restriction_2d_kernel_float = _cuda_module.get_function("restriction_2d_kernel<float>")

def restriction_2d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return restriction_2d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return restriction_2d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



prolongation_3d_kernel_double = _cuda_module.get_function("prolongation_3d_kernel<double>")
prolongation_3d_kernel_float = _cuda_module.get_function("prolongation_3d_kernel<float>")

def prolongation_3d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return prolongation_3d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return prolongation_3d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



prolongation_2d_kernel_double = _cuda_module.get_function("prolongation_2d_kernel<double>")
prolongation_2d_kernel_float = _cuda_module.get_function("prolongation_2d_kernel<float>")

def prolongation_2d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return prolongation_2d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return prolongation_2d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")