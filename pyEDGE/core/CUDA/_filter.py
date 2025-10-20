
from .Kernels import (
    apply_filter_2D_kernel,
    apply_filter_2D_transpose_kernel,
    apply_filter_3D_kernel,
    apply_filter_3D_transpose_kernel,
    get_filter_2D_weights_kernel,
    get_filter_3D_weights_kernel,
)

def apply_filter_2D_cuda(v_in, v_out, nelx, nely, offsets, weights):
    threads_per_block = 256
    blocks_per_grid = (nelx * nely + threads_per_block - 1) // threads_per_block


    apply_filter_2D_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (v_in, v_out, nelx, nely, offsets, weights, len(weights))
    )
    
def apply_filter_3D_cuda(v_in, v_out, nelx, nely, nelz, offsets, weights):
    threads_per_block = 256
    blocks_per_grid = (nelx * nely * nelz + threads_per_block - 1) // threads_per_block


    apply_filter_3D_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (v_in, v_out, nelx, nely, nelz, offsets, weights, len(weights))
    )
    
def get_filter_weights_2D_cuda(nelx, nely, offsets, weights, normalization):
    threads_per_block = 256
    blocks_per_grid = (nelx * nely + threads_per_block - 1) // threads_per_block
    get_filter_2D_weights_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (normalization, nelx, nely, offsets, weights, len(weights))
    )
    return normalization

def get_filter_weights_3D_cuda(nelx, nely, nelz, offsets, weights, normalization):
    threads_per_block = 256
    blocks_per_grid = (nelx * nely * nelz + threads_per_block - 1) // threads_per_block
    get_filter_3D_weights_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (normalization, nelx, nely, nelz, offsets, weights, len(weights))
    )
    return normalization

def apply_filter_2D_transpose_cuda(v_in, v_out, nelx, nely, offsets, weights, normalization):
    threads_per_block = 256
    blocks_per_grid = (nelx * nely + threads_per_block - 1) // threads_per_block
    apply_filter_2D_transpose_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (v_in, v_out, nelx, nely, offsets, weights, normalization, len(weights))
    )

def apply_filter_3D_transpose_cuda(v_in, v_out, nelx, nely, nelz, offsets, weights, normalization):
    threads_per_block = 256
    blocks_per_grid = (nelx * nely * nelz + threads_per_block - 1) // threads_per_block
    apply_filter_3D_transpose_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (v_in, v_out, nelx, nely, nelz, offsets, weights, normalization, len(weights))
    )