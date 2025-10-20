from ..solver import Solver
from ..MaterialModels import PenalizedMultiMaterial, SingleMaterial
from ..geometry import generate_structured_mesh


def generate_topology(config, multi=True, solver_instance=None, gpu=False):
    dims, loads, load_v, bcs, bc_c, E_mat, vf = config
    if multi:
        material_model = PenalizedMultiMaterial(
            n_material=E_mat.shape[0],
            E_mat=E_mat,
            mass=vf,
            penalty=3,
            penalty_schedule=lambda p, i: (
                3.0 if i >= 75 else 1 + (i // 15) * (p - 1) / 5
            ),
        )
    else:
        material_model = SingleMaterial(
            penalty=3,
            volume_fraction = vf.flatten()[0],
            penalty_schedule=lambda p, i: (
                3.0 if i >= 75 else 1 + (i // 15) * (p - 1) / 5
            ),
        )

    if solver_instance is None:
        elements, nodes = generate_structured_mesh(dims, dims)
        mesh = (nodes / dims.max(), elements)
        solver_instance = Solver(
            mesh=mesh,
            material_model=material_model,
            structured=True,
            max_iter=1000,
            move=0.2,
            ch_tol=1e-4,
            fun_tol=1e-5,
        )
    else:
        solver_instance.material_model = material_model

    if gpu:
        solver_instance.compute_engine = "gpu"
        solver_instance.solver = "gpu"

    solver_instance.reset_BC()
    solver_instance.reset_F()
    solver_instance.add_BCs(bcs / dims.max(), bc_c)
    solver_instance.add_Forces(loads / dims.max(), load_v)

    rho, status, hist = solver_instance.optimize(
        save_change_history=False, save_rho_history=False, save_comp_history=True
    )

    return rho, status, hist
