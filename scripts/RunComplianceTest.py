import os 
import pickle
import torch
import numpy as np
from pyEDGE.CUDA import *
from tqdm.auto import tqdm, trange
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import itertools
import tabulate
import cupy

args = argparse.ArgumentParser()
args.add_argument("--splits", type=str, default="test+NITO_test", help="+ separated dataset splits to use, default 'test+NITO_test'")
args.add_argument("--start_idx", type=int, default=0, help="start index of dataset to evaluate. default 0")
args.add_argument("--end_idx", type=int, default=None, help="end index of dataset to evaluate. default None (till end of dataset)")
args.add_argument("--generations_path", type=str, default="GeneratedSamples.pkl", help="Path to the generated samples. Default: 'GeneratedSamples.pkl'")
args.add_argument("--output_path", type=str, default='ComplianceResults.pkl', help="Path to save the compliance results. Default: 'ComplianceResults.pkl'")
args.add_argument("--few_steps", action="store_true", help="Also apply few optimization steps from generated samples and measure compliance")
args.add_argument("--num_workers_per_gpu", type=int, default=1, help="Number of parallel workers to use per GPU. Default: 1")
args.add_argument("--select_generations", action="store_true", help="If set, start and end idx will be used on generations as well to select a subset of generations to evaluate.")
args = args.parse_args()

def simulate_optimize(sample, generations, device_idx = 0, few_steps=False):
    with cupy.cuda.Device(device_idx):
        shape = sample['topology'].size[::-1]
        gt_rho = np.array(sample['topology']).astype(np.float64)
        loads = np.array(sample['loads'])
        bc = np.array(sample['boundary conditions'])
        vf = sample['volume fraction']
        
        mesh = StructuredMesh2D(shape[0], shape[1], shape[0]/max(shape), shape[1]/max(shape))
        kernel = StructuredStiffnessKernel(mesh)
        filter = StructuredFilter2D(mesh, 1.5)
        solver = SPSOLVE(kernel)
        FE = FiniteElement(mesh, kernel, solver)
        
        FE.reset_dirichlet_boundary_conditions()
        FE.reset_forces()
        FE.add_dirichlet_boundary_condition(positions=bc[:,0:2], dofs=bc[:,2:])
        FE.add_point_forces(positions=loads[:,0:2], forces=loads[:,2:])

        problem = MinimumCompliance(FE, filter, void=1e-6, volume_fraction=[vf], heavyside=False)
        if few_steps:
            optimizer = PGD(problem)
        
        problem.desvars = gt_rho.reshape(-1) > 0
        gt_compliance = float(problem.FEA()['compliance'])
        
        compliances = []
        volume_fractions = []
        
        if few_steps:
            SIMP5_compliances = []
            SIMP5_volume_fractions = []
            
            SIMP10_compliances = []
            SIMP10_volume_fractions = []
        
        for i in range(len(generations)):
            problem.desvars = generations[i].float().numpy().reshape(-1)
            comp = float(problem.FEA()['compliance'])
            compliances.append(comp)
            volume_fractions.append((problem.desvars>0.5).sum()/problem.N())
            
            if few_steps:
                problem.set_desvars(cupy.array(generations[i].float().numpy().reshape(-1).astype(np.float64)))
                for it in range(10):
                    optimizer.iter()
                    if it == 4:
                        comp = float(problem.FEA()['compliance'].get())
                        SIMP5_compliances.append(comp)
                        SIMP5_volume_fractions.append((problem.desvars>0.5).sum().get()/problem.N())
                        
                    if it == 9:
                        comp = float(problem.FEA()['compliance'].get())
                        SIMP10_compliances.append(comp)
                        SIMP10_volume_fractions.append((problem.desvars>0.5).sum().get()/problem.N())
            
        compliances = np.array(compliances)
        volume_fractions = np.array(volume_fractions)
        
        CEs = (compliances - gt_compliance)/gt_compliance * 100
        VFEs = (volume_fractions - vf)/vf * 100
        
        if few_steps:
            SIMP5_compliances = np.array(SIMP5_compliances)
            SIMP5_volume_fractions = np.array(SIMP5_volume_fractions)
            
            SIMP10_compliances = np.array(SIMP10_compliances)
            SIMP10_volume_fractions = np.array(SIMP10_volume_fractions)
            
            SIMP5_CEs = (SIMP5_compliances - gt_compliance)/gt_compliance * 100
            SIMP5_VFEs = (SIMP5_volume_fractions - vf)/vf * 100
            
            SIMP10_CEs = (SIMP10_compliances - gt_compliance)/gt_compliance * 100
            SIMP10_VFEs = (SIMP10_volume_fractions - vf)/vf * 100
            
            return CEs, VFEs, SIMP5_CEs, SIMP5_VFEs, SIMP10_CEs, SIMP10_VFEs
        else:
            return CEs, VFEs
    
def run_eval_parallel_executor(data, generated_samples, device_list, few_steps=False, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()
    
    n_samples = min(len(data), len(generated_samples))
    results = [None] * n_samples
    errors = {}
    
    print(f"Processing {n_samples} samples using {n_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(simulate_optimize, data[i], generated_samples[i], device_list[i], few_steps): i
            for i in range(n_samples)
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_idx), total=n_samples, desc="Evaluating samples"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                errors[idx] = str(e)
                print(f"Error in sample {idx}: {e}")
    
    return results, errors

def main():
    print("Loading data and generations ...")
    generated_samples = pickle.load(open(args.generations_path, "rb"))
    if args.select_generations:
        generated_samples = generated_samples[args.start_idx:args.end_idx] if args.end_idx is not None else generated_samples[args.start_idx:]
    data = load_dataset("OpenTO/OpenTO", split=args.splits)
    data = data.select(range(args.start_idx, args.end_idx) if args.end_idx is not None else range(args.start_idx, len(data)))
    print(f"Data samples: {len(data)}, Generated samples: {len(generated_samples)}")
    assert len(data) == len(generated_samples), "Data and generated samples length mismatch!"
    
    num_samples = len(generated_samples[0])
    n_devices = torch.cuda.device_count()
    print(f"Using {n_devices} GPUs for evaluation ...")
    device_list = [i % n_devices for i in range(len(data))]

    results, errors = run_eval_parallel_executor(data, generated_samples, device_list, few_steps=args.few_steps, n_workers=args.num_workers_per_gpu * n_devices)

    # process results
    CEs = []
    VFEs = []
    if args.few_steps:
        SIMP5_CEs = []
        SIMP5_VFEs = []
        SIMP10_CEs = []
        SIMP10_VFEs = []
    
    for res in results:
        if args.few_steps:
            ce, vfe, simp5_ce, simp5_vfe, simp10_ce, simp10_vfe = res
            if simp5_ce is not None:
                SIMP5_CEs.append(simp5_ce)
            else:
                SIMP5_CEs.append(np.ones_like(SIMP10_CEs[0]) * np.inf)
            if simp5_vfe is not None:
                SIMP5_VFEs.append(simp5_vfe)
            else:
                SIMP5_VFEs.append(np.ones_like(SIMP10_VFEs[0]) * np.inf)
            if simp10_ce is not None:
                SIMP10_CEs.append(simp10_ce)
            else:
                SIMP10_CEs.append(np.ones_like(SIMP10_CEs[0]) * np.inf)
            if simp10_vfe is not None:
                SIMP10_VFEs.append(simp10_vfe)
            else:
                SIMP10_VFEs.append(np.ones_like(SIMP10_VFEs[0]) * np.inf)
                
        else:
            ce, vfe = res
            
        if ce is not None:
            CEs.append(ce)
        else:
            CEs.append(np.ones_like(CEs[0]) * np.inf)
        if vfe is not None:
            VFEs.append(vfe)
        else:
            VFEs.append(np.ones_like(VFEs[0]) * np.inf)
            
    CEs = np.array(CEs)
    VFEs = np.array(VFEs)
    if args.few_steps:
        SIMP5_CEs = np.array(SIMP5_CEs)
        SIMP5_VFEs = np.array(SIMP5_VFEs)
        SIMP10_CEs = np.array(SIMP10_CEs)
        SIMP10_VFEs = np.array(SIMP10_VFEs)
        
    results_dict = {
        "CEs": CEs,
        "VFEs": VFEs,
    }
    if args.few_steps:
        results_dict.update({
            "SIMP5_CEs": SIMP5_CEs,
            "SIMP5_VFEs": SIMP5_VFEs,
            "SIMP10_CEs": SIMP10_CEs,
            "SIMP10_VFEs": SIMP10_VFEs,
        })
    
    pickle.dump(results_dict, open(args.output_path, "wb"))
    print(f"Saved compliance results to {args.output_path}")
    
    print("Computing statistics ...")
    # gather statistics for reporting
    stats = {}
    
    # failures
    passed = CEs < 100
    
    if args.few_steps:
        simp5_passed = SIMP5_CEs < 100
        simp10_passed = SIMP10_CEs < 100
    
    for i in range(num_samples):
        if i == 0:
            average_pass_rate = passed.mean(0).mean()
            stats['Pass @ 1'] = average_pass_rate
            average_valid_ce = np.where(passed, CEs, 0.0).sum(axis=0) / (passed.sum(axis=0))
            stats['Avg CE @ 1'] = average_valid_ce.mean()
            stats['STD CE @ 1'] = np.std(CEs[passed])
            stats['Median CE @ 1'] = np.median(CEs[passed])
            average_valid_vfe = np.where(passed, VFEs, 0.0).sum(axis=0) / (passed.sum(axis=0))
            stats['Avg VFE @ 1'] = average_valid_vfe.mean()
            stats['STD VFE @ 1'] = np.std(VFEs[passed])
            stats['Median VFE @ 1'] = np.median(VFEs[passed])

            if args.few_steps:
                stats['+5 Pass @ 1'] = simp5_passed.mean(0).mean()
                stats['+10 Pass @ 1'] = simp10_passed.mean(0).mean()
                
                average_valid_ce = np.where(simp5_passed, SIMP5_CEs, 0.0).sum(axis=0) / (simp5_passed.sum(axis=0))
                stats['+5 Avg CE @ 1'] = average_valid_ce.mean()
                stats['+5 STD CE @ 1'] = np.std(SIMP5_CEs[simp5_passed])
                stats['+5 Median CE @ 1'] = np.median(SIMP5_CEs[simp5_passed])
                average_valid_vfe = np.where(simp5_passed, SIMP5_VFEs, 0.0).sum(axis=0) / (simp5_passed.sum(axis=0))
                stats['+5 Avg VFE @ 1'] = average_valid_vfe.mean()
                stats['+5 STD VFE @ 1'] = np.std(SIMP5_VFEs[simp5_passed])
                stats['+5 Median VFE @ 1'] = np.median(SIMP5_VFEs[simp5_passed])

                average_valid_ce = np.where(simp10_passed, SIMP10_CEs, 0.0).sum(axis=0) / (simp10_passed.sum(axis=0))
                stats['+10 Avg CE @ 1'] = average_valid_ce.mean()
                stats['+10 STD CE @ 1'] = np.std(SIMP10_CEs[simp10_passed])
                stats['+10 Median CE @ 1'] = np.median(SIMP10_CEs[simp10_passed])
                average_valid_vfe = np.where(simp10_passed, SIMP10_VFEs, 0.0).sum(axis=0) / (simp10_passed.sum(axis=0))
                stats['+10 Avg VFE @ 1'] = average_valid_vfe.mean()
                stats['+10 STD VFE @ 1'] = np.std(SIMP10_VFEs[simp10_passed])
                stats['+10 Median VFE @ 1'] = np.median(SIMP10_VFEs[simp10_passed])

        else:
            # all_possible_selections = itertools.combinations(range(num_samples), i+1)
            # 10 random selections
            all_possible_selections = [list(np.random.choice(range(num_samples), i+1, replace=False)) for _ in range(10)]
            pass_rates = []
            best_of_n_CE = []
            best_of_n_median_CE = []
            best_of_n_std_CE = []
            best_of_n_VFE = []
            best_of_n_median_VFE = []
            best_of_n_std_VFE = []
            counter = 0
            for selection in all_possible_selections:
                counter += 1
                if counter > 10:
                    break
                combined_passed = np.any(passed[:, list(selection)], axis=1)
                pass_rate = combined_passed.mean()
                pass_rates.append(pass_rate)

                selected_CE = CEs[:, list(selection)]
                best_CE = np.min(selected_CE, axis=1)
                average_valid_ce = np.where(best_CE < 100, best_CE, 0.0).sum() / np.sum(best_CE < 100)
                best_of_n_CE.append(average_valid_ce)
                best_of_n_median_CE.append(np.median(best_CE[best_CE < 100]))
                best_of_n_std_CE.append(np.std(best_CE[best_CE < 100]))
                
                selected_VFE = VFEs[:, list(selection)]
                best_VFE = selected_VFE[np.arange(selected_VFE.shape[0]), np.argmin(selected_CE, axis=1)]
                average_valid_vfe = np.where(best_CE < 100, best_VFE, 0.0).sum() / np.sum(best_CE < 100)
                best_of_n_VFE.append(average_valid_vfe)
                best_of_n_median_VFE.append(np.median(best_VFE[best_CE < 100]))
                best_of_n_std_VFE.append(np.std(best_VFE[best_CE < 100]))
                
            average_pass_rate = np.mean(pass_rates)
            stats[f'Pass @ {i+1}'] = average_pass_rate
            stats[f'Avg CE Best of {i+1}'] = np.mean(best_of_n_CE)
            stats[f'STD CE Best of {i+1}'] = np.mean(best_of_n_std_CE)
            stats[f'Median CE Best of {i+1}'] = np.mean(best_of_n_median_CE)
            stats[f'Avg VFE Best CE of {i+1}'] = np.mean(best_of_n_VFE)
            stats[f'STD VFE Best CE of {i+1}'] =  np.mean(best_of_n_std_VFE)
            stats[f'Median VFE Best CE of {i+1}'] = np.mean(best_of_n_median_VFE)

            if args.few_steps:
                # +5 steps
                pass_rates = []
                best_of_n_CE = []
                best_of_n_median_CE = []
                best_of_n_std_CE = []
                best_of_n_VFE = []
                best_of_n_median_VFE = []
                best_of_n_std_VFE = []
                counter = 0
                for selection in all_possible_selections:
                    counter += 1
                    if counter > 10:
                        break
                    combined_passed = np.any(simp5_passed[:, list(selection)], axis=1)
                    pass_rate = combined_passed.mean()
                    pass_rates.append(pass_rate)
                    
                    selected_CE = SIMP5_CEs[:, list(selection)]
                    best_CE = np.min(selected_CE, axis=1)
                    average_valid_ce = np.where(best_CE < 100, best_CE, 0.0).sum() / np.sum(best_CE < 100)
                    best_of_n_CE.append(average_valid_ce)
                    best_of_n_median_CE.append(np.median(best_CE[best_CE < 100]))
                    best_of_n_std_CE.append(np.std(best_CE[best_CE < 100]))

                    selected_VFE = SIMP5_VFEs[:, list(selection)]
                    best_VFE = selected_VFE[np.arange(selected_VFE.shape[0]), np.argmin(selected_CE, axis=1)]
                    average_valid_vfe = np.where(best_CE < 100, best_VFE, 0.0).sum() / np.sum(best_CE < 100)
                    best_of_n_VFE.append(average_valid_vfe)
                    best_of_n_median_VFE.append(np.median(best_VFE[best_CE < 100]))
                    best_of_n_std_VFE.append(np.std(best_VFE[best_CE < 100]))
                    
                average_pass_rate = np.mean(pass_rates)
                stats[f'+5 Pass @ {i+1}'] = average_pass_rate
                stats[f'+5 Avg CE Best of {i+1}'] = np.mean(best_of_n_CE)
                stats[f'+5 STD CE Best of {i+1}'] = np.mean(best_of_n_std_CE)
                stats[f'+5 Median CE Best of {i+1}'] = np.mean(best_of_n_median_CE)
                stats[f'+5 Avg VFE Best CE of {i+1}'] = np.mean(best_of_n_VFE)
                stats[f'+5 STD VFE Best CE of {i+1}'] = np.mean(best_of_n_std_VFE)
                stats[f'+5 Median VFE Best CE of {i+1}'] = np.mean(best_of_n_median_VFE)
                
                # +10 steps
                pass_rates = []
                best_of_n_CE = []
                best_of_n_median_CE = []
                best_of_n_std_CE = []
                best_of_n_VFE = []
                best_of_n_median_VFE = []
                best_of_n_std_VFE = []
                counter = 0
                for selection in all_possible_selections:
                    counter += 1
                    if counter > 10:
                        break
                    combined_passed = np.any(simp10_passed[:, list(selection)], axis=1)
                    pass_rate = combined_passed.mean()
                    pass_rates.append(pass_rate)
                    selected_CE = SIMP10_CEs[:, list(selection)]
                    best_CE = np.min(selected_CE, axis=1)
                    average_valid_ce = np.where(best_CE < 100, best_CE, 0.0).sum() / np.sum(best_CE < 100)
                    best_of_n_CE.append(average_valid_ce)
                    best_of_n_median_CE.append(np.median(best_CE[best_CE < 100]))
                    best_of_n_std_CE.append(np.std(best_CE[best_CE < 100]))

                    selected_VFE = SIMP10_VFEs[:, list(selection)]
                    best_VFE = selected_VFE[np.arange(selected_VFE.shape[0]), np.argmin(selected_CE, axis=1)]
                    average_valid_vfe = np.where(best_CE < 100, best_VFE, 0.0).sum() / np.sum(best_CE < 100)
                    best_of_n_VFE.append(average_valid_vfe)
                    best_of_n_median_VFE.append(np.median(best_VFE[best_CE < 100]))
                    best_of_n_std_VFE.append(np.std(best_VFE[best_CE < 100]))

                average_pass_rate = np.mean(pass_rates)
                stats[f'+10 Pass @ {i+1}'] = average_pass_rate
                stats[f'+10 Avg CE Best of {i+1}'] = np.mean(best_of_n_CE)
                stats[f'+10 STD CE Best of {i+1}'] = np.mean(best_of_n_std_CE)
                stats[f'+10 Median CE Best of {i+1}'] = np.mean(best_of_n_median_CE)
                stats[f'+10 Avg VFE Best CE of {i+1}'] = np.mean(best_of_n_VFE)
                stats[f'+10 STD VFE Best CE of {i+1}'] = np.mean(best_of_n_std_VFE)
                stats[f'+10 Median VFE Best CE of {i+1}'] = np.mean(best_of_n_median_VFE)
                
    print("Compliance Test Statistics:")
    # print using table format
    
    # each metric is a column and each experiment number is a row
    rows = []
    keys = ['Pass @ {N}', 'Avg CE @ {N}', 'STD CE @ {N}', 'Median CE @ {N}', 'Avg VFE @ {N}', 'STD VFE @ {N}', 'Median VFE @ {N}']
    keys_n = ['Pass @ {N}', 'Avg CE Best of {N}', 'STD CE Best of {N}', 'Median CE Best of {N}', 'Avg VFE Best CE of {N}', 'STD VFE Best CE of {N}', 'Median VFE Best CE of {N}']
    few_keys = ['+5 Pass @ {N}', '+5 Avg CE @ {N}', '+5 STD CE @ {N}', '+5 Median CE @ {N}', '+5 Avg VFE @ {N}', '+5 STD VFE @ {N}', '+5 Median VFE @ {N}', '+10 Pass @ {N}', '+10 Avg CE @ {N}', '+10 STD CE @ {N}', '+10 Median CE @ {N}', '+10 Avg VFE @ {N}', '+10 STD VFE @ {N}', '+10 Median VFE @ {N}']
    few_keys_n = ['+5 Pass @ {N}', '+5 Avg CE Best of {N}', '+5 STD CE Best of {N}', '+5 Median CE Best of {N}', '+5 Avg VFE Best CE of {N}', '+5 STD VFE Best CE of {N}', '+5 Median VFE Best CE of {N}', '+10 Pass @ {N}', '+10 Avg CE Best of {N}', '+10 STD CE Best of {N}', '+10 Median CE Best of {N}', '+10 Avg VFE Best CE of {N}', '+10 STD VFE Best CE of {N}', '+10 Median VFE Best CE of {N}']
    
    for i in range(num_samples):
        if i == 0:
            rows.append([f"N = 1"] + [stats[key.format(N=1)] for key in keys])
            if args.few_steps:
                rows.append([f"N = 1 + 5"] + [stats[key.format(N=1)] for key in few_keys[:7]])
                rows.append([f"N = 1 + 10"] + [stats[key.format(N=1)] for key in few_keys[7:]])
        else:
            rows.append([f"N = {i+1}"] + [stats[key.format(N=i+1)] for key in keys_n])
            if args.few_steps:
                rows.append([f"N = {i+1} + 5"] + [stats[key.format(N=i+1)] for key in few_keys_n[:7]])
                rows.append([f"N = {i+1} + 10"] + [stats[key.format(N=i+1)] for key in few_keys_n[7:]])

    table = tabulate.tabulate(rows, headers=["N"] + ['Pass Rate', 'Avg CE', 'STD CE', 'Median CE', 'Avg VFE', 'STD VFE', 'Median VFE'], tablefmt="grid", floatfmt=".4f")
    # table = tabulate.tabulate(stats.items(), headers=["Metric", "Value"], tablefmt="grid", floatfmt=".4f")
    print(table)
    
    results_dict.update({
        "stats": stats
    })
    
    pickle.dump(results_dict, open(args.output_path, "wb"))
    print(f"Saved compliance results to {args.output_path}")
    
if __name__ == "__main__":
    main()