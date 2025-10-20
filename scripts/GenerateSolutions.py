import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
from typing import Optional
from OAT.Models import NFAE, CTOPUNet
from OAT.Pipelines import OATPipeline, DDIMPipeline
from OAT.DataUtils import OpenTO, NFAECollator, DiffusionCollator
from datasets import load_dataset
import os
import pickle
torch.set_float32_matmul_precision('high')


args = argparse.ArgumentParser()
args.add_argument("--splits", type=str, default="test+NITO_test", help="+ separated dataset splits to use, default 'test+NITO_test'")
args.add_argument("--batch_size", type=int, default=4, help="batch size. default 4")
args.add_argument("--ae_model", type=str, default="OpenTO/NFAE", help="path to trained hf autoencoder model checkpoint. default 'OpenTO/NFAE'")
args.add_argument("--ldm_model", type=str, default="OpenTO/LDM", help="path to trained hf latent diffusion model checkpoint. default 'OpenTO/LDM'")
args.add_argument("--n_samples", type=int, default=64, help="number of samples to generate per example. default 64")
args.add_argument("--num_sampling_steps", type=int, default=20, help="number of sampling steps. default 20")
args.add_argument("--guidance_scale", type=float, default=1.0, help="classifier-free guidance scale. default 1.0")
args.add_argument("--ddim", action="store_true", help="use DDIM sampling instead of ancestral sampling")
args.add_argument("--output_path", type=str, default='GeneratedSamples.pkl', help="Path to save the generated samples. Default: 'GeneratedSamples.pkl'")
args = args.parse_args()

def generate_samples(
              ae_model: NFAE,
              ldm_model: CTOPUNet,
              dataset: OpenTO,
              pipeline: OATPipeline,
              collator: DiffusionCollator,
              accelerator: Optional[Accelerator],
              data_idx: Optional[np.ndarray] = None,
              n_samples: Optional[int] = 64,
              batch_size: Optional[int] = 4,
              num_sampling_steps: Optional[int] = 20,
              guidance_scale: Optional[float] = 1.0,
              ddpm: Optional[bool] = True,
              verbose: Optional[bool] = True
              ):
    
    ae_model = accelerator.prepare_model(ae_model)
    ldm_model = accelerator.prepare_model(ldm_model)
    ae_model.eval()
    ldm_model.eval()
    
    if data_idx is None:
        data_idx = np.array(list(range(len(dataset))))
    
    if accelerator.num_processes > 1:
        data_idx = np.array_split(data_idx, accelerator.num_processes)
        data_idx = data_idx[accelerator.process_index]
    
    
    collate_fn = collator
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, data_idx.tolist()), batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    
    all_generated = []
    
    if verbose and accelerator.is_main_process:
        prog = tqdm(loader, total=len(loader))
    else:
        prog = loader
        
    for i, tupe_batch in enumerate(prog):
        conditions, ae_batch = tupe_batch
        conditions = conditions.to(accelerator.device)
        ae_batch = ae_batch.to(accelerator.device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            gen = pipeline.inference(
                neural_field_inputs=ae_batch,
                conditions=conditions,
                verbose=False,
                n_samples=n_samples,
                num_sampling_steps=num_sampling_steps,
                classifier_free_guidance=guidance_scale,
                ddpm=ddpm,
                clamp_latents=False,
                remap_latents=False
            )
        all_generated.extend(gen)
            
    return all_generated

def main():
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    
    data = load_dataset("OpenTO/OpenTO", split=args.splits)
    dataset = OpenTO(data, full_sampling=True)
    
    collator = DiffusionCollator(
        unconditional_prob=0.0,
        inference=True,
        inference_collator=NFAECollator(zero_centering=False, full_sampling=True, coords_only=True)
    )
    AE = NFAE.from_pretrained(args.ae_model).to(torch.bfloat16)
    LDM = CTOPUNet.from_pretrained(args.ldm_model).to(torch.bfloat16)
    
    AE.decoder.compile()
    for name, module in LDM.named_modules():
        if "problem_encoder" not in name and len(list(module.children())) == 0:
            module.compile()
            
    DDIM = DDIMPipeline()
    OAT = OATPipeline(
        diffusion_model=LDM,
        nfae=AE,
        DDIM=DDIM
    )
    
    generated_samples = generate_samples(
        ae_model=AE,
        ldm_model=LDM,
        dataset=dataset,
        pipeline=OAT,
        collator=collator,
        accelerator=accelerator,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_sampling_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale,
        ddpm= not args.ddim,
        verbose= accelerator.is_main_process
    )
    
    if accelerator.num_processes > 1:
        if not accelerator.is_main_process:
            temporary_file_name = args.output_path.replace('.pkl', f'_tmp_{accelerator.process_index}.pkl')
            with open(temporary_file_name, 'wb') as f:
                pickle.dump(generated_samples, f)
                
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            all_samples = generated_samples
            for i in range(1, accelerator.num_processes):
                temporary_file_name = args.output_path.replace('.pkl', f'_tmp_{i}.pkl')
                with open(temporary_file_name, 'rb') as f:
                    part_samples = pickle.load(f)
                os.remove(temporary_file_name)
                all_samples.extend(part_samples)
            with open(args.output_path, 'wb') as f:
                pickle.dump(all_samples, f)
            print(f"Saved {len(all_samples)} x {args.n_samples} generated samples to {args.output_path}")
    
    else:
        with open(args.output_path, 'wb') as f:
            pickle.dump(generated_samples, f)
        print(f"Saved {len(generated_samples)} x {args.n_samples} generated samples to {args.output_path}")

if __name__ == "__main__":
    main()
