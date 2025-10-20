import torch
import numpy as np
import accelerate
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
from OAT.Models import NFAE
from OAT.DataUtils.Collators import NFAECollator
from OAT.DataUtils.Datasets import OpenTO
from datasets import load_dataset
torch.set_float32_matmul_precision('high')

args = argparse.ArgumentParser()
args.add_argument("--splits", type=str, default="labeled+NITO", help="+ separated dataset splits to use, default 'labeled+NITO'")
args.add_argument("--batch_size", type=int, default=256, help="batch size. default 256")
args.add_argument("--model", type=str, default="NFAE", help="path to trained hf model checkpoint. default 'NFAE'")
args.add_argument("--output_path", type=str, default='Latents.pth', help="Path to save the computed latents. Default: 'Latents.pth'")
args = args.parse_args()

def compute_latents(
              model,
              dataset,
              data_idx=None,
              batch_size=32,
              collator=None,
              verbose=True):
    
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        
        model = accelerator.prepare_model(model)
        model.eval()
        
        if data_idx is None:
            data_idx = np.array(list(range(len(dataset))))
        
        if accelerator.num_processes > 1:
            data_idx = np.array_split(data_idx, accelerator.num_processes)
            data_idx = data_idx[accelerator.process_index]
        
        collate_fn = collator
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, data_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, collate_fn=collate_fn)
        
        all_latents = []
        
        if verbose and accelerator.is_main_process:
            prog = tqdm(loader, total=len(loader))
        else:
            prog = loader
        
        for i, batch in enumerate(prog):
            if accelerator.num_processes > 1:
                inputs_batch = batch.to(accelerator.device)
            else:
                inputs_batch = batch
            with torch.no_grad():
                if accelerator.num_processes > 1:
                    with model.no_sync():
                        latents = model(inputs_batch, latent_only=True)['latent']
                    gathered_latents = accelerator.gather(latents).chunk(accelerator.num_processes, dim=0)
                    if accelerator.is_main_process:
                        all_latents.append([latent.detach().cpu() for latent in gathered_latents])
                else:
                    latents = model(inputs_batch, latent_only=True)['latent']
                    all_latents.append(latents.detach().cpu())

        if accelerator.num_processes > 1 and accelerator.is_main_process:
            per_device_latents = [torch.cat([latent[i] for latent in all_latents], dim=0) for i in range(accelerator.num_processes)]
            all_latents = torch.cat(per_device_latents, dim=0)
            print(f"Computed latents shape: {all_latents.shape}")
        elif accelerator.num_processes > 1:
            all_latents = None
        else:
            all_latents = torch.cat(all_latents, dim=0)
        return all_latents, accelerator.is_main_process
    
def main():
    model = NFAE.from_pretrained(args.model)
    model.encoder.compile()
    
    data = load_dataset("OpenTO/OpenTO", split=args.splits)
    dataset = OpenTO(data, encoder_res=model.encoder.resolution, patch_size=4, full_sampling=False)
    collator = NFAECollator(full_sampling=False, zero_centering=True if model.renderer.out_act == 'tanh' else False)
    
    latents, is_main = compute_latents(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        collator=collator,
        verbose=True
    )
    
    if is_main:
        torch.save(latents, args.output_path)
        print(f"Latents saved to {args.output_path}")
    
if __name__ == "__main__":
    main()