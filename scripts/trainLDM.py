import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os
import numpy as np
import torch
from datasets import load_dataset
from OAT.Trainer import Trainer
from OAT.DataUtils.Datasets import OpenTO
from OAT.Models import CTOPUNet
from OAT.Pipelines import DDIMPipeline
from OAT.DataUtils.Collators import DiffusionCollator
torch.set_float32_matmul_precision('high')

args = ArgumentParser()
# dataset parameters
args.add_argument("--splits", type=str, default="labeled+NITO", help="+ separated dataset splits to use, default 'labeled+NITO'")
args.add_argument("--latents_path", type=str, default='Latents.pth', help="Path to the latents file. Default: 'Latents.pth'")

# Model parameters
args.add_argument("--layers_per_block", type=int, default=2, help="number of layers per block in UNet. default 2")
args.add_argument("--block_out_channels", type=str, default="320,640,1280,1280", help="comma separated output channels for each block. default '320,640,1280,1280'")
args.add_argument("--norm_num_groups", type=int, default=32, help="number of groups for group normalization. default 32")
args.add_argument("--attention_head_dim", type=int, default=8, help="dimension of attention heads. default 8")
args.add_argument('--BC_n_layers', type=int, default=4, help='Number of layers in each Boundary Condition. Default: 4')
args.add_argument('--BC_hidden_size', type=int, default=512, help='Hidden size in each Boundary Condition. Default: 512')
args.add_argument('--BC_emb_size', type=int, default=512, help='Embedding size in each Boundary Condition. Default: 512')
args.add_argument('--C_n_layers', type=int, default=2, help='Number of layers in each Constraints. Default: 2')
args.add_argument('--C_hidden_size', type=int, default=512, help='Hidden size in each Constraints. Default: 512')
args.add_argument('--C_mapping_size', type=int, default=256, help='Mapping size in each Constraints. Default: 256')
args.add_argument('--condition_latent_size', type=int, default=1280, help='Latent size for the UNet model. Default: 1280')

# Training parameters
args.add_argument("--num_epochs", type=int, default=50, help="number of epochs to train, default 50")
args.add_argument("--batch_size", type=int, default=32, help="batch size. default 32")
args.add_argument("--lr", type=float, default=1e-4, help="learning rate. default 1e-4")
args.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for optimizer. default 0.0")
args.add_argument("--cosine_scheduler", action="store_true", help="use cosine scheduler")
args.add_argument("--warmup_steps", type=int, default=200, help="number of warmup steps. default 200")
args.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate. default 1e-6")
args.add_argument("--checkpoint_dir", type=str, default="CheckpointsLDM", help="directory to save checkpoints. default CheckpointsLDM")
args.add_argument("--resume", action="store_true", help="resume training from checkpoint")
args.add_argument("--resume_path", type=str, default="CheckpointsLDM/last", help="path to resume checkpoint. default CheckpointsLDM/last will find the latest checkpoint in CheckpointsLDM")
args.add_argument('--optimizer_type', type=str, default='AdamW', help='Optimizer to use. Default: AdamW, Options: Adam, AdamW, SGD, Adam8, Adafactor')
args.add_argument('--unconditional_prob', type=float, default=0.25, help='Unconditional probability. Default: 0.25')
args.add_argument('--BC_dropout_prob', type=float, default=0.1, help='Boundary Condition dropout probability. Default: 0.1')
args.add_argument('--C_dropout_prob', type=float, default=0.1, help='Constraints dropout probability. Default: 0.1')
args.add_argument("--final_checkpoint", type=str, default="LDM", help="path to save the final model. default LDM")

args = args.parse_args()

def main():
    
    if args.latents_path is None:
        raise ValueError("Latents path must be provided.")

    data = load_dataset("OpenTO/OpenTO", split=args.splits)
    dataset = OpenTO(data,
                    torch.load(args.latents_path),
                    BC_dropout_prob=args.BC_dropout_prob,
                    C_dropout_prob=args.C_dropout_prob,
                    diffusion_minimal=True)
    collator = DiffusionCollator(unconditional_prob=args.unconditional_prob)
    
    model = CTOPUNet(
        sample_size       = dataset.latent_tensors.shape[2],          # latent spatial size
        in_channels       = dataset.latent_tensors.shape[1],           # AE latent channels
        out_channels      = dataset.latent_tensors.shape[1],           # AE latent channels
        layers_per_block  = args.layers_per_block,
        block_out_channels= tuple(map(int, args.block_out_channels.split(","))),
        down_block_types= tuple([
            "AttnDownBlock2D" if i >= 2 else "DownBlock2D"
            for i in range(len(args.block_out_channels.split(",")))
        ]),
        up_block_types= tuple([
            "AttnUpBlock2D" if i >= 2 else "UpBlock2D"
            for i in reversed(range(len(args.block_out_channels.split(","))))
        ]),
        norm_num_groups   = args.norm_num_groups,
        mid_block_type    = "UNetMidBlock2DAttn",
        attention_head_dim = args.attention_head_dim,
        BCs = [4,4],
        BC_n_layers = [args.BC_n_layers]*2,
        BC_hidden_size = [args.BC_hidden_size]*2, 
        BC_emb_size = [args.BC_emb_size]*2, 
        Cs = [2,1],
        C_n_layers = [args.C_n_layers]*2,
        C_hidden_size = [args.C_hidden_size]*2,
        C_mapping_size = [args.C_mapping_size]*2,
        latent_size = args.condition_latent_size,
        latentShift=dataset.shift,
        latentScale=dataset.scale
    )
    
    if args.resume:
        if os.path.exists(args.resume_path):
            resume_from = args.resume_path
        elif "/last" in args.resume_path:
            # find the latest checkpoint in the resume directory
            r_dir = args.resume_path.replace("/last", "/")

            if os.path.exists(r_dir):
                checkpoints = [f for f in os.listdir(r_dir) if os.path.isdir(os.path.join(r_dir, f))]
                if len(checkpoints) > 0:
                    dates_modified = [os.path.getmtime(os.path.join(r_dir, f)) for f in checkpoints]
                    latest_checkpoint = checkpoints[np.argmax(dates_modified)]
                    resume_from = os.path.join(r_dir, latest_checkpoint)
                    print(f"Resuming from checkpoint: {latest_checkpoint}")
                else:
                    print("No checkpoint found in the directory")
    else:
        resume_from = None
    
    diffusion = DDIMPipeline()
    max_steps = (len(dataset) + args.batch_size - 1) // args.batch_size * args.num_epochs
    
    trainer = Trainer(
        model = model,
        lr = args.lr,
        weight_decay= args.weight_decay,
        schedule_type= 'cosine_with_warmup' if args.cosine_scheduler else 'constant_with_warmup',
        lr_final= args.final_lr,
        warmup_steps= args.warmup_steps,
        max_steps=max_steps,
        checkpoint_path=resume_from,
        optimizer_type=args.optimizer_type,
        custom_loss=diffusion.compute_loss
    )
    
    trainer.train(
                  dataset=dataset,
                  batch_size=args.batch_size,
                  epochs=args.num_epochs,
                  verbose=True,
                  collator=collator,
                  checkpoint_dir=args.checkpoint_dir,
                  pass_batch_as_dict=False
                )
    
    model.save_pretrained(args.final_checkpoint)

if __name__ == "__main__":
    main()