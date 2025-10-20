import os
import torch
from datasets import load_dataset
from OAT.Models import NFAE
from OAT.DataUtils.Collators import NFAECollator
from OAT.DataUtils.Datasets import OpenTO
from OAT.Trainer import Trainer
import argparse
import numpy as np
torch.set_float32_matmul_precision('high')

args = argparse.ArgumentParser()

# dataset parameters
args.add_argument("--splits", type=str, default="pretraining", help="+ separated dataset splits to use, default 'pretraining'")
args.add_argument("--patch_size", type=int, default=64, help="patch size. default 64")
args.add_argument("--full_sampling", action='store_true', help="use full sampling for collator. default False")

# Model Parameters
args.add_argument("--encoder_res", type=int, default=256, help="encoder resolution. default 256")
args.add_argument("--x", type=int, default=1, help="number of channels in the latent space. default 1")
args.add_argument("--hidden_channels", type=int, default=128, help="number of output channels in the decoder. default 128")
args.add_argument("--recon_loss", type=str, default='l1', help="reconstruction loss type. default 'l1'")
args.add_argument("--out_act", type=str, default='sigmoid', help="output activation function. default 'sigmoid'")
args.add_argument("--n_embed", type=int, default=8192*3, help="number of embeddings for vector quantization. default 8192*3")
args.add_argument("--vq_beta", type=float, default=0.25, help="beta parameter for vector quantization. default 0.25")
args.add_argument("--use_vq", action='store_true', help="use vector quantization. default False")
args.add_argument('--use_kl', action='store_true', help="use KL divergence loss. default False")
args.add_argument('--kl_weight', type=float, default=1e-6, help="weight for KL divergence loss. default 1e-6")
args.add_argument("--z_channels", type=int, default=1, help="number of channels in the latent space. default 1")
args.add_argument("--extra_decoder_layers", action='store_true', help="use extra layers in the decoder. default False")

# training parameters
args.add_argument("--num_epochs", type=int, default=50, help="number of epochs to train, default 50")
args.add_argument("--batch_size", type=int, default=32, help="batch size. default 32")
args.add_argument("--lr", type=float, default=1e-4, help="learning rate. default 1e-4")
args.add_argument("--cosine_scheduler", action="store_true", help="use cosine scheduler for learning rate. default False")
args.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate. default 1e-6")
args.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for optimizer. default 0.0")
args.add_argument("--warmup_steps", type=int, default=100, help="number of warmup steps for learning rate scheduler. default 100")
args.add_argument("--checkpoint_dir", type=str, default="checkpointsAE", help="Directory to save checkpoints. Default: checkpointsAE")
args.add_argument("--resume", action="store_true", help="Whether to resume training from a checkpoint. Default: False")
args.add_argument("--resume_path", type=str, default='checkpointsAE/last', help="Path to resume training from a checkpoint. Default: 'checkpointsAE/last'")
args.add_argument("--optimizer_type", type=str, default="AdamW", help="Type of optimizer to use. Default: AdamW")
args.add_argument("--final_checkpoint", type=str, default="NFAE", help="Filename for the final model checkpoint. Default: NFAE")
args = args.parse_args()

def main():
    # Load dataset
    data = load_dataset("OpenTO/OpenTO", split=args.splits)
    dataset = OpenTO(data, encoder_res=args.encoder_res, patch_size=args.patch_size, full_sampling=args.full_sampling, AE_minimal=True)
    collator = NFAECollator(full_sampling=args.full_sampling, zero_centering=True if args.out_act == 'tanh' else False)
    
    # Initialize model
    model = NFAE(
        resolution=args.encoder_res,
        z_channels=args.z_channels,
        hidden_channels=args.hidden_channels,
        recon_loss=args.recon_loss,
        out_act=args.out_act,
        n_embed=args.n_embed,
        vq_beta=args.vq_beta,
        use_vq=args.use_vq,
        use_kl=args.use_kl,
        kl_weight=args.kl_weight,
        extra_decoder_layers=args.extra_decoder_layers
    )
    
    # manually flag only constant sized modules for compilation (pytorch allocates too much memory otherwise)
    # IMPORTANT NOTE: This will break the code if FSDP is used!
    # TODO: Figure out a way to make this work with FSDP
    model.encoder.compile()
    model.decoder.compile()
    
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
        optimizer_type=args.optimizer_type
    )
    
    trainer.train(
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        verbose=True,
        collator=collator,
        checkpoint_dir=args.checkpoint_dir,
    )

    model.save_pretrained(args.final_checkpoint)

if __name__ == "__main__":
    main()
