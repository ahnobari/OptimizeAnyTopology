# Optimize Any Topology
This repository is the official implemantation of the Optimize Any Topology model for minimum compliance topology optimization.

![image](https://github.com/user-attachments/assets/6200fa2c-0cd5-49af-897c-67688f28c446)


**NEW: Accepted to Neurips 2025**

## Environment
To run this code create a new python environment and run:

```bash
pip install -r requirements.txt
```

> **NOTE**: To run the optimizer and FEA as efficiently as possible on Intel hardware please create an environment with MKL complied packages. To make this easy for linux users we provide pre-compiled wheels for CHOLMOD and scikit-sparse. 

### MKL Environmnent Setup
If you already have MKL compiled SuitSparse and CHOLMOD all you need to do is to run `bash mkl_setup.sh`. If you wish to use the pre-compiled wheels instead run `bash mkl_wheels_setup.sh`. Note that this is just a matter of running the FEA part as fast as possible on Intel hardware **you do not need this** and just using `requirements.txt` will allow you to run the code.

## The Open-TO Dataset
The openTO dataset is publicly available on HuggingFace 🤗 at [OpenTO](https://huggingface.co/datasets/OpenTO/OpenTO). When you run the code we use the datasets module to automatically download this data. Feel free to grab the data and all the splits and train and test you own code or our checkpoints.

## Training
The model is trained in two stages, first a neural field auto-encoder (NFAE) which maps variable resolution and shapes into a common latent space and a latent diffusion model (LDM) which is trained to generate samples using a conditional diffusion process.

### Pretrained Checkpoints
You do not need to run these scripts to replicate results, since we provide trained checkpoints on huggingface. The checkpoints for both the NFAE and LDM can be found at and loaded from:


To load checkpoints using these pretrained weights you can use the `.from_pretrained` function on both the `NFAE` class and `CTOPUNET` class in the `OAT.Models` module.

### Training The Autoencoder
To train the autoencoder run `trainAE.py` in the scripts folder. To see options run help. This script is run using accelerate. In our work we train the model using 8 gpus with a per-gpu batchsize of 15 (120 effective batch size) with dynamic compilation at full precision. To run training on the full pretraining dataset run:

```
accelerate launch --config-file ./accelerate_configs/AE.yml -m  scripts.trainAE --full_sampling --cosine_scheduler --batch_size 15 --num_epochs 30 --extra_decoder_layers --encoder_res 256 --checkpoint_dir checkpointsAE --final_checkpoint NFAE
```

This script will save a hf compatible checkpoint at the `--final_checkpoint` directory

### Training Diffusion Model
Traininf the latent diffusion model (LDM) is done in two steps:

#### Precomputing Latents
To train the latent diffusion model you will first need to pre-compute the latent tensors using the auto-encoder. To do this we provice the `ComputeLatents.py` script. We run this script using accelerate as well to distribute computation of latents across multiple GPUs. To run this script you can run:

```
accelerate launch --config-file ./accelerate_configs/AE.yml -m scripts.ComputeLatents --batch_size 256 --model NFAE --output_path Latents.pth
```

This will save the latents in `Latents.pth`, which we pass to the LDM trainer to use. 
NOTE: the model you pass must be an hf compatible checkpoint including the model config. You can use the NFAE checkpoint from the last script or pass the pretrained AE we provide on hugging face.

#### Training The LDM
Once the latents are obtained we can start training the LDM model. To do this we provide the `trainLDM.py` script which you can run using accelerate as well. For LDM training we use 4 GPUs and a batch size of 32 per gpu (128 effective batch size). To train the   LDM you can run:

```
accelerate launch --config-file ./accelerate_configs/LDM.yml -m scripts.trainLDM --batch_size 32 --cosine_scheduler --latents_path Latents.pth --final_checkpoint LDM
```

### Pre-Trained Checkpoints
Our checkpoint is availble on HF. This checkpoints are available at the following repos on HF:

## Inference and FEA
In this code base we also provide the FEA solver and TO Optimizer we use to generate the data. This solver is a fork of the pyEDGE repository which we use here.