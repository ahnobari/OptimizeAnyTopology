from diffusers import DDIMScheduler, DDPMScheduler
import torch
from typing import Optional, Callable, Union
from ..Models import CTOPUNet, NFAE
from tqdm.auto import tqdm
from ..DataUtils._utils import BatchDict

class DDIMPipeline:
    def __init__(self,
                 num_training_steps: Optional[int] = 1000,
                 prediction_type: Optional[str] = 'v_prediction',
                 rescale_betas_zero_snr: Optional[bool] = True,
                 timestep_spacing: Optional[str] = 'trailing',
                 cosine_schedule: Optional[bool] = True,
                 loss_type: Optional[str] = 'l2'):
        
        self.TrainScheduler = DDPMScheduler(
            num_train_timesteps=num_training_steps,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            timestep_spacing=timestep_spacing,
            beta_schedule = 'linear' if not cosine_schedule else 'squaredcos_cap_v2',
            clip_sample = False
        )
        
        self.InferenceScheduler = DDIMScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=self.TrainScheduler.config.beta_start,
            beta_end=self.TrainScheduler.config.beta_end,
            beta_schedule=self.TrainScheduler.config.beta_schedule,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            timestep_spacing=timestep_spacing,
            clip_sample = False
        )
        
        self.prediction_type = prediction_type
        
        self.loss_fn = torch.nn.MSELoss() if loss_type == 'l2' else torch.nn.L1Loss()

    def get_target(self, x: torch.Tensor, noise: torch.Tensor, t: Union[torch.Tensor, int]):
        """
        Get the target for the model based on the prediction type.
        """
        if self.prediction_type == 'epsilon':
            return noise
        elif self.prediction_type == 'v_prediction':
            return self.TrainScheduler.get_velocity(x, noise, t)
        elif self.prediction_type == 'sample':
            return x
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    def compute_loss(self, model: CTOPUNet, sample: torch.Tensor, noise: Optional[torch.Tensor] = None, **kwargs):
        
        bs = sample.shape[0]
        timesteps = torch.randint(
                    0, self.TrainScheduler.config.num_train_timesteps, (bs,), device=model.device,
                    dtype=torch.long
                )
        noise = torch.randn_like(sample) if noise is None else noise
        
        noisy_sample = self.TrainScheduler.add_noise(sample, noise, timesteps)
        
        pred = model(noisy_sample, timesteps, **kwargs)[0]
        
        target = self.get_target(sample, noise, timesteps)
        
        loss = self.loss_fn(pred, target)
        
        return {'loss': loss }
    
    @torch.no_grad()
    def inference(self, 
                  model: CTOPUNet,
                  num_sampling_steps: Optional[int] = 50,
                  noise: Optional[torch.Tensor] = None,
                  batch_size: Optional[int] = 1,
                  guidance_function: Optional[Callable] = None,
                  guidance_scale: Optional[float] = 1.0,
                  static_guidance: Optional[bool] = True,
                  direct_guidance: Optional[bool] = True,
                  final_callable: Optional[Callable] = None,
                  history: Optional[bool] = False,
                  verbose: Optional[bool] = False,
                  conditioning_over_relaxation_factor: Optional[float] = None,
                  ddpm : Optional[bool] = False,
                  low_rank: Optional[bool] = False,
                  tau: Optional[float] = 0.99,
                  **kwargs):
        
        """
        Perform inference using the DDIM scheduler.
        """
        if noise is None:
            noise = torch.randn((batch_size, model.conv_in.in_channels, model.sample_size, model.sample_size), device=model.device)
        
        batch_size = noise.shape[0]
        
        sched = self.InferenceScheduler if not ddpm else self.TrainScheduler
        
        # Prepare the scheduler
        sched.set_timesteps(num_sampling_steps)
        
        if history:
            hist = []
        else: 
            hist = None
        
        if verbose:
            prog = tqdm(sched.timesteps)
        else:
            prog = sched.timesteps
            
        st = 0
        grads = None
        for t in prog:
            pred_noise = model(noise, t, **kwargs).sample
            
            if conditioning_over_relaxation_factor is not None and conditioning_over_relaxation_factor != 1.0:
                kwargs['unconditioned'] = True
                pred_noise_uncond = model(noise, t, **kwargs).sample
                kwargs['unconditioned'] = False
                cond_delta = pred_noise - pred_noise_uncond
                pred_noise = pred_noise_uncond + conditioning_over_relaxation_factor * cond_delta
            
            denoised = sched.step(pred_noise, t, noise)
            pred = denoised.pred_original_sample
            
            if guidance_function is not None:
                grads = guidance_function(denoised, t, history = hist, model = model, final_callable = final_callable, step=st, total_steps=num_sampling_steps, previous_grad = grads, **kwargs)

                if low_rank:
                    Z_t = denoised.prev_sample.clone().squeeze()
                    # svd
                    U, S, V = torch.svd(Z_t)
                    
                    eig = S.square().flatten()
                    c = torch.cumsum(eig, dim=0)/torch.sum(eig)
                    r = torch.min(torch.where(c>=tau)[0])
                    
                    U_r = U[:, :r]
                    V_r = V[:r, :]
                    
                    G = grads.squeeze()
                    G_ = U_r.T @ G @ V_r.T
                    grad_r = U_r @ G_ @ V_r
                    grad_r = grad_r.reshape(denoised.pred_original_sample.shape)
                else:
                    grad_r = grads
                    
                alpha_bar = sched.alphas_cumprod[t]        # scalar tensor
                beta_bar  = 1.0 - alpha_bar
                
                if static_guidance or t == self.TrainScheduler.config.num_train_timesteps-1:
                    mult = guidance_scale
                else:
                    mult = guidance_scale * (beta_bar / alpha_bar).sqrt()
                
                if direct_guidance:
                    noise = denoised.prev_sample
                    noise = noise - mult * grad_r
                    # clip to [-1, 1]
                    noise = torch.clamp(noise, -1, 1)
                    
                else:
                    if t == self.TrainScheduler.config.num_train_timesteps-1:
                        grad_eps  = -grad_r
                    else:
                        grad_eps  = -grad_r * (beta_bar / alpha_bar).sqrt()   # dε = −√β/α ∇x0
                    noise_pred = pred_noise + guidance_scale * grad_eps
                    noise = sched.step(noise_pred, t, noise).prev_sample
            else:
                noise = denoised.prev_sample
                
            if history:
                if final_callable is not None:
                    hist.append({'x_0': (pred + 1) / 2 * model.latent_scale - model.latent_shift,
                                'x_t': (noise + 1) / 2 * model.latent_scale - model.latent_shift,
                                'X_0': final_callable((pred + 1) / 2 * model.latent_scale - model.latent_shift),
                                'X_t': final_callable((noise + 1) / 2 * model.latent_scale - model.latent_shift)})
                else:
                    hist.append({'x_0': (pred + 1) / 2 * model.latent_scale - model.latent_shift,
                                'x_t': (noise + 1) / 2 * model.latent_scale - model.latent_shift})

            st += 1
        noise = (noise + 1) / 2 * model.latent_scale - model.latent_shift
        
        if final_callable is not None:
            noise = final_callable(noise)
        
        noise = noise.detach().cpu().numpy()
        
        if history:
            return noise, hist
    
        return noise

class OATPipeline:
    def __init__(self, 
                 DDIM: DDIMPipeline,
                 diffusion_model: CTOPUNet,
                 nfae: NFAE):
        self.DDIM = DDIM
        self.diffusion_model = diffusion_model
        self.nfae = nfae
    
    @staticmethod
    def expand_batch_to_multi_sample(n_samples, neural_field_inputs, batch_size, conditions=None):
        
        nfae_batch = BatchDict({
            'gt_coord' : neural_field_inputs['gt_coord'] * n_samples,
            'gt_cell' : neural_field_inputs['gt_cell'] * n_samples
        })
        
        if conditions is not None:
            Cs = [c.repeat(n_samples, 1) for c in conditions['Cs']]
            BCs = [b.repeat(n_samples, 1) for b in conditions['BCs']]
            BC_Batch = [
                (bcs.repeat(n_samples).view(n_samples, -1) +
                 torch.arange(n_samples, device=bcs.device).view(-1, 1) * batch_size).view(-1)
                 for bcs in conditions['BC_Batch']
                ]
            conditions = BatchDict({
                'Cs': Cs,
                'BCs': BCs,
                'BC_Batch': BC_Batch
            })
            
            return nfae_batch, conditions
        else:
            return nfae_batch, None
    
    @torch.no_grad()
    def inference(self,
                  neural_field_inputs: dict,
                  num_sampling_steps: Optional[int] = 50,
                  noise: Optional[torch.Tensor] = None,
                  random_seed: Optional[int] = None,
                  n_samples: Optional[int] = 1,
                  classifier_free_guidance: Optional[float] = 1.0,
                  history: Optional[bool] = False,
                  verbose: Optional[bool] = False,
                  ddpm: Optional[bool] = False,
                  conditions: Optional[dict] = None,
                  clamp_latents: Optional[bool] = False,
                  remap_latents: Optional[bool] = False
                  ):
        
        batch_size = len(neural_field_inputs['gt_coord'])

        if conditions is None and classifier_free_guidance != 1.0:
            raise ValueError("Conditions must be provided for classifier-free guidance.")
        
        if n_samples > 1:
            neural_field_inputs, conditions = self.expand_batch_to_multi_sample(n_samples, neural_field_inputs, batch_size, conditions)
        
        neural_field_inputs = neural_field_inputs.to(self.diffusion_model.device)
        
        if conditions is not None:
            conditions = conditions.to(self.diffusion_model.device)
            conditions['unconditioned'] = False
        else:
            conditions = {'unconditioned': True}
        
        total_count = len(neural_field_inputs['gt_coord'])
        
        if noise is None:
            if random_seed is not None:
                torch.manual_seed(random_seed)
            
            noise = torch.randn((total_count,
                                 self.diffusion_model.conv_in.in_channels,
                                 self.diffusion_model.sample_size,
                                 self.diffusion_model.sample_size),
                                device=self.diffusion_model.device)
        else:
            noise = noise.to(self.diffusion_model.device)
            
        if history:
            hist = []
        else: 
            hist = None

        sched = self.DDIM.InferenceScheduler if not ddpm else self.DDIM.TrainScheduler

        # Prepare the scheduler
        sched.set_timesteps(num_sampling_steps)
        
        if verbose:
            prog = tqdm(sched.timesteps)
        else:
            prog = sched.timesteps
        
        for step in prog:
            pred_noise = self.diffusion_model(noise, step, **conditions).sample
            
            if classifier_free_guidance != 1.0:
                pred_noise_uncond = self.diffusion_model(noise, step, unconditioned=True).sample
                
                delta = pred_noise - pred_noise_uncond
                
                pred_noise = pred_noise_uncond + classifier_free_guidance * delta
                
            denoised = sched.step(pred_noise, step, noise)    
            noise = denoised.prev_sample
            
            if clamp_latents and ddpm:
                noise = torch.clamp(noise, -1, 1)
            
            if history:
                pred = denoised.pred_original_sample
                pred_h = pred.cpu().reshape(n_samples, batch_size, *pred.shape[1:]).swapaxes(0, 1).squeeze(1)
                noise_h = noise.cpu().reshape(n_samples, batch_size, *noise.shape[1:]).swapaxes(0, 1).squeeze(1)
                
                if remap_latents:
                    pred_h = (pred_h + 1) / 2 * self.diffusion_model.latent_scale - self.diffusion_model.latent_shift
                    noise_h = (noise_h + 1) / 2 * self.diffusion_model.latent_scale - self.diffusion_model.latent_shift
                else:
                    pred_h = pred_h * self.diffusion_model.latent_scale - self.diffusion_model.latent_shift
                    noise_h = noise_h * self.diffusion_model.latent_scale - self.diffusion_model.latent_shift
                    
                hist.append({
                    'x_0': pred_h,
                    'x_t': noise_h
                })
        
        if clamp_latents:
            noise = torch.clamp(noise, -1, 1)
        
        if remap_latents:
            noise = (noise + 1) / 2 * self.diffusion_model.latent_scale - self.diffusion_model.latent_shift
        else:
            noise = noise * self.diffusion_model.latent_scale - self.diffusion_model.latent_shift
        
        if self.nfae.use_vq:
            noise, _, _ = self.nfae.quantizer(noise)

        phi = self.nfae.decoder(noise)
        prediction = self.nfae.renderer(phi, neural_field_inputs['gt_coord'], neural_field_inputs['gt_cell'])

        prediction = [
            [prediction[i + j * batch_size].cpu() for j in range(n_samples)] for i in range(batch_size)
        ]
        
        return prediction, hist if history else None