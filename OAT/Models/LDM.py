from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput, TimestepEmbedding
from ..nn import ProblemEncoder
import torch
from typing import Optional, Tuple, Union
from diffusers.configuration_utils import register_to_config
from diffusers.models import ModelMixin
from huggingface_hub import PyTorchModelHubMixin

class CTOPUNet(UNet2DModel, PyTorchModelHubMixin, ModelMixin):
    @register_to_config
    def __init__(self,
                 BCs = [4,4],
                 BC_n_layers = [4,4],
                 BC_hidden_size = [256,256], 
                 BC_emb_size = [64,64], 
                 Cs = [1,2],
                 C_n_layers = [4,4],
                 C_hidden_size = [256,256],
                 C_mapping_size = [128,128],
                 latent_size = 256,
                 latentShift: float = 0.,
                 latentScale: float = 1.,
                 sample_size: Optional[Union[int, Tuple[int, int]]] = None,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 center_input_sample: bool = False,
                 time_embedding_type: str = "positional",
                 time_embedding_dim: Optional[int] = None,
                 freq_shift: int = 0,
                 flip_sin_to_cos: bool = True,
                 down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
                 mid_block_type: Optional[str] = "UNetMidBlock2D",
                 up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                 block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
                 layers_per_block: int = 2,
                 mid_block_scale_factor: float = 1,
                 downsample_padding: int = 1,
                 downsample_type: str = "conv",
                 upsample_type: str = "conv",
                 dropout: float = 0.0,
                 act_fn: str = "silu",
                 attention_head_dim: Optional[int] = 8,
                 norm_num_groups: int = 32,
                 attn_norm_num_groups: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 resnet_time_scale_shift: str = "default",
                 add_attention: bool = True,
                 class_embed_type: Optional[str] = None,
                 num_class_embeds: Optional[int] = None,
                 num_train_timesteps: Optional[int] = None,
                 ):
        super(CTOPUNet, self).__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            dropout=dropout,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            attn_norm_num_groups=attn_norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            num_train_timesteps=num_train_timesteps
        )
        
        self.problem_encoder = ProblemEncoder(
            BCs = BCs,
            BC_n_layers = BC_n_layers,
            BC_hidden_size = BC_hidden_size, 
            BC_emb_size = BC_emb_size, 
            Cs = Cs,
            C_n_layers = C_n_layers,
            C_hidden_size = C_hidden_size,
            C_mapping_size = C_mapping_size,
            latent_size = latent_size
        )
        
        time_embed_dim = self.time_embedding.linear_1.out_features
        self.problem_embedding = TimestepEmbedding(latent_size, time_embed_dim)
        
        self.latent_size = latent_size
        if isinstance(latentShift, float):
            self.latent_shift = torch.nn.Parameter(torch.tensor([latentShift]), requires_grad=False)
        elif isinstance(latentShift, tuple) or isinstance(latentShift, list):
            self.latent_shift = torch.nn.Parameter(torch.zeros(latentShift).unsqueeze(0), requires_grad=False)
        else:
            self.latent_shift = torch.nn.Parameter(latentShift.unsqueeze(0), requires_grad=False)
        
        if isinstance(latentScale, float):
            self.latent_scale = torch.nn.Parameter(torch.tensor([latentScale]), requires_grad=False)
        elif isinstance(latentScale, tuple) or isinstance(latentScale, list):
            self.latent_scale = torch.nn.Parameter(torch.ones(latentScale).unsqueeze(0), requires_grad=False)
        else:
            self.latent_scale = torch.nn.Parameter(latentScale.unsqueeze(0), requires_grad=False)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        unconditioned: Optional[bool] = True,
        **kwargs
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 1.1 Problem Encoder
        if not unconditioned:
            pe = self.problem_encoder(**kwargs)
            pe = self.problem_embedding(pe)
            emb = emb + pe
        
        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)
    
    def map_to_unnormalized_latent(self, latents):
        return latents * self.latent_scale - self.latent_shift