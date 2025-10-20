from ..nn import Encoder, Decoder, ConcatWrapper, VectorQuantizer
import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

class NFAE(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 in_channels=1,
                 resolution=256,
                 z_channels=1,
                 hidden_channels=128,
                 hidden_channels_decoder=128,
                 hidden_channels_renderer=256,
                 recon_loss='l1',
                 out_act='',
                 n_embed=8192*3,
                 vq_beta=0.25,
                 use_vq=False,
                 use_kl=False,
                 extra_decoder_layers=False,
                 kl_weight=1e-4):
        super(NFAE, self).__init__()
        
        self.in_channels = in_channels
        ch_mult_d = (1,2,4) if not extra_decoder_layers else (1,1,2,4)
        z_channels_ = z_channels * 2 if use_kl else z_channels
        self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, resolution=resolution, z_channels=z_channels_)
        self.decoder = Decoder(hidden_channels=hidden_channels_decoder, resolution=resolution, z_channels=z_channels, ch_mult=ch_mult_d)
        self.renderer = ConcatWrapper(z_dec_channels=hidden_channels_decoder, out_channels=in_channels, out_act=out_act, hidden_channels=hidden_channels_renderer)
        
        self.use_vq = use_vq
        self.use_kl = use_kl
        self.kl_weight = kl_weight
        
        if use_vq:
            self.quantizer = VectorQuantizer(n_embed, z_channels, beta=vq_beta, remap=None, sane_index_shape=False)#, legacy=False)

        if recon_loss == 'l1':
            self.recon_loss = nn.L1Loss()
        elif recon_loss == 'l2':
            self.recon_loss = nn.MSELoss()
        elif recon_loss == 'cross_entropy':
            self.recon_loss = nn.BCEWithLogitsLoss()
            if out_act != '':
                print("Warning: out_act should be empty when using cross_entropy loss.")
        else:
            self.recon_loss = recon_loss
    
    def forward(self, input_batch, compute_loss=True, latent_only=False):
        return self.call(input_batch, compute_loss=compute_loss, latent_only=latent_only)
    
    def call(self, input_batch, compute_loss=True, latent_only=False, inference_mode=False):
        z = self.encoder(input_batch['inp'])

        if latent_only:
            if self.use_kl:
                mu, log_var = torch.chunk(z, 2, dim=1)
                z = mu
            return {'latent': z}
        
        if compute_loss:
            loss = 0.0
            loss_dict = dict()
        
        if self.use_kl:
            mu, log_var = torch.chunk(z, 2, dim=1)
            variance = torch.exp(log_var)
            std = torch.exp(0.5 * log_var)

            if compute_loss:
                kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - variance, dim=[1,2,3]))
                loss += self.kl_weight * kl_loss
                loss_dict['kl_loss'] = kl_loss
            
            if not inference_mode:
                z = mu + torch.randn_like(std) * std
            else:
                z = mu

        if self.use_vq:
            z, quant_loss, _ = self.quantizer(z)
            if compute_loss:
                loss += quant_loss
                loss_dict['quant_loss'] = quant_loss
        
        phi = self.decoder(z)
        
        pred_patch = self.renderer(phi,input_batch['gt_coord'],input_batch['gt_cell'])
        
        if compute_loss:
            if isinstance(pred_patch, list):
                recon_loss = self.recon_loss(torch.cat([item.flatten() for item in pred_patch]), torch.cat([item.flatten() for item in input_batch['gt']]))
            else:
                recon_loss = self.recon_loss(pred_patch, input_batch['gt'])

            loss_dict['recon_loss'] = recon_loss
            
            loss += recon_loss
                
            loss_dict['loss'] = loss
            
            output = {
                'pred': pred_patch,
                'latent': z
            }
            
            return output, loss_dict
        else:
            output = {
                'pred': pred_patch,
                'latent': z
            }
                
            return output