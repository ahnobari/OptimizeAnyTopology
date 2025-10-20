import torch
from ._utils import *
import numpy as np

class NFAECollator:
    def __init__(self, zero_centering=True, full_sampling=False, coords_only=False):
        self.full_sampling = full_sampling
        self.coords_only = coords_only

        if zero_centering:
            self.mul = 2
            self.add = -1
        else:
            self.mul = 1
            self.add = 0
    
    def __call__(self, batch):
        collated_batch = {}
        
        if self.coords_only:
            if not self.full_sampling:
                collated_batch['gt_coord'] = torch.stack([item['gt_coord'] for item in batch])
                collated_batch['gt_cell'] = torch.stack([item['gt_cell'] for item in batch])
                
            else:
                collated_batch['gt_coord'] = [item['gt_coord'].unsqueeze(0) for item in batch]
                collated_batch['gt_cell'] = [item['gt_cell'].unsqueeze(0) for item in batch]
            
            return BatchDict(collated_batch)
        
        collated_batch['inp'] = torch.stack([item['inp']*self.mul+self.add for item in batch])
        if batch[0]['latent'] is not None:
            collated_batch['latent'] = torch.stack([item['latent'] for item in batch])
        
        if not self.full_sampling:
            collated_batch['gt'] = torch.stack([item['gt']*self.mul+self.add for item in batch])
            collated_batch['gt_coord'] = torch.stack([item['gt_coord'] for item in batch])
            collated_batch['gt_cell'] = torch.stack([item['gt_cell'] for item in batch])
            
        else:
            collated_batch['gt'] = [item['gt'].unsqueeze(0)*self.mul+self.add for item in batch]
            collated_batch['gt_coord'] = [item['gt_coord'].unsqueeze(0) for item in batch]
            collated_batch['gt_cell'] = [item['gt_cell'].unsqueeze(0) for item in batch]
            
        return BatchDict(collated_batch)
    
class DiffusionCollator:
    def __init__(self, unconditional_prob=0.0,
                 inference=False,
                 inference_collator=None):
        self.unconditional_prob = unconditional_prob
        self.inference = inference
        self.inference_collator = inference_collator

    def __call__(self, batch):
            
        Cs = [torch.stack([torch.tensor(b['Cs'][i]) for b in batch]).float() for i in range(len(batch[0]['Cs']))]
        for i in range(len(Cs)):
            if Cs[i].dim() == 1:
                Cs[i] = Cs[i].unsqueeze(1)

        BCs = [torch.cat([torch.tensor(b['BCs'][i].astype(np.float32)).float() for b in batch]) for i in range(len(batch[0]['BCs']))]
        BC_batch = [torch.cat([torch.tensor(np.repeat(j,b['sizes'][i])).long() for j,b in enumerate(batch)]) for i in range(len(batch[0]['BCs']))]
        
        if batch[0]['latent'] is not None:
            latents = torch.stack([b['latent'] for b in batch])
        
            out =  {
                'sample': latents.float(),
                'Cs': Cs,
                'BCs': BCs,
                'BC_Batch': BC_batch,
                'unconditioned': np.random.rand() < self.unconditional_prob
            }
        else:
            out =  {
                'Cs': Cs,
                'BCs': BCs,
                'BC_Batch': BC_batch,
                'unconditioned': np.random.rand() < self.unconditional_prob
            }
        
        if self.inference:
            decoder_batch = self.inference_collator(batch)
            
            return BatchDict(out), decoder_batch
        
        return BatchDict(out)