import random
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from ._utils import *
from torchvision.transforms import functional as F

class OpenTO(Dataset):
    def __init__(self, 
                 data_dicts,
                 latent_tensors=None,
                 encoder_res=256,
                 patch_size=32,
                 do_resize=False,
                 resize_gt_lb=64,
                 resize_gt_ub=1024,
                 full_sampling=False,
                 unconditional_prob=0.0,
                 BC_dropout_prob=0.0,
                 C_dropout_prob=0.0,
                 shift = None,
                 scale = None,
                 normalization_mode='safe_minmax',
                 normalize_per_dim=False,
                 diffusion_minimal=False,
                 AE_minimal=False):
        
        self.data_dicts = data_dicts
        self.latent_tensors = latent_tensors
        self.encoder_res = encoder_res
        self.patch_size = patch_size
        self.do_resize = do_resize
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.full_sampling = full_sampling
        self.unconditional_prob = unconditional_prob
        self.BC_dropout_prob = BC_dropout_prob  
        self.C_dropout_prob = C_dropout_prob
        self.normalization_mode = normalization_mode
        self.normalize_per_dim = normalize_per_dim
        self.diffusion_minimal = diffusion_minimal
        self.AE_minimal = AE_minimal
        
        if latent_tensors is not None:
            if scale is None:
                if normalization_mode == 'minmax':
                    if not self.normalize_per_dim:
                        scale = (latent_tensors.max() - latent_tensors.min())/2
                    else:
                        scale = (latent_tensors.max(0).values - latent_tensors.min(0).values)/2
                elif normalization_mode == 'safe_minmax':
                    scale = float((latent_tensors.view(latent_tensors.shape[0], -1).max(dim=1).values - latent_tensors.view(latent_tensors.shape[0], -1).min(dim=1).values).median()/2)
                elif normalization_mode == 'std':
                    if not self.normalize_per_dim:
                        scale = latent_tensors.std()
                    else:
                        scale = latent_tensors.std(dim=0)
                else:
                    raise ValueError(f"Unknown normalization mode: {normalization_mode}")
                
            if shift is None:
                if normalization_mode == 'minmax':
                    if not self.normalize_per_dim:
                        shift = -latent_tensors.min() - scale
                    else:
                        shift = -latent_tensors.min(0).values - scale
                elif normalization_mode == 'safe_minmax':
                    shift = float(-latent_tensors.view(latent_tensors.shape[0], -1).min(dim=1).values.median() - scale)
                elif normalization_mode == 'std':
                    if not self.normalize_per_dim:
                        shift = -latent_tensors.mean()
                    else:
                        shift = -latent_tensors.mean(dim=0)
                else:
                    raise ValueError(f"Unknown normalization mode: {normalization_mode}")
            
            self.has_latent_tensors = True
        else:
            self.has_latent_tensors = False
        
        self.shift = shift
        self.scale = scale

    def __len__(self):
        return len(self.data_dicts)
    
    def __getitem__(self, idx):
        if not self.diffusion_minimal:
            img = self.data_dicts[idx]['topology']
            img = F.to_tensor(img)
            img, pad_info = center_pad_square(img, fill=0)
            
            orig_size = img.shape[-1]
            
            if orig_size < self.encoder_res:
                inp = F.resize(img, self.encoder_res, interpolation=F.InterpolationMode.NEAREST_EXACT)
            else:
                inp = F.resize(img, self.encoder_res, interpolation=F.InterpolationMode.BICUBIC)
                
            if self.do_resize:
                gt_res = random.randint(self.resize_gt_lb, self.resize_gt_ub)
                gt = F.resize(img, gt_res, interpolation=F.InterpolationMode.BICUBIC)
            else:
                gt_res = orig_size
                gt = img.clone()
            
            if self.full_sampling:
                start_h = pad_info[0]
                start_w = pad_info[2]
                end_h = gt.shape[1] - pad_info[1]
                end_w = gt.shape[2] - pad_info[3]
                
                gt = gt[:, start_h:end_h, start_w:end_w]
                        
                # Calculate relative coordinates for the patch
                rel_start_h = start_h / gt_res
                rel_start_w = start_w / gt_res
                rel_end_h = end_h / gt_res
                rel_end_w = end_w / gt_res
                
                # bring to -1 to 1 range
                rel_start_h = 2 * rel_start_h - 1
                rel_start_w = 2 * rel_start_w - 1
                rel_end_h = 2 * rel_end_h - 1
                rel_end_w = 2 * rel_end_w - 1
                
                # Generate coordinate and cell grids for the patch
                coord, cell = make_coord_cell_grid(
                    (end_h - start_h, end_w - start_w),
                    range=[[rel_start_w, rel_end_w], 
                        [rel_start_h, rel_end_h]]
                )
                
                cell[:] = torch.tensor([2/gt_res, 2/gt_res])
                
            elif gt_res > self.patch_size:
                padded_ratio_h = pad_info[0] / img.shape[-1]
                padded_ratio_w = pad_info[2] / img.shape[-1]
                
                min_start_h = int(gt_res * padded_ratio_h)
                min_start_w = int(gt_res * padded_ratio_w)
                max_start_h = int(gt_res * (1 - padded_ratio_h) - self.patch_size)
                max_start_w = int(gt_res * (1 - padded_ratio_w) - self.patch_size)
                
                if max_start_h < min_start_h:
                    min_start_h = max_start_h = gt_res // 2 - self.patch_size // 2
                if max_start_w < min_start_w:
                    min_start_w = max_start_w = gt_res // 2 - self.patch_size // 2
                
                start_h = random.randint(min_start_h, max_start_h)
                start_w = random.randint(min_start_w, max_start_w)
                gt = gt[:, start_h:start_h + self.patch_size, 
                        start_w:start_w + self.patch_size]
                        
                # Calculate relative coordinates for the patch
                rel_start_h = start_h / gt_res
                rel_start_w = start_w / gt_res
                rel_end_h = (start_h + self.patch_size) / gt_res
                rel_end_w = (start_w + self.patch_size) / gt_res
                
                # bring to -1 to 1 range
                rel_start_h = 2 * rel_start_h - 1
                rel_start_w = 2 * rel_start_w - 1
                rel_end_h = 2 * rel_end_h - 1
                rel_end_w = 2 * rel_end_w - 1
                
                # Generate coordinate and cell grids for the patch
                coord, cell = make_coord_cell_grid(
                    self.patch_size,
                    range=[[rel_start_w, rel_end_w], 
                        [rel_start_h, rel_end_h]]
                )
                
                cell[:] = torch.tensor([2/gt_res, 2/gt_res])
            else:
                coord, cell = make_coord_cell_grid(self.patch_size)
        else:
            inp = None
            gt = None
            coord = None
            cell = None
        
        if not self.AE_minimal:
            if np.random.rand() < self.unconditional_prob:
                Cs = [
                    np.array([-1.0, -1.0]),
                    np.array([-1.0])
                ]
                
                BCs = [
                    np.zeros((1, 4)) - 1.0,
                    np.zeros((1, 4)) - 1.0
                ]
                
                sizes = [1, 1]
                
            else:
                # AR = self.data_dicts[idx]['topology'].size[0] / self.data_dicts[idx]['topology'].size[1]
                b = max(self.data_dicts[idx]['topology'].size[0], self.data_dicts[idx]['topology'].size[1])
                AR = np.array([self.data_dicts[idx]['topology'].size[0] / b, 
                            self.data_dicts[idx]['topology'].size[1] / b])
                vf = self.data_dicts[idx]['volume fraction']
                if vf is None:
                    vf = -1.0
                
                if np.random.rand() < self.C_dropout_prob:
                    AR = np.array([-1.0, -1.0])
                    
                if np.random.rand() < self.C_dropout_prob:
                    vf = -1.0
                
                Cs = [
                    AR,
                    np.array([vf])
                ]
                
                
                BCs = []
                sizes = []
                
                if isinstance(self.data_dicts[idx]['boundary conditions'],list):
                    bc = np.array(self.data_dicts[idx]['boundary conditions'])
                    size = bc.shape[0]
                    if np.random.rand() < self.BC_dropout_prob:
                        BCs.append(np.zeros((1, 4)) - 1.0)
                        sizes.append(1)
                    else:
                        BCs.append(bc)
                        sizes.append(size)
                else:
                    BCs.append(np.zeros((1, 4)) - 1.0)
                    sizes.append(1)
                    
                if isinstance(self.data_dicts[idx]['loads'],list):
                    load = np.array(self.data_dicts[idx]['loads'])
                    size = load.shape[0]
                    if np.random.rand() < self.BC_dropout_prob:
                        BCs.append(np.zeros((1, 4)) - 1.0)
                        sizes.append(1)
                    else:
                        BCs.append(load)
                        sizes.append(size)
                else:
                    BCs.append(np.zeros((1, 4)) - 1.0)
                    sizes.append(1)
            if self.has_latent_tensors:
                # if self.normalization_mode == 'minmax':
                #     latent = (self.latent_tensors[idx] + self.shift)/self.scale * 2 - 1
                # else:
                latent = (self.latent_tensors[idx] + self.shift)/self.scale
        else:
            Cs = None
            BCs = None
            sizes = None
            latent = None
        
        return {
            'inp': inp,
            'gt': gt,
            'gt_coord': coord,
            'gt_cell': cell,
            'Cs': Cs,
            'BCs': BCs,
            'sizes': sizes,
            'latent': latent if self.has_latent_tensors else None,
        }
        
    def visualize(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self) - 1)
        samples = self[idx]
        # Set up the figure
        fig = plt.figure(figsize=(20, 10))

        # 1. Original Input Image
        ax = plt.subplot(1, 5, 1)
        img = F.to_tensor(self.data_dicts[idx]['topology'])
        scale = max(img.shape)
        # Convert from [-1,1] to [0,1] for visualization
        plt.imshow(img.permute(1, 2, 0).numpy().squeeze().T, cmap='Grays')
        # ax.set_title('Original Input')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        VF = samples['Cs'][1][0]
        AR = samples['Cs'][0][0]
        ax.set_title(f'Original Input (VF: {VF:.2f}, AR: {AR.min():.2f})')
        
        Loads = samples['BCs'][1]
        BC = samples['BCs'][0]
        if not np.all(Loads == -1):
            ax.quiver(Loads[:,0]*scale, Loads[:,1]*scale, Loads[:,2]*scale/10, Loads[:,3]*scale/10,
                      angles='xy', scale_units='xy', scale=1, lw=2, color='orange', label='Loads')
        if not np.all(BC == -1):
            x_only = BC[np.logical_and(BC[:, 2] == 1, BC[:, 3] == 0), 0:2]*scale
            y_only = BC[np.logical_and(BC[:, 2] == 0, BC[:, 3] == 1), 0:2]*scale
            xy_only = BC[np.logical_and(BC[:, 2] == 1, BC[:, 3] == 1), 0:2]*scale

            if len(x_only) > 0:
                ax.scatter(x_only[:, 0], x_only[:, 1], color='tomato', label='Fixed Along X', s=20)
            if len(y_only) > 0:
                ax.scatter(y_only[:, 0], y_only[:, 1], color='royalblue', label='Fixed Along Y', s=20)
            if len(xy_only) > 0:
                ax.scatter(xy_only[:, 0], xy_only[:, 1], color='limegreen', label='Fixed Along Both', s=20)
        
        ax.legend(loc='lower center', fontsize='small', markerscale=2, framealpha=0.5)

        # 2. Encoder Input
        ax = plt.subplot(1, 5, 2)
        inp_viz = (samples['inp'] + 1) / 2
        plt.imshow(inp_viz.permute(1, 2, 0).numpy(), cmap='Grays')
        ax.set_title(f'Encoder Input ({self.encoder_res}x{self.encoder_res})')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # 3. GT Patch
        ax = plt.subplot(1, 5, 3)
        patch = samples['gt']
        plt.imshow(patch.permute(1, 2, 0).numpy(), cmap='Grays')
        ax.set_title('Random GT Patch')
        ax.axis('off')

        # 4. Coordinate Grid Visualization
        ax = plt.subplot(1, 5, 4)
        coord = (samples['gt_coord'].reshape(-1, 2) + 1 )/2
        sizes = samples['gt_cell'].reshape(-1, 2)/2
        patch_vals = samples['gt'].reshape(-1)
        plt.imshow(inp_viz.permute(1, 2, 0).numpy(), cmap='Grays')
        # draw square around each cell
        for i in range(len(coord)):
            x, y = coord[i] * self.encoder_res - sizes[i][0]* self.encoder_res/2
            s = sizes[i] * self.encoder_res
            # face_alpha = f"#0000ff{int((patch_vals[i].numpy())*255):02x}"
            if patch_vals[i] > 0:
                face_alpha = f"#0000ff{int((patch_vals[i].numpy())*128):02x}"
            else:
                face_alpha = f"#ff0000{int((-patch_vals[i].numpy()+1)*128):02x}"
            rect = plt.Rectangle((x-s[0]/2, y-s[1]/2), s[0], s[1], linewidth=0, edgecolor='r', facecolor=face_alpha)
            ax.add_patch(rect)

        plt.tight_layout()
        plt.show()