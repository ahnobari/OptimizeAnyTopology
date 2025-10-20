import torch

def get_center_pad_info(h, w):
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    pad_h2 = size - h - pad_h
    pad_w2 = size - w - pad_w
    
    return (pad_h, pad_h2, pad_w, pad_w2), size

def get_full_grid_cell(img_shape):
    h, w = img_shape
    
    pad_info, orig_size = get_center_pad_info(h, w)
    
    start_h = pad_info[0]
    start_w = pad_info[2]
    end_h = orig_size - pad_info[1]
    end_w = orig_size - pad_info[3]
    
    # Calculate relative coordinates for the patch
    rel_start_h = start_h / orig_size
    rel_start_w = start_w / orig_size
    rel_end_h = end_h / orig_size
    rel_end_w = end_w / orig_size
    
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
    
    cell[:] = torch.tensor([2/orig_size, 2/orig_size])
    
    return coord, cell
    
def make_coord_cell_grid(size, range=(-1, 1)):
    """Generate coordinate and cell-size grids"""
    h, w = size if isinstance(size, tuple) else (size, size)
    
    # Handle range being a list of ranges or a single range
    if isinstance(range[0], (list, tuple)):
        x_range = range[0]
        y_range = range[1]
    else:
        x_range = range
        y_range = range
        
    # Generate coordinates
    x = torch.linspace(x_range[0], x_range[1], w)
    y = torch.linspace(y_range[0], y_range[1], h)
    
    y, x = torch.meshgrid(y, x, indexing='ij')
    coord = torch.stack([x, y], dim=-1)  # H, W, 2
    
    # Generate cell sizes
    cell = torch.ones_like(coord)
    cell[..., 0] *= (x_range[1] - x_range[0]) / w
    cell[..., 1] *= (y_range[1] - y_range[0]) / h
    
    return coord, cell

def center_pad_square(img, fill=0):
        """Center pad tensor to square"""
        _, h, w = img.shape
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        pad_h2 = size - h - pad_h
        pad_w2 = size - w - pad_w
        
        return torch.nn.functional.pad(img, (pad_w, pad_w2, pad_h, pad_h2), value=fill), (pad_h, pad_h2, pad_w, pad_w2)


class BatchDict(dict):
    """
    A dictionary that allows for attribute-style access to its keys.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
    def to(self, device):
        """
        Move all tensors in the dictionary to the specified device.
        """
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
            elif isinstance(value, list):
                self[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
            elif isinstance(value, dict):
                self[key] = BatchDict(value).to(device)
        return self