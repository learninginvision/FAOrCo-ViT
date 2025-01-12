import torch

def transform_aug_reverse(x, augment, aug_min_val=0, aug_max_val=1.0, x_min_val=None, x_max_val=None, clip=True):
    l = x_min_val if x_min_val is not None else torch.min(x)
    h = x_max_val if x_max_val is not None else torch.max(x)

    # [l, h] --> [0, 1]
    x = (x - l) / (h - l + 1e-8)

    # [0, 1] --> [low, high]
    x = x * (aug_max_val - aug_min_val) + aug_min_val

    # Apply augmentation
    x = augment(x)

    # Clip values
    if clip:
        x = torch.clamp(x, aug_min_val, aug_max_val)

    # [low, high] --> [l, h]
    x = (x - aug_min_val) / (aug_max_val - aug_min_val) * (h - l + 1e-8) + l
    return x

def get_random_brightness(max_delta=0.1, clip=False):
    def _random_brightness(image):
        # Random value in [-max_delta, +max_delta]
        delta = torch.empty(1).uniform_(-max_delta, max_delta).item()
        return image + delta
    
    def tar(x):
        return transform_aug_reverse(
            x, augment=_random_brightness,
            aug_min_val=0, aug_max_val=1.0, clip=clip)
    
    return tar

def get_random_brightness_per_channel_v2(max_delta=0.1, clip=True):
    random_brightness = get_random_brightness(max_delta, clip)

    def _random_brightness_pc(x):
        # Split channels
        x = torch.unsqueeze(x, dim=2) # (H, W, 1, C)
        x = torch.split(x, 1, dim=-1) # C x (H, W, 1)
        
        # Apply brightness augmentation to each channel
        x = [random_brightness(x_i.squeeze(2)) for x_i in x]
        
        # Concatenate the channels back
        return torch.cat(x, dim=-1)
    
    return _random_brightness_pc

def brightness_augmentation(x, max_delta=0.1, aug_min_val=0, aug_max_val=1.0, 
                            x_min_val=None, x_max_val=None, clip=True):
    bz = x.shape[0]
    dim = x.shape[-1]
    H = W = int(x.shape[1] ** 0.5)
    
    x = x.view(bz, H, W, dim)
    x = torch.split(x, 1, dim=-1) # C x (B, H, W)
    # x = torch.unsqueeze(x, dim=3) # (B, H, W, 1, C)
    # x = torch.split(x, 1, dim=-1) # C x (B, H, W, 1)
    
    bathc_aug_x = []
    
    for x_i in x:
        
        x_i = torch.squeeze(x_i, dim=-1) # (B, H, W)
        aug_x = []
        
        for i in range(bz):
            # x_i[i] = x_i[i].squeeze(0)
            l = x_min_val if x_min_val is not None else torch.min(x_i[i])
            h = x_max_val if x_max_val is not None else torch.max(x_i[i])
            
            # [l, h] --> [0, 1]
            x_i[i] = (x_i[i] - l) / (h - l + 1e-8)
            # [0, 1] --> [low, high]
            x_i[i] = x_i[i] * (aug_max_val - aug_min_val) + aug_min_val
            
            delta = torch.empty(1).uniform_(-max_delta, max_delta).item()
            
            x_i[i] = x_i[i] + delta
            
            if clip:
                x_i[i] = torch.clamp(x_i[i], aug_min_val, aug_max_val)
            
            x_i[i] = (x_i[i] - aug_min_val) / (aug_max_val - aug_min_val) * (h - l + 1e-8) + l
            
            aug_x.append(x_i[i])
        
        aug_x = torch.stack(aug_x, dim=0)
            
        
        bathc_aug_x.append(aug_x)
    
    bathc_aug_x = torch.stack(bathc_aug_x, dim=0).permute(1, 2, 3, 0)
    bathc_aug_x = bathc_aug_x.view(bz, H*W, dim)
    
    return bathc_aug_x


def brightness_augmentation_v2(x, max_delta=0.1, aug_min_val=0, aug_max_val=1.0, 
                            x_min_val=None, x_max_val=None, clip=True):
    
    x = torch.split(x, 1, dim=-1) # 768 x (128, 196, 1)
    # x = torch.unsqueeze(x, dim=3) # (B, H, W, 1, C)
    # x = torch.split(x, 1, dim=-1) # C x (B, H, W, 1)
    
    batch_aug_x = []
    
    for x_i in x:
        
        x_i = torch.squeeze(x_i, dim=-1) # (128, 196)
        
        x_min_val = torch.min(x_i, dim=1)[0].view(-1, 1)
        x_max_val = torch.max(x_i, dim=1)[0].view(-1, 1)
        
        x_i = (x_i - x_min_val) / (x_max_val - x_min_val + 1e-8)
        
        x_i = x_i * (aug_max_val - aug_min_val) + aug_min_val
        
        delta = torch.empty(x_i.shape[0]).uniform_(-max_delta, max_delta).to(x_i.device)
        
        x_i = x_i + delta.view(-1, 1)
        
        if clip:
                x_i = torch.clamp(x_i, aug_min_val, aug_max_val)
            
        x_i = (x_i - aug_min_val) / (aug_max_val - aug_min_val) * (x_max_val - x_min_val + 1e-8) + x_min_val
        
        batch_aug_x.append(x_i)
        
    batch_aug_x = torch.stack(batch_aug_x, dim=0).permute(1, 2, 0)
    
    
    return batch_aug_x

def brightness_augmentation_v3(x, max_delta=0.1, aug_min_val=0, aug_max_val=1.0, 
                            x_min_val=None, x_max_val=None, clip=True):
    
    
    x_min_val, _ = torch.min(x, dim=1, keepdim=True)
    x_max_val, _ = torch.max(x, dim=1, keepdim=True)
    
    denorm = x_max_val - x_min_val + 1e-8
    
    x_norm = (x - x_min_val) / denorm
    
    delta = torch.empty(x_norm.shape[0], 1, x_norm.shape[2]).uniform_(-max_delta, max_delta).to(x_norm.device)
    
    x_norm = x_norm + delta
    
    if clip:
        x_norm = torch.clamp(x_norm, aug_min_val, aug_max_val)
    
    x_aug = x_norm * denorm + x_min_val
    
    return x_aug


def contrast_augmentation(x, scale_factor=1.0, aug_min_val=0, aug_max_val=1.0, clip=True):
    
    
    x_min_val, _ = x.view(x.size(0), -1).min(dim=1, keepdim=True)  # Shape (B, 1)
    x_max_val, _ = x.view(x.size(0), -1).max(dim=1, keepdim=True)  # Shape (B, 1)
    
    # Reshape min and max to allow broadcasting
    x_min_val = x_min_val.view(-1, 1, 1)  # Shape (B, 1, 1)
    x_max_val = x_max_val.view(-1, 1, 1)  # Shape (B, 1, 1)
    
    denorm = x_max_val - x_min_val + 1e-8
    
    x_norm = (x - x_min_val) / denorm
    
    delta = torch.empty(x_norm.shape[0], 1, x_norm.shape[2]).uniform_(1/scale_factor, scale_factor).to(x_norm.device)
    
    x_norm = x_norm + delta
    
    if clip:
        x_norm = torch.clamp(x_norm, aug_min_val, aug_max_val)
    
    x_aug = x_norm * denorm + x_min_val
    
    return x_aug

if __name__=="__main__":
    # random_brightness = get_random_brightness_per_channel_v2()
    image = torch.rand(10, 196, 768)
    augmented_image = brightness_augmentation_v2(image)
    # augmented_image = random_brightness(image)
    print(augmented_image.shape)