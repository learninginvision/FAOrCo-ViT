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

if __name__=="__main__":
    random_brightness = get_random_brightness_per_channel_v2()
    image = torch.rand(10, 196, 768)
    augmented_image = random_brightness(image)
    print(augmented_image.shape)