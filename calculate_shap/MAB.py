import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self._init_weight_()

    def forward(self, x, *args):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _init_weight_(self):
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        
        nn.init.trunc_normal_(self.proj.weight, std=.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def _init_weight_(self):
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

class MultiAttentionBlock(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_layer=Attention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, **kwargs):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiAttentionBlock_V2(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_layer=nn.MultiheadAttention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(embed_dim=dim, num_heads=num_heads)


    def forward(self, x, **kwargs):
        attn_output, attn_output_weights = self.attn(x, x, x)
        x = x + self.norm1(attn_output)
        return x
    