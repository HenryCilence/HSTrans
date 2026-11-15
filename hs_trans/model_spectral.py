import torch
from torch import nn

def _noncyclic_shift(layer, x):
    shape = x.shape
    x = x.flatten(2)
    length = x.shape[1]
    group_length = length // layer.num_group
    half = group_length // 2
    chunks = [x[:, :half]]
    chunks.extend((x[:, start:start + group_length] for start in range(half, length - half, group_length)))
    chunks.append(x[:, length - half:])
    return layer.norm(torch.cat([layer.mhsa(chunk) for chunk in chunks], dim=1)).reshape(shape)

class Attention(nn.Module):

    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpectralInGroupAttention(nn.Module):

    def __init__(self, dim, num_group, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.mhsa = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_group = num_group

    def forward(self, x):
        (B, N, D) = x.shape
        g = self.num_group
        group_channel = N // g
        x = x.reshape(B, g, group_channel, D)
        x = x.reshape(B * g, group_channel, D)
        x = self.mhsa(x)
        x = x.reshape(B, g, group_channel, D)
        x = x.reshape(B, N, D)
        return x

class SpectralCrossGroupAttention(nn.Module):

    def __init__(self, dim, input_channel, num_group, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.group_channel = input_channel // num_group
        self.ds = nn.Conv1d(dim, dim, kernel_size=self.group_channel, stride=self.group_channel)
        self.mhsa = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_group = num_group

    def forward(self, x):
        _feature = x
        (B, N, D) = x.shape
        x = x.transpose(1, 2)
        x = self.ds(x)
        x = x.transpose(1, 2)
        x = self.mhsa(x)
        gc = self.group_channel
        x = x.repeat_interleave(gc, dim=1)
        x = x.reshape(B, N, D)
        x = x * _feature
        return x

class SpectralShiftWindowAttention(nn.Module):

    def __init__(self, dim, num_group, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, shift_mode='cyclic'):
        super().__init__()
        self.mhsa = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_group = num_group
        self.norm = nn.LayerNorm(dim)
        if shift_mode not in ('cyclic', 'noncyclic'):
            raise ValueError('invalid shift_mode')
        self.shift_mode = shift_mode

    def forward(self, x):
        if self.shift_mode == 'noncyclic':
            return _noncyclic_shift(self, x)
        (B, N, D) = x.shape
        g = self.num_group
        group_channel = N // g
        shift = group_channel // 2
        x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
        x = x.reshape(B, g, group_channel, D)
        x = x.reshape(B * g, group_channel, D)
        x = self.mhsa(x)
        x = x.reshape(B, g, group_channel, D)
        x = x.reshape(B, N, D)
        x = self.norm(x)
        x = torch.cat([x[:, -shift:], x[:, :-shift]], dim=1)
        return x

class SpectralEncoder(nn.Module):

    def __init__(self, dim, input_channel, num_group):
        super().__init__()
        self.iga = SpectralInGroupAttention(dim, num_group)
        self.cga = SpectralCrossGroupAttention(dim, input_channel, num_group)
        self.siga = SpectralShiftWindowAttention(dim, num_group)

    def forward(self, x):
        x = self.iga(x)
        x = self.siga(x)
        x = self.cga(x)
        return x

class SpectralStage(nn.Module):

    def __init__(self, input_size, input_channel, spectral_group):
        super().__init__()
        self.spectral_encoder = SpectralEncoder(input_size, input_channel, spectral_group)

    def forward(self, x):
        return self.spectral_encoder(x)

class SpectralCalibration(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GSC(nn.Module):

    def __init__(self, dim_in, dim_out, padding=1, num_groups=8):
        super().__init__()
        self.gpwc = nn.Conv1d(dim_in, dim_out, groups=num_groups, kernel_size=1)
        self.bn = nn.BatchNorm1d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.gpwc(x)
        x = self.bn(x)
        self.relu(x)
        return x
