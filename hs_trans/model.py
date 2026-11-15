import torch
from torch import nn
from . import model_spectral
Attention = model_spectral.Attention
SpectralShiftWindowAttention = model_spectral.SpectralShiftWindowAttention

class HS_Trans(nn.Module):

    def __init__(self, in_bands, num_classes, patch_size=8, variant='spectral', shift_mode='cyclic', components='igi+sigi+cgi'):
        super().__init__()
        if not isinstance(in_bands, int) or in_bands < 1:
            raise ValueError('in_bands must be a positive integer')
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError('num_classes must be a positive integer')
        if not isinstance(patch_size, int) or patch_size < 2 or patch_size % 2:
            raise ValueError('patch_size must be an even integer >= 2')
        if variant != 'spectral':
            raise ValueError('Only the spectral model is supported')
        if shift_mode not in ('cyclic', 'noncyclic'):
            raise ValueError("shift_mode must be 'cyclic' or 'noncyclic'")
        if components not in ('igi', 'igi+sigi', 'cgi', 'igi+cgi', 'igi+sigi+cgi'):
            raise ValueError('Unsupported component combination')
        self.in_bands = in_bands
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.shift_mode = shift_mode
        self.components = components
        self.sc = model_spectral.SpectralCalibration(in_bands, 512)
        self.gsc1 = model_spectral.GSC(512, 256, num_groups=8)
        self.ss_encoder1 = model_spectral.SpectralStage(input_size=patch_size ** 2, input_channel=256, spectral_group=8)
        self.bn1 = nn.BatchNorm1d(256)
        self.gsc2 = model_spectral.GSC(256, 128, num_groups=4)
        self.ss_encoder2 = model_spectral.SpectralStage(input_size=patch_size ** 2, input_channel=128, spectral_group=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, num_classes)
        for module in self.modules():
            if isinstance(module, model_spectral.SpectralShiftWindowAttention):
                module.shift_mode = shift_mode
        enabled = set(components.split('+'))
        for stage in (self.ss_encoder1, self.ss_encoder2):
            for component, name in (('igi', 'iga'), ('sigi', 'siga'), ('cgi', 'cga')):
                if component not in enabled:
                    setattr(stage.spectral_encoder, name, nn.Identity())

    def forward(self, x):
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        expected = (self.in_bands, self.patch_size, self.patch_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f'expected [batch, {expected[0]}, {expected[1]}, {expected[2]}] input')
        x = self.sc(x)
        x = x.flatten(2)
        x = self.gsc1(x)
        x = self.ss_encoder1(x) + x
        x = torch.relu(self.bn1(x))
        x = self.gsc2(x)
        x = self.ss_encoder2(x) + x
        x = torch.relu(self.bn2(x))
        x = self.avg_pool(x).flatten(1)
        return self.head(self.norm(x))
