import torch
from torch import nn
from typing import List

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class DCN(nn.Module):
    def __init__(self,
                 args,
                 latent_dim: int = 64,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.,
                 block_out_channels: List[int] = [25, 25, 50, 100, 200],#[8, 16, 32, 64, 128],
                 pool_size: int = 4,
                 weight_init_method=None):
        super(DCN, self).__init__()
        self.nchans = args.nchans
        self.ntimes = args.ntimes

        self.first_conv_block = nn.Sequential(
            Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, kernel_1), max_norm=2, padding=(0, kernel_1 // 2)),
            Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(self.nchans, 1), bias=False,max_norm=2),
            nn.BatchNorm2d(block_out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.deep_block = nn.ModuleList(
            [self.default_block(block_out_channels[i - 1], block_out_channels[i], kernel_2, pool_size) for i in
             range(2, 5)]
        )

        self.linear = nn.Linear(self.feature_dim(args), latent_dim, bias=False)

    def default_block(self, in_channels, out_channels, T, P):
        block = nn.Sequential(
            nn.Dropout(0.25),
            Conv2dWithConstraint(in_channels, out_channels, (1, T), bias=False, max_norm=2, padding=(0, T // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.AvgPool2d((1, P))
        )
        return block

    def feature_dim(self,args):
        self.nchans = args.nchans
        self.ntimes = args.ntimes
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.nchans, self.ntimes)
            mock_eeg = self.first_conv_block(mock_eeg)
            for block in self.deep_block:
                mock_eeg = block(mock_eeg)
        return mock_eeg.shape[1] * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.first_conv_block(x)
        for block in self.deep_block:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x



###############################2a###################################################
# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, max_norm: int = 1, **kwargs):
#         self.max_norm = max_norm
#         super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
#         return super(Conv2dWithConstraint, self).forward(x)

# class DCN(nn.Module):
#     def __init__(self,
#                  ntimes: int = 1000,
#                  nchans: int = 22,
#                  latent_dim: int = 64,
#                  kernel_1: int = 64,
#                  kernel_2: int = 16,
#                  dropout: float = 0.,
#                  block_out_channels: List[int] = [8, 16, 32, 64, 128],
#                  pool_size: int = 4,
#                  weight_init_method=None):
#         super(DCN, self).__init__()
#         self.nchans = nchans
#         self.ntimes = ntimes

#         self.first_conv_block = nn.Sequential(
#             Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, kernel_1), max_norm=2, padding=(0, kernel_1 // 2)),
#             Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(nchans, 1), bias=False,
#                                  max_norm=2),
#             nn.BatchNorm2d(block_out_channels[1]),
#             nn.ELU(),
#             nn.AvgPool2d((1, pool_size)),
#             nn.Dropout(p=dropout)
#         )

#         self.deep_block = nn.ModuleList(
#             [self.default_block(block_out_channels[i - 1], block_out_channels[i], kernel_2, pool_size, dropout) for i in
#              range(2, len(block_out_channels))]
#         )

#         self.linear = nn.Linear(self.feature_dim(), latent_dim, bias=False)

#     def default_block(self, in_channels, out_channels, T, P, dropout):
#         block = nn.Sequential(
#             nn.Dropout(dropout),
#             Conv2dWithConstraint(in_channels, out_channels, (1, T), bias=False, max_norm=2, padding=(0, T // 2)),
#             nn.BatchNorm2d(out_channels),
#             nn.ELU(),
#             nn.AvgPool2d((1, P))
#         )
#         return block

#     def feature_dim(self):
#         with torch.no_grad():
#             mock_eeg = torch.zeros(1, 1, self.nchans, self.ntimes)
#             mock_eeg = self.first_conv_block(mock_eeg)
#             for block in self.deep_block:
#                 mock_eeg = block(mock_eeg)
#         return mock_eeg.shape[1] * mock_eeg.shape[3]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.first_conv_block(x)
#         for block in self.deep_block:
#             x = block(x)
#         x = x.flatten(start_dim=1)
#         x = self.linear(x)
#         return x






