import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):

    def __init__(self,
                 args,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 latent_dim: int = 128,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.ntimes = args.ntimes
        self.num_classes = latent_dim
        self.nchans = args.nchans
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.nchans, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.linear = nn.Linear(self.feature_dim(), latent_dim, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.nchans, self.ntimes)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x



#################################################################################

# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, max_norm: int = 1, **kwargs):
#         self.max_norm = max_norm
#         super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
#         return super(Conv2dWithConstraint, self).forward(x)


# class EEGNet(nn.Module):

#     def __init__(self,
#                  ntimes: int = 750,
#                  nchans: int = 3,
#                  F1: int = 8,
#                  F2: int = 16,
#                  D: int = 2,
#                  latent_dim: int = 128,
#                  kernel_1: int = 8,   #64
#                  kernel_2: int = 2,   #16
#                  dropout: float = 0.25):
#         super(EEGNet, self).__init__()
#         self.F1 = F1
#         self.F2 = F2
#         self.D = D
#         self.ntimes = ntimes
#         self.num_classes = latent_dim
#         self.nchans = nchans
#         self.kernel_1 = kernel_1
#         self.kernel_2 = kernel_2
#         self.dropout = dropout

#         self.block1 = nn.Sequential(
#             nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
#             nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
#             Conv2dWithConstraint(self.F1,
#                                  self.F1 * self.D, (self.nchans, 1),
#                                  max_norm=1,
#                                  stride=1,
#                                  padding=(0, 0),
#                                  groups=self.F1,
#                                  bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
#             nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

#         self.block2 = nn.Sequential(
#             nn.Conv2d(self.F1 * self.D,
#                       self.F1 * self.D, (1, self.kernel_2),
#                       stride=1,
#                       padding=(0, self.kernel_2 // 2),
#                       bias=False,
#                       groups=self.F1 * self.D),
#             nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
#             nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
#             nn.Dropout(p=dropout))

#         self.linear = nn.Linear(self.feature_dim(), latent_dim, bias=False)

#     def feature_dim(self):
#         with torch.no_grad():
#             mock_eeg = torch.zeros(1, 1, self.nchans, self.ntimes)

#             mock_eeg = self.block1(mock_eeg)
#             mock_eeg = self.block2(mock_eeg)

#         return self.F2 * mock_eeg.shape[3]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.block1(x)
#         x = self.block2(x)
#         x = x.flatten(start_dim=1)
#         x = self.linear(x)
#         return x
