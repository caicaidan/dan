import torch
import torch.nn as nn



class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)
    


    
# 24 depth
class ShallowConvNet(nn.Module):
    def __init__(self, args, F1=8, F2=16, D=2, latent_dim=128, depth=24, dropout=0.25):
        super(ShallowConvNet, self).__init__()
        # Convert args.ntimes and args.nchans to integers
        self.ntimes = int(args.ntimes)
        self.nchans = int(args.nchans)
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.latent_dim = latent_dim
        self.depth = depth
        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.depth), stride=1, padding=(0, self.depth // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nchans, 1), max_norm=1, stride=1, padding=(0, 0), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout))

        self.linear = nn.Linear(self.feature_dim(), latent_dim, bias=False)


    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.nchans, self.ntimes)
            mock_eeg = self.block(mock_eeg)
            return int(np.prod(mock_eeg.size()[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.block(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x

# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, max_norm: int = 1, **kwargs):
#         self.max_norm = max_norm
#         super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
#         return super(Conv2dWithConstraint, self).forward(x)


# class ShallowConvNet(nn.Module):

#     def __init__(self,
#                  ntimes: int = 1000,
#                  nchans: int = 22,
#                  F1: int = 8,
#                  F2: int = 16,
#                  D: int = 2,
#                  latent_dim: int = 128,
#                  depth: int = 24,
#                  dropout: float = 0.25):
#         super(ShallowConvNet, self).__init__()
#         self.F1 = F1
#         self.F2 = F2
#         self.D = D
#         self.ntimes = ntimes
#         self.num_classes = latent_dim
#         self.nchans = nchans
#         self.depth = depth

#         self.dropout = dropout

#         self.block = nn.Sequential(
#             nn.Conv2d(1, self.F1, (1, self.depth), stride=1, padding=(0, self.depth // 2), bias=False),
#             nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
#             Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nchans, 1), max_norm=1, stride=1, padding=(0, 0),
#                                  groups=self.F1, bias=False),
#             nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3), nn.ELU(),
#             nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

#         self.linear = nn.Linear(self.feature_dim(), latent_dim, bias=False)

#     def feature_dim(self):
#         with torch.no_grad():
#             mock_eeg = torch.zeros(1, 1, self.nchans, self.ntimes)

#             mock_eeg = self.block(mock_eeg)

#         return self.F2 * mock_eeg.shape[3]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.float()
#         x = self.block(x)
#         x = x.flatten(start_dim=1)
#         x = self.linear(x)
#         return x
