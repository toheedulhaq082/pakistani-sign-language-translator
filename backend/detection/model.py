import torch
import torch.nn as nn

class SpatialGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = A
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = SpatialGCN(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), padding=(4,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if not residual:
            self.res = lambda x: 0
        elif (in_channels == out_channels) and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.res(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCNModel(nn.Module):
    def __init__(self, in_channels=2, num_class=52, A=None, dropout=0.3):
        super().__init__()
        NUM_JOINTS = A.shape[0]
        self.register_buffer('A', A)
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        self.layer1 = STGCNBlock(in_channels, 64, self.A)
        self.layer2 = STGCNBlock(64, 128, self.A, stride=1)
        self.layer3 = STGCNBlock(128, 256, self.A, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        N, C, T, V = x.shape
        x_resh = x.permute(0,2,1,3).contiguous().view(N, T, C*V)
        x_resh = x_resh.permute(0,2,1)
        x_bn = self.data_bn(x_resh)
        x_bn = x_bn.permute(0,2,1).contiguous().view(N, T, C, V).permute(0,2,1,3)
        x = x_bn
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(N, -1)
        x = self.dropout(x)
        return self.fc(x)
