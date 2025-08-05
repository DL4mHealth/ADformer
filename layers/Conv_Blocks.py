import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):  # handle different spatial sizes
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)  # register kernels by ModuleList
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        # Compute the padding size required for causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups
        )

    def forward(self, x):
        # only left-side padding is required
        x = F.pad(x, (self.padding, 0))
        # Perform the convolution
        out = self.conv(x)
        return out


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = CausalConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            DilatedConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


class TemporalSpatialConv(nn.Module):
    def __init__(self, channels=64, dropout=0.5):
        super(TemporalSpatialConv, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (1, 1), padding='valid', bias=False)  # Kernel size will be adjusted in forward
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        self.conv2 = nn.Conv2d(16, 32, (channels, 1), padding='valid', bias=False, groups=16)
        self.batchnorm2 = nn.BatchNorm2d(32, False)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 16))  # Adaptive pooling
        self.dropout1 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(32, 32, (1, 8), padding='same', bias=False, groups=32)
        self.conv4 = nn.Conv2d(32, 32, (1, 1), padding='valid', bias=False, groups=32)
        self.batchnorm3 = nn.BatchNorm2d(32, False)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 4))  # Adaptive pooling
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, time_points, channels)
        x = x.unsqueeze(1)  # (batch_size, 1, time_points, channels)
        time_points = x.shape[2]

        # Adjust first conv kernel size
        self.conv1.kernel_size = (1, time_points)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        return x


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        bottleneck_channels,
        activation,
        dropout,
    ):
        super().__init__()
        self.activation = activation
        self.use_bottleneck = bottleneck_channels is not None and bottleneck_channels > 0

        # 1×1 bottleneck
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if self.use_bottleneck else nn.Identity()
        )
        branch_in = bottleneck_channels if self.use_bottleneck else in_channels

        # 每个分支输出通道数均等
        branch_out = out_channels // len(kernel_sizes)
        self.branches = nn.ModuleList([
            nn.Conv1d(
                branch_in, branch_out,
                kernel_size=k, stride=1,
                padding=k // 2,  # SAME padding
                bias=False)
            for k in kernel_sizes
        ])

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        x = self.bottleneck(x)
        outs = [branch(x) for branch in self.branches]          # [ [B, out_channels // num_kernels, T] for each branch ]
        x = torch.cat(outs, dim=1)                              # [B, out_channels, T]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.drop(x)
        return x


class SpatialBlock(nn.Module):
    """
    Depthwise-Separable Spatial Block for 1D time series data.
    """
    def __init__(self, in_channels: int, depth_multiplier: int = 2, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels * depth_multiplier,
            kernel_size=(1, 1), groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier, in_channels,
            kernel_size=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = activation

    def forward(self, x):  # x: [B, C, T]  →  [B, C, 1, T]
        x = x.unsqueeze(2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x.squeeze(2)            # back to [B, C, T]


