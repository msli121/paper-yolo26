# @Time       : 2026/2/28 15:50
# @File       : ABRNet_ARF_MCA_TGRS2025.py
# @Description: ARF 模块通过自适应调整感受野，增强了小目标特征提取的能力，而 BC 模块则通过多尺度交叉轴注意力融合全局信息，提高了目标定位的准确性和网络的鲁棒性。两者的结合使得 ABRNet 在红外小目标检测任务中取得了更好的性能。

import numbers

import torch
import torch.nn as nn
from einops import rearrange

from ultralytics.nn.modules import C3, C2f, Conv
from ultralytics.nn.modules.block import PSABlock

__all__ = ["C3k2_ARF", "C3k2_MCA"]


class ARFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定义多个卷积核大小的卷积层
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        # 平均池化和最大池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # 1x1卷积
        self.conv1x1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)  # 7,3     3,1
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分别通过3x3, 5x5, 7x7卷积
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(conv3x3_out)
        conv7x7_out = self.conv7x7(conv5x5_out)
        # 拼接所有卷积的结果
        out = torch.cat([conv3x3_out, conv5x5_out, conv7x7_out], dim=1)

        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        Fs = torch.cat([avg_out, max_out], dim=1)
        Fs = self.conv1(Fs)
        out = self.sigmoid(Fs) * out
        out = self.conv1x1(out)

        return out * x


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class MCAttention(nn.Module):
    def __init__(self, dim, num_heads=8, LayerNorm_type="WithBias"):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        _b, _c, h, w = x.shape
        x1 = self.norm1(x)
        attn_00 = self.conv0_1(x1)
        attn_01 = self.conv0_2(x1)
        attn_10 = self.conv1_1(x1)
        attn_11 = self.conv1_2(x1)
        attn_20 = self.conv2_1(x1)
        attn_21 = self.conv2_2(x1)
        out1 = attn_00 + attn_10 + attn_20
        out2 = attn_01 + attn_11 + attn_21
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, "b (head c) h w -> b head h (w c)", head=self.num_heads)
        v1 = rearrange(out1, "b (head c) h w -> b head h (w c)", head=self.num_heads)
        k2 = rearrange(out2, "b (head c) h w -> b head w (h c)", head=self.num_heads)
        v2 = rearrange(out2, "b (head c) h w -> b head w (h c)", head=self.num_heads)
        q2 = rearrange(out1, "b (head c) h w -> b head w (h c)", head=self.num_heads)
        q1 = rearrange(out2, "b (head c) h w -> b head h (w c)", head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = q1 @ k1.transpose(-2, -1)
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = q2 @ k2.transpose(-2, -1)
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, "b head h (w c) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, "b head w (h c) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.sigmod(self.project_out(out3) + self.project_out(out4)) * x
        return out


# --------
class Bottleneck_MCA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = MCAttention(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C3k_MCA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_MCA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_MCA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck_MCA(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k_MCA(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_MCA(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k_ARF(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ARFModule(c_, c_) for _ in range(n)))


class C3k2_ARF(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                ARFModule(self.c, self.c),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k_ARF(self.c, self.c, 2, shortcut, g)
            if c3k
            else ARFModule(self.c, self.c)
            for _ in range(n)
        )
