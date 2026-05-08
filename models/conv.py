import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np
from einops import rearrange

def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array(
            [
                1.0,
            ]
        )
    elif filt_size == 2:
        a = np.array([1.0, 1.0])
    elif filt_size == 3:
        a = np.array([1.0, 2.0, 1.0])
    elif filt_size == 4:
        a = np.array([1.0, 3.0, 3.0, 1.0])
    elif filt_size == 5:
        a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filt_size == 6:
        a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filt_size == 7:
        a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt

def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer

class Downsample(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )

class EncodeBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        no_antialias=False,
        norm_layer=nn.BatchNorm2d,
        use_bias=False,
    ):
        super().__init__()
        layers = []
        if no_antialias:
            layers += [
                nn.Conv2d(
                    in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=use_bias
                ),
                norm_layer(out_dim),
                nn.ReLU(True),
            ]
        else:
            layers += [
                nn.Conv2d(
                    in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias
                ),
                norm_layer(out_dim),
                nn.ReLU(True),
                Downsample(out_dim),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def seq_to_map(x):
    if x.dim() == 4:
        return x
    B, N, C = x.shape
    H = int(N**0.5)
    if H * H != N:
        raise ValueError(
            f"Input sequence length N={N} is not a perfect square, cannot reshape to square map."
        )
    x = x.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
    return x

class Conv_Encode_Module(nn.Module):
    def __init__(
        self,
        input_type,
        in_dim,
        out_dim,
        n_blocks=3,
        norm_layer=nn.InstanceNorm2d,
        use_bias=True,
    ):
        super().__init__()
        self.input_type = input_type
        blocks = []
        ngf = in_dim
        mult = 1
        for i in range(n_blocks):
            in_dim_ = ngf * mult if ngf * mult < out_dim else out_dim
            out_dim_ = in_dim * 2 if in_dim * 2 < out_dim else out_dim
            blocks.append(EncodeBlock(in_dim_, out_dim_))
            mult *= 2
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if not self.input_type == "2d":
            B, N, C = x.shape
            H = int(np.sqrt(N))
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=H)
            x = seq_to_map(x)
        feat_map = self.blocks(x)
        out_vec = self.gap(feat_map).flatten(1)
        feat_seq = rearrange(feat_map, "b c h w -> b (h w) c")
        return out_vec, feat_seq

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.InstanceNorm2d):
        super(CBAMResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class CBAM_encode_Module(nn.Module):
    def __init__(
        self, input_type, in_dim, out_dim, n_blocks, norm_layer=nn.InstanceNorm2d
    ):
        super().__init__()
        self.input_type = input_type
        blocks = []
        cur_dim = in_dim
        for _ in range(n_blocks):
            in_dim = cur_dim
            out_dim = cur_dim * 2 if cur_dim * 2 < out_dim else out_dim
            blocks.append(
                CBAMResBlock(in_dim, out_dim, stride=2, norm_layer=norm_layer)
            )
            cur_dim *= 2
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if self.input_type != "2d":
            x = seq_to_map(x)
        feat_map = self.blocks(x)
        out_feat = self.gap(feat_map).flatten(1)
        return out_feat, feat_map
