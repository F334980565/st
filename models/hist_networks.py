import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import init
from .conv import Conv_Encode_Module
from .stylegan_networks import FusedLeakyReLU, ModulatedConv2d, NoiseInjection, ToRGB
from .trans import Trans_encode_Module

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

class Upsample(nn.Module):
    def __init__(self, channels, pad_type="repl", filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad(inp),
            self.filt,
            stride=self.stride,
            padding=1 + self.pad_size,
            groups=inp.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

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

def init_weights(net, init_type="normal", init_gain=0.02, log_layers=False):
    def init_func(m):
        classname = m.__class__.__name__
        if classname == "ModulatedConv2d":
            return
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if log_layers:
                print(classname)
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

def init_net(
    net,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    log_layers=False,
    initialize_weights=True,
):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, log_layers=log_layers)
    return net

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class EncodeBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        no_antialias=False,
        norm_layer=nn.BatchNorm2d,
        use_bias=False,
        is_first_block=False,
    ):
        super().__init__()
        self.is_first_block = is_first_block
        self.no_antialias = no_antialias
        if is_first_block:
            self.pad = nn.ReflectionPad2d(3)
            self.conv = nn.Conv2d(
                in_dim, out_dim, kernel_size=7, padding=0, bias=use_bias
            )
            self.norm = norm_layer(out_dim)
            self.act = nn.ReLU(True)
            self.pool = None
        else:
            self.pad = None
            if no_antialias:
                self.conv = nn.Conv2d(
                    in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=use_bias
                )
                self.norm = norm_layer(out_dim)
                self.act = nn.ReLU(True)
                self.pool = None
            else:
                self.conv = nn.Conv2d(
                    in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=use_bias
                )
                self.norm = norm_layer(out_dim)
                self.act = nn.ReLU(True)
                self.pool = Downsample(out_dim)

    def forward(self, x, return_feat=False):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        if return_feat:
            intermediate = x
        x = self.norm(x)
        x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)
        if return_feat:
            return x, intermediate
        else:
            return x

class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer, dropout, use_bias):
        super(ResBlock, self).__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm1 = norm_layer(dim)
        self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm2 = norm_layer(dim)

    def forward(self, x, style=None, return_feat=False):
        h = self.pad(x)
        h = self.conv1(h)
        if return_feat:
            intermediate = h
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.pad(h)
        h = self.conv2(h)
        h = self.norm2(h)
        if return_feat:
            return h + x, intermediate
        else:
            return x + h

class Mod_ResBlock(nn.Module):
    def __init__(
        self,
        dim,
        style_dim,
        alpha=1.0,
        demodulate=True,
        inject_noise=False,
        norm_layer=nn.InstanceNorm2d,
        dropout=0.0,
        use_bias=False,
    ):
        super().__init__()
        self.alpha = alpha
        self.conv1 = ModulatedConv2d(dim, dim, 3, style_dim, demodulate)
        self.conv2 = ModulatedConv2d(dim, dim, 3, style_dim, demodulate)
        if style_dim is not None:
            self.style_activate = FusedLeakyReLU(dim)
        self.activate = nn.ReLU(True)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.inject_noise = inject_noise
        if inject_noise:
            self.noise = NoiseInjection()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, style=None, return_feat=False):
        res = x
        if style is not None:
            h = self.conv1(x, style)
            if return_feat:
                intermediate = h
            if self.inject_noise:
                h = self.noise(h)
            h = self.style_activate(h)
            h = self.dropout(h)
            h = self.conv2(h, style)
            if self.inject_noise:
                h = self.noise(h)
        else:
            h = self.conv1(x, style)
            if return_feat:
                intermediate = h
            h = self.norm1(h)
            h = self.activate(h)
            h = self.dropout(h)
            h = self.conv2(h, style)
            h = self.norm2(h)
        if return_feat:
            return h + res, intermediate
        else:
            return x + self.alpha * h

class Mod_DecodeBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        style_dim,
        alpha=1.0,
        demodulate=True,
        inject_noise=False,
        norm_layer=nn.BatchNorm2d,
        dropout=0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.conv1 = ModulatedConv2d(
            in_dim, out_dim, 3, style_dim, demodulate, upsample=True
        )
        self.conv2 = ModulatedConv2d(out_dim, out_dim, 3, style_dim, demodulate)
        self.norm = norm_layer(out_dim)
        self.activate = nn.ReLU(True)
        if style_dim is not None:
            self.style_activate1 = FusedLeakyReLU(out_dim)
            self.style_activate2 = FusedLeakyReLU(out_dim)
        self.inject_noise = inject_noise
        if inject_noise:
            self.noise = NoiseInjection()

    def forward(self, x, style=None, return_feat=False):
        if style is not None:
            h = self.conv1(x, style)
            if return_feat:
                intermediate = h
            if self.inject_noise:
                h = self.noise(h)
            h = self.style_activate1(h)
            h = self.conv2(h, style)
            if self.inject_noise:
                h = self.noise(h)
            h = self.style_activate2(h)
        else:
            h = self.conv1(x, style)
            if return_feat:
                intermediate = h
            h = self.norm(h)
            h = self.activate(h)
            h = self.conv2(h, style)
            h = self.norm(h)
            h = self.activate(h)
        if return_feat:
            return h, intermediate
        else:
            return h

class DecodeBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.BatchNorm2d, dropout=0.0):
        super().__init__()
        self.up_conv1 = Upsample(in_dim)
        self.conv2 = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm_layer = norm_layer(out_dim)
        self.activate = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, style=None, return_feat=False):
        h = self.up_conv1(x)
        if return_feat:
            intermediate = h
        h = self.conv2(h)
        h = self.norm_layer(h)
        h = self.activate(h)
        if return_feat:
            return h, intermediate
        else:
            return h

class ToRGB_noMod(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.BatchNorm2d, dropout=0.0):
        super().__init__()
        self.pad = nn.ReflectionPad2d(3)
        self.up_conv = nn.Conv2d(in_dim, out_dim, kernel_size=7, padding=0)
        self.activate = nn.Tanh()

    def forward(self, x, style=None):
        x = self.pad(x)
        x = self.up_conv(x)
        x = self.activate(x)
        return x

class HIST_Generator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=0.0,
        no_antialias=False,
        n_blocks=6,
        res_out_i=1,
        res_in_j=3,
        style_up=True,
        local_encode_mode="trans",
        global_encode_mode="trans",
        mod_features=512,
        drop=0.2,
        att_drop=0.2,
    ):
        assert n_blocks >= 0
        super(HIST_Generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.local_encode_mode = local_encode_mode
        self.global_encode_mode = global_encode_mode
        self.style_up = style_up
        self.n_blocks = n_blocks
        self.res_out_i = res_out_i
        self.res_in_j = res_in_j
        dropout = 0.3 if use_dropout else 0.0
        self.encode_blocks = nn.ModuleList()
        self.encode_blocks.append(
            EncodeBlock(
                3, ngf, no_antialias, norm_layer, use_bias=use_bias, is_first_block=True
            )
        )
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            self.encode_blocks.append(
                EncodeBlock(
                    ngf * mult,
                    ngf * mult * 2,
                    no_antialias,
                    norm_layer,
                    use_bias=use_bias,
                )
            )
        mult = 2**n_downsampling
        self.bottleneck_blocks = nn.ModuleList()
        for i in range(n_blocks):
            if i < res_in_j:
                self.bottleneck_blocks.append(
                    ResBlock(
                        dim=ngf * mult,
                        norm_layer=norm_layer,
                        dropout=dropout,
                        use_bias=use_bias,
                    )
                )
            else:
                inner_style_dim = mod_features
                self.bottleneck_blocks.append(
                    Mod_ResBlock(
                        dim=ngf * mult,
                        style_dim=inner_style_dim,
                        alpha=1.0,
                        demodulate=True,
                        inject_noise=False,
                        norm_layer=norm_layer,
                        dropout=dropout,
                        use_bias=use_bias,
                    )
                )
        self.decode_blocks = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            ii = n_blocks + i
            if style_up:
                self.decode_blocks.append(
                    Mod_DecodeBlock(
                        in_dim=ngf * mult,
                        out_dim=ngf * mult // 2,
                        style_dim=mod_features,
                        alpha=1.0,
                        demodulate=True,
                        inject_noise=False,
                        norm_layer=norm_layer,
                        dropout=dropout,
                    )
                )
            else:
                self.decode_blocks.append(
                    DecodeBlock(
                        in_dim=ngf * mult,
                        out_dim=ngf * mult // 2,
                        norm_layer=norm_layer,
                        dropout=dropout,
                    )
                )
        self.decode_blocks.append(ToRGB_noMod(ngf * mult // 2, output_nc))
        self.num_decode_blocks = len(self.decode_blocks)
        self.num_encode_blocks = len(self.encode_blocks)
        self.num_bottleneck_blocks = len(self.bottleneck_blocks)
        if local_encode_mode == "trans":
            self.local_encoder = Trans_encode_Module(
                input_type="2d",
                in_dim=ngf * 2**n_downsampling,
                out_dim=mod_features,
                n_blocks=3,
                patch_size=4,
                num_heads=4,
                mlp_ratio=3,
                drop=0.1,
                attn_drop=0.1,
                rezero=True,
            )
        elif local_encode_mode == "conv":
            self.local_encoder = Conv_Encode_Module(
                input_type="2d",
                in_dim=ngf * 2**n_downsampling,
                out_dim=mod_features,
                n_blocks=3,
            )
        else:
            raise ValueError(f"Unsupported local_encode_mode: {local_encode_mode}")
        if global_encode_mode == "trans":
            self.global_encoder = Trans_encode_Module(
                input_type="1d",
                in_dim=mod_features,
                out_dim=mod_features,
                n_blocks=2,
                patch_size=4,
                num_heads=4,
                mlp_ratio=3,
                drop=0.1,
                attn_drop=0.1,
                rezero=True,
                pool_type="attention",
            )
        else:
            raise ValueError(f"Unsupported global_encode_mode: {global_encode_mode}")
        style_dim = mod_features
        self.style_mlp = nn.Sequential(
            nn.Linear(mod_features, style_dim * 2),
            nn.ReLU(),
            nn.Linear(style_dim * 2, style_dim),
        )

    def _set_model(self, style_start, k):
        self.style_start = style_start
        self.k = k

    def forward_encode(self, x, blocks=[]):
        feats = []
        for block_id, block in enumerate(self.encode_blocks):
            if block_id in blocks:
                x, feat = block(x, return_feat=True)
                feats.append(feat)
            else:
                x = block(x)
        return x, feats

    def forward_decode(self, x, style, blocks=[]):
        feats = []
        for block_id, block in enumerate(self.decode_blocks):
            block_id = block_id + self.num_encode_blocks + self.num_bottleneck_blocks
            if block_id < self.style_start:
                if block_id in blocks:
                    x, feat = block(x, None, return_feat=True)
                    feats.append(feat)
                else:
                    x = block(x, None)
            else:
                if block_id in blocks:
                    x, feat = block(x, style, return_feat=True)
                    feats.append(feat)
                else:
                    x = block(x, style)
        return x, feats

    def forward_bottle(self, x, blocks=[], k=1, encode_only=False):
        feats_l = []
        for block_id, block in enumerate(self.bottleneck_blocks[: self.res_out_i]):
            fixed_block_id = block_id + self.num_encode_blocks
            if fixed_block_id in blocks:
                x, feat = block(x, return_feat=True)
                feats_l.append(feat)
            else:
                x = block(x)
        feats_i, feats_m = self.local_encoder(x)
        refined_feat_i = rearrange(feats_i, "(b p) c -> b p c", p=k * k)
        if k > 1:
            feat_g, refined_feat_i = self.global_encoder(refined_feat_i)
            refined_feat_i = rearrange(refined_feat_i, "b p c -> (b p) c", p=k * k)
        else:
            refined_feat_i = feats_i
            feat_g = feats_i
        l_style = self.style_mlp(feats_i)
        style = l_style
        if k > 1:
            g_style = self.style_mlp(refined_feat_i)
            B = g_style.shape[0] // 2
            style = torch.cat((g_style[:B], l_style[B:]), dim=0)
        for block_id, block in enumerate(self.bottleneck_blocks[self.res_out_i :]):
            fixed_block_id = block_id + self.num_encode_blocks + self.res_out_i
            if (
                fixed_block_id >= self.style_start
                and (block_id + self.res_out_i) >= self.res_in_j
            ):
                if fixed_block_id in blocks:
                    x, feat = block(x, style, return_feat=True)
                    feats_l.append(feat)
                else:
                    x = block(x, style)
            else:
                if fixed_block_id in blocks:
                    x, feat = block(x, None, return_feat=True)
                    feats_l.append(feat)
                else:
                    x = block(x, None)
        if encode_only:
            return feats_l, feats_m, feats_i, refined_feat_i, feat_g
        else:
            return x, feats_l, feats_m, feats_i, refined_feat_i, feat_g, style
