import torch
import torch.nn as nn
import numpy as np

class FourierPositionEmbed(nn.Module):
    def __init__(self, embed_dim, input_dim):
        super().__init__()
        self.register_buffer("projector", torch.randn(2, embed_dim) * 0.5)
        self.register_buffer("pe_cache", torch.zeros(1, 1, embed_dim), persistent=False)
        self.fusion = nn.Linear(input_dim + embed_dim, input_dim)
        self.is_set_grid = False

    def set_grid_shape(self, height, width):
        device = self.projector.device
        h_range = torch.linspace(-1, 1, steps=height, device=device)
        w_range = torch.linspace(-1, 1, steps=width, device=device)
        grid_y, grid_x = torch.meshgrid(h_range, w_range, indexing="ij")
        L = height * width
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, L, 2)
        embed = torch.sin(grid @ self.projector)
        self.pe_cache = embed
        self.is_set_grid = True

    def forward(self, x):
        if not self.is_set_grid:
            print(x.shape)
            B, L, C = x.shape
            side = int(np.sqrt(L))
            if side * side != L:
                raise ValueError(
                    f"Input sequence length {L} is not a perfect square. "
                    f"Cannot auto-infer (H, W). Please call `set_grid_shape(h, w)` manually."
                )
            self.set_grid_shape(side, side)
        combined = torch.cat([x, self.pe_cache.expand(x.shape[0], -1, -1)], dim=-1)
        return self.fusion(combined)

class SinePositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, temperature=10000):
        super().__init__()
        self.d_model = embed_dim
        self.temperature = temperature
        self.register_buffer("pe_cache", torch.zeros(1, 1, embed_dim), persistent=False)
        self.is_set_grid = False

    def set_grid_shape(self, h, w):
        device = self.pe_cache.device
        y_pos = torch.arange(h, dtype=torch.float32, device=device)
        x_pos = torch.arange(w, dtype=torch.float32, device=device)
        dim_h = self.d_model // 2
        dim_w = self.d_model - dim_h
        div_term_h = self.temperature ** (torch.arange(0, dim_h, 2).float() / dim_h).to(
            device
        )
        div_term_w = self.temperature ** (torch.arange(0, dim_w, 2).float() / dim_w).to(
            device
        )
        pe_y = torch.zeros(h, dim_h, device=device)
        pe_y[:, 0::2] = torch.sin(y_pos.unsqueeze(1) / div_term_h)
        pe_y[:, 1::2] = torch.cos(y_pos.unsqueeze(1) / div_term_h)
        pe_x = torch.zeros(w, dim_w, device=device)
        pe_x[:, 0::2] = torch.sin(x_pos.unsqueeze(1) / div_term_w)
        pe_x[:, 1::2] = torch.cos(x_pos.unsqueeze(1) / div_term_w)
        pe_y = pe_y.unsqueeze(1).repeat(1, w, 1)
        pe_x = pe_x.unsqueeze(0).repeat(h, 1, 1)
        pe = torch.cat([pe_x, pe_y], dim=-1)
        self.pe_cache = pe.flatten(0, 1).unsqueeze(0)
        self.is_set_grid = True

    def forward(self, x):
        if not self.is_set_grid:
            B, N, C = x.shape
            k = int(np.sqrt(N))
            self.set_grid_shape(k, k)
        return x + self.pe_cache

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3.0,
        act_layer=nn.GELU,
        rezero=True,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        self.rezero = rezero
        if rezero:
            self.alpha1 = nn.Parameter(torch.zeros(1))
            self.alpha2 = nn.Parameter(torch.zeros(1))
        else:
            self.alpha1 = 1.0
            self.alpha2 = 1.0

    def forward(self, x, return_attn=False):
        res = x
        x = self.norm1(x)
        if return_attn:
            x, attn = self.attn(x, return_attn=True)
        else:
            x = self.attn(x, return_attn=False)
        x = res + self.alpha1 * x
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + self.alpha2 * x
        if return_attn:
            return x, attn
        else:
            return x

class Trans_encode_Module(nn.Module):
    def __init__(
        self,
        input_type,
        in_dim,
        out_dim,
        n_blocks,
        patch_size=4,
        num_heads=4,
        mlp_ratio=3.0,
        drop=0.0,
        attn_drop=0.0,
        rezero=True,
    ):
        super().__init__()
        self.use_patch_embed = input_type == "2d"
        self.out_dim = out_dim
        if self.use_patch_embed:
            self.patch_embed = nn.Conv2d(
                in_dim, out_dim, kernel_size=patch_size, stride=patch_size
            )
        if self.use_patch_embed:
            self.pos_embed = FourierPositionEmbed(embed_dim=out_dim, input_dim=out_dim)
        else:
            self.pos_embed = SinePositionalEncoding2D(embed_dim=out_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    rezero=rezero,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        if self.use_patch_embed:
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        attn_list = []
        for blk in self.blocks:
            if return_attn:
                x, attn = blk(x, return_attn=True)
                attn_list.append(attn)
            else:
                x = blk(x)
        cls_token = x[:, 0, :]
        seq_token = x[:, 1:, :]
        if not return_attn:
            return cls_token, seq_token

class Trans_encode_Module(nn.Module):
    def __init__(
        self,
        input_type,
        in_dim,
        out_dim,
        n_blocks,
        patch_size=4,
        num_heads=4,
        mlp_ratio=3.0,
        drop=0.0,
        attn_drop=0.0,
        rezero=True,
        pool_type="attention",
    ):
        super().__init__()
        self.use_patch_embed = input_type == "2d"
        self.out_dim = out_dim
        self.pool_type = pool_type
        if self.use_patch_embed:
            self.patch_embed = nn.Conv2d(
                in_dim, out_dim, kernel_size=patch_size, stride=patch_size
            )
        if self.use_patch_embed:
            self.pos_embed = FourierPositionEmbed(embed_dim=out_dim, input_dim=out_dim)
        else:
            self.pos_embed = SinePositionalEncoding2D(embed_dim=out_dim)
        if self.pool_type == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, out_dim))
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=out_dim, num_heads=num_heads, batch_first=True
            )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    rezero=rezero,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        if self.use_patch_embed:
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_embed(x)
        attn_list = []
        for blk in self.blocks:
            if return_attn:
                x, attn = blk(x, return_attn=True)
                attn_list.append(attn)
            else:
                x = blk(x)
        seq_token = self.norm(x)
        if self.pool_type == "gap":
            global_feat = seq_token.mean(dim=1)
        elif self.pool_type == "attention":
            q = self.pool_query.expand(B, -1, -1)
            global_feat, _ = self.pool_attn(q, seq_token, seq_token)
            global_feat = global_feat.squeeze(1)
        else:
            global_feat = seq_token[:, 0, :]
        if not return_attn:
            return global_feat, seq_token
        return global_feat, seq_token, attn_list
