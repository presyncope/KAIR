import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from timm.layers.weight_init import trunc_normal_
from timm.layers.drop import DropPath
from typing import Literal, Sequence

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNormCF(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, c, h, w)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (b, c, h, w) -> (b, h, w, c)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (b, h, w, c) -> (b, c, h, w)
        x = input + self.drop_path(x)
        return x


class LayerNormCF(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape: int, 
        eps: float = 1e-6,
        data_format: Literal["channels_last", "channels_first"] = "channels_last",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape) == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                assert len(x.shape) == 5
                x = (
                    self.weight[:, None, None, None] * x
                    + self.bias[:, None, None, None]
                )
            return x


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(
        self,
        scale: int,
        num_feat: int,
    ):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1))
        else:
            raise ValueError(f"scale {scale} is not supported. " "Supported scales: 2^n.")
        super().__init__(*m)


class SepConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))

class UpsampleLite(nn.Sequential):
    def __init__(self, scale: int, num_feat: int):
        m = []
        assert (scale & (scale - 1)) == 0
        for _ in range(int(math.log2(scale))):
            m.append(SepConv2d(num_feat, num_feat * 4, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.LeakyReLU(0.1, inplace=True))
        # 마지막 3x3도 DW-sep로
        m.append(SepConv2d(num_feat, num_feat, kernel_size=3, padding=1))
        super().__init__(*m)

class PreNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        fn: nn.Module,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MlpMixer(nn.Module):
    """
    MLP-Mixer 스타일: Token-mixing -> Channel-mixing
    x: (b, tokens, dim) -> (b, tokens, dim)

    - params
        tok_dim: num_tokens
        dim: embedding dim
    """
    def __init__(
        self,
        tok_dim: int,
        dim: int,
        depth: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_mixer = nn.ModuleList()
        self.channel_mixer = nn.ModuleList()

        for _ in range(depth):
            self.token_mixer.append(
                PreNorm(tok_dim, FeedForward(tok_dim, mlp_dim, dropout=dropout))
            )
            self.channel_mixer.append(
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for tok_ff, ch_ff in zip(self.token_mixer, self.channel_mixer):
            x_tok = x.transpose(-1, -2)
            x = tok_ff(x_tok) + x_tok
            x = x.transpose(-1, -2)
            x = ch_ff(x) + x
        return x


class rebotnet(nn.Module):
    '''
    ## commented based on "Supplementary Material for ReBotNet: Fast Real-time Video Enhancement"

    ### Branch I: ConvNeXt
    - Number of Layers = 4 (ok)
    - Depths per layers = depths (ok)
    - Embedding dimensions = depths (ok)

    ### Branch II: MlpMixer
    - Patch size: patch_size (ok)
    - Embedding Dimension: mlp_dim (논문에서는 256이고 써있어서 맞는 값은 mlp_dim이었음, 하지만 실제 token의 embedding dim은 embed_dims[-1] 이고 hidden_dim 이 mlp_dim 이었음)
    - Depth: bottle_depth (ok)

    ### Bottleneck: MlpMixer(?)
    - Input Dimension: embed_dims[-1] (ok)
    - Hidden Dimension: mlp_dim (??? 논문의 728이라는 숫자를 어디서도 찾을수가 없다.) 
    
    ## Net Structure
    - Input: (b, t=2, c=3, h, w)
    
    ### Image token
    - self.big_embedding1
    - self.big_embedding2
    
    ### Tublet token

    '''
    def __init__(
        self,
        upscale: int = 4,
        in_chans: int = 3,
        img_size: Sequence[int] = (272, 480), # (h, w)
        depths: Sequence[int] = (3, 3, 3, 3),
        embed_dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.2,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        bottle_depth: int = 4,
        patch_size: int = 2,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        big_patch = 16
        h, w = img_size

        if len(embed_dims) != 4:
            raise ValueError("len(embed_dims) must be equals to 4")

        if len(depths) != 4:
            raise ValueError("len(depths) must be equals to 4")

        if upscale not in [4]:
            raise ValueError("supported upscale only 4")

        if (h % big_patch != 0) or (w % big_patch != 0):
            raise ValueError(f"img_size {img_size} must be divisible by big_patch {big_patch}.")

        self.in_chans = in_chans
        self.upscale = upscale

        ### Tublet token
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans * 2, embed_dims[0], kernel_size=2, stride=2),
            LayerNormCF(embed_dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            layer = nn.Sequential(
                LayerNormCF(embed_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=embed_dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(4):
            self.norm_layers.append(LayerNormCF(embed_dims[i], eps=1e-6, data_format="channels_first"))

        self.apply(self._init_weights)
        
        ### Image token
        patch_dim_big = 3 * big_patch * big_patch
        num_patches = (h // big_patch) * (w // big_patch)
        self.img_embedding1 = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=big_patch, p2=big_patch),
            nn.Linear(patch_dim_big, embed_dims[-1]),
        )
        self.img_embedding2 = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=big_patch, p2=big_patch),
            nn.Linear(patch_dim_big, embed_dims[-1]),
        )

        ### Bottleneck
        patch_dim = embed_dims[-1] * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, embed_dims[-1]),
        )

        self.pool = nn.MaxPool1d(2, 2)
        self.bottleneck = MlpMixer(num_patches, embed_dims[-1], bottle_depth, mlp_dim, dropout)
        self.temporal_transformer = MlpMixer(num_patches, embed_dims[-1], bottle_depth, mlp_dim, dropout)

        ### Decoder
        self.upsample1 = Upsample(2, embed_dims[-1])
        self.upsample2 = Upsample(2, embed_dims[-2])
        self.upsample3 = Upsample(2, embed_dims[-3])
        self.chchange1 = nn.Conv2d(embed_dims[-1], embed_dims[-2], kernel_size=3, padding=1)
        self.chchange2 = nn.Conv2d(embed_dims[-2], embed_dims[-3], kernel_size=3, padding=1)
        self.chchange3 = nn.Conv2d(embed_dims[-3], embed_dims[-4], kernel_size=3, padding=1)

        self.upsamplef1 = Upsample(2, embed_dims[-4])
        self.upsamplef2 = UpsampleLite(4, embed_dims[-4])
        self.conv_last = nn.Conv2d(embed_dims[-4], 3, kernel_size=3, padding=1)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (b, t = 2, c, h/4, w/4)
        # num_patches = h * w / (256 * 16)

        x_org = x[:, 1, ...].clone()
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        
        ### Image token
        x1 = self.img_embedding1(x[:,0:3, ...])
        x2 = self.img_embedding2(x[:,3:6, ...])
        img_token = torch.cat((x1, x2), dim=1)

        ### Tublet token
        outs: list[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x_out = self.norm_layers[i](x)
            outs.append(x_out)
        x = outs[-1]  # shape: (b, embed_dim, h/16, w/16)

        ### Bottleneck
        b, embed, hlo, wlo = x.shape
        img_token = self.pool(img_token.transpose(1, 2))  # shape: (b, embed_dim, num_patches)
        img_token = self.temporal_transformer(img_token.transpose(1, 2)) # shape: (b, num_patches, embed_dim)

        x = self.to_patch_embedding(x)  # shape: (b, num_patches, embed_dim)
        x = self.bottleneck(x)

        x = x + img_token  # shape: (b, num_patches, embed_dim)
        x = x.transpose(1, 2).contiguous().view(b, embed, hlo, wlo)

        ### Decoder
        x = self.upsample1(x)
        x = self.chchange1(x) + outs[-2]
        x = self.upsample2(x)
        x = self.chchange2(x) + outs[-3]
        x = self.upsample3(x)
        x = self.chchange3(x) + outs[-4] 

        x = self.upsamplef1(x) # shape: (b, ?, h, w)
        x = self.upsamplef2(x) # shape: (b, ?, 4h, 4w)
        x = self.conv_last(x) 
        
        return x + F.interpolate(x_org, scale_factor=4, mode='bilinear', align_corners=False)
