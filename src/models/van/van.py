"""Define Visual Attention Network"""
import math
import numpy as np
import mindspore.common.initializer as weight_init

from functools import partial
from mindspore import nn, Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import ReduceMean, Reshape, Transpose, ExpandDims

from src.models.van.misc import Identity, DropPath2D


class DWConv(nn.Cell):
    """DWConv Cell"""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, stride=1, pad_mode="pad", padding=1, group=dim, has_bias=True)

    def construct(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, pad_mode='valid', has_bias=True)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, stride=1, pad_mode='valid', has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0-drop)

        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2.0 / fan_out), mean=0),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def construct(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Cell):
    """LKA Cell"""

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, stride=1, pad_mode="pad", padding=2, group=dim, has_bias=True)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, pad_mode="pad", padding=9, dilation=3, group=dim, has_bias=True)
        self.conv1 = nn.Conv2d(dim, dim, 1, stride=1, pad_mode='valid', has_bias=True)

    def construct(self, x):
        u = x.copy()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Cell):
    """Attention Cell"""

    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1, stride=1, pad_mode='valid', has_bias=True)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1, stride=1, pad_mode='valid', has_bias=True)

    def construct(self, x):
        shorcut = x.copy()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Cell):
    """Block Cell"""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 插眼
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = Parameter(Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32), name="layer_scale_1", requires_grad=True)
        self.layer_scale_2 = Parameter(Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32), name="layer_scale_2", requires_grad=True)
        self.expand_dims = ExpandDims()
        self.reshape = Reshape()

        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2.0 / fan_out), mean=0),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def construct(self, x):
        u0 = x.copy()
        x = self.norm1(x)
        x = self.attn(x)
        x = self.reshape(self.layer_scale_1, (self.dim, 1, 1)) * x
        x = self.drop_path(x)
        x = u0 + x

        u1 = x.copy()
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.reshape(self.layer_scale_2, (self.dim, 1, 1)) * x
        x = self.drop_path(x)
        x = u1 + x

        return x


class OverlapPatchEmbed(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=stride, pad_mode="pad", padding=patch_size//2, has_bias=True)
        self.norm = nn.BatchNorm2d(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2.0 / fan_out), mean=0),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def construct(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Cell):
    """Visual Attention Network"""

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()  # stochastic depth decay rule
        cur = 0

        self.cell_list = nn.CellList()
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.SequentialCell([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])

            norm = norm_layer([embed_dims[i]])
            cur += depths[i]

            self.cell_list.append(patch_embed)
            self.cell_list.append(block)
            self.cell_list.append(norm)

        # classification head
        self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()

        # operations
        self.reshape = Reshape()
        self.transpose = Transpose()
        self.mean = ReduceMean(keep_dims=False)

        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2.0 / fan_out), mean=0),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def construct_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = self.cell_list[i * 3]
            block = self.cell_list[i * 3 + 1]
            norm = self.cell_list[i * 3 + 2]
            x, H, W = patch_embed(x)
            x = block(x)

            x = self.reshape(x, (B, -1, H*W))
            x = self.transpose(x, (0, 2, 1))
            x = norm(x)
            if i != self.num_stages - 1:
                x = self.reshape(x, (B, H, W, -1))
                x = self.transpose(x, (0, 3, 1, 2))

        x = self.mean(x, 1)
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)

        return x


def van_tiny(**kwargs):
    """van_tiny"""

    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    return model


def van_small(**kwargs):
    """van_small"""

    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    return model


def van_base(**kwargs):
    """van_base"""

    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    return model


def van_large(**kwargs):
    """van_base"""

    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    return model
