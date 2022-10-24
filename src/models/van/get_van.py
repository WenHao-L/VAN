# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Get Visual Attention Network of different size for args"""
from .van import VAN


def get_van(args):
    """get van according to args"""
    # override args
    image_size = args.image_size
    in_chans = args.in_channel
    num_classes = args.num_classes
    embed_dims = args.embed_dims
    mlp_ratios = args.mlp_ratios
    drop_path_rate = args.drop_path_rate
    depths = args.depths
    num_stages = args.num_stages
    print(25 * "=" + "MODEL CONFIG" + 25 * "=")
    print(f"==> IMAGE_SIZE:         {image_size}")
    print(f"==> IN_CHANS:           {in_chans}")
    print(f"==> NUM_CLASSES:        {num_classes}")
    print(f"==> EMBED_DIMS:         {embed_dims}")
    print(f"==> MLP_RATIOS:         {mlp_ratios}")
    print(f"==> DROP_PATH_RATE:     {drop_path_rate}")
    print(f"==> DEPTHS:             {depths}")
    print(f"==> NUM_STAGES:         {num_stages}")
    print(25 * "=" + "FINISHED" + 25 * "=")

    model = VAN(
        img_size=image_size, in_chans=in_chans, num_classes=num_classes,
        embed_dims=embed_dims, mlp_ratios=mlp_ratios, drop_path_rate=drop_path_rate,
        depths=depths, num_stages=num_stages)

    return model


def van_tiny_224(args):
    """van tiny 224"""
    return get_van(args)


def van_small_224(args):
    """van small 224"""""
    return get_van(args)


def van_base_224(args):
    """van base 224"""
    return get_van(args)


def van_large_224(args):
    """van large 224"""
    return get_van(args)
