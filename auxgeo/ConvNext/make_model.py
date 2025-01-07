import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
# from torch.nn import init
from auxgeo.Utils import init
import cv2


def SatelliteSlice(rectangle, center=(0, 0), num_fans=8):
    b, c, h, w = rectangle.shape
    center_h, center_w = center
    fans = []
    for i in range(num_fans):
        # --
        angle_start = i * (360 / num_fans)
        angle_end = (i + 1) * (360 / num_fans)
        # --
        fan = np.array([[center_w, center_h],
                        [center_w + w * np.cos(np.radians(angle_start)),
                         center_h + h * np.sin(np.radians(angle_start))],
                        [center_w + w * np.cos(np.radians(angle_end)), center_h + h * np.sin(np.radians(angle_end))]])
        # --
        fans.append(fan)
    masks = []
    for i in fans:
        mask = np.zeros((h, w), dtype=np.float32)
        points = [(center_h, center_w), i[0], i[1]]
        vertices = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        # 使用 fillPoly 填充三角形区域
        cv2.fillPoly(mask, [vertices], 1)
        masks.append(mask)
    return torch.tensor(np.array(masks)).to('cuda')


def StreetSlice(feature_map, parts=8):
    b, c, h, w = feature_map.shape
    part_width = w // parts
    masks = []
    for i in range(parts):
        start_col = i * part_width
        end_col = (i + 1) * part_width if i < parts - 1 else w
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:, start_col:end_col] = 1.0
        masks.append(mask)
    return torch.tensor(np.array(masks)).to('cuda')


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)

    def init_weights(self, init_linear='kaiming'):  # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.deconv(x)
        return x


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='kaiming'):  # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class build_convnext(nn.Module):
    def __init__(self, resnet=False):
        super(build_convnext, self).__init__()
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_base"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=True)

        # -- network of CVGL
        # dim = 512
        # self.norm = nn.LayerNorm(dim, eps=1e-6)

        # -- deconv-1
        # self.deconv_layer1 = DeConv(1024, 512)
        # self.deconv_layer1.init_weights()
        #
        # # -- deconv-2
        # self.deconv_layer2 = DeConv(512, 256)
        # self.deconv_layer2.init_weights()

        # -- MLP
        # self.proj1 = MLP1D(1024, 1024, 512, norm_layer=None, num_mlp=3)  # -- for global feature
        # self.proj1.init_weights()
        # self.proj2 = MLP1D(512, 1024, 512, norm_layer=None, num_mlp=2)  # -- for geometry feature

    def forward(self, x):
        # -- backbone feature extractor
        gap_feature, part_features = self.convnext(x)

        # zip_gap_feature = self.proj1(gap_feature.unsqueeze(2)).squeeze(2)

        # -- Training
        if self.training:

            return gap_feature, part_features

            # -- Eval
        else:
            pass

        return gap_feature, part_features


def make_convnext_model(resnet=False):
    print('===========building convnext===========')
    model = build_convnext(resnet=resnet)
    return model
