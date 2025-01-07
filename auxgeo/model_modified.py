import torch.nn as nn
from .ConvNext import make_convnext_model
import torch
import numpy as np
from auxgeo.Utils import init


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


class CVGL_net(nn.Module):
    def __init__(self, resnet=False):
        super(CVGL_net, self).__init__()
        self.model = make_convnext_model(resnet=resnet)

        # 1. temperature factor for contrastive learning
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 2. position prior guided
        self.AP_6x6 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.AP_3x3 = nn.AvgPool2d(kernel_size=4, stride=4)

    def get_config(self):
        input_size = (3, 224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        config = {
            'input_size': input_size,
            'mean': mean,
            'std': std
        }
        return config

    def forward(self, x1, x2=None, x3=None, positions=None):
        if x2 is not None and x3 is not None:
            # --
            y1 = self.model(x1)  # query
            y2 = self.model(x2)  # query_bev
            y3 = self.model(x3)  # ref

            if positions is not None:
                # prior positions [x,y], x -> right, y -> down
                positions_gt_12 = positions * (12 / 384)
                positions_gt_6 = positions * (6 / 384)
                positions_gt_3 = positions * (3 / 384)

                # -- fine-grained position constrain
                global_pano = y1[0]
                global_bev = y2[0]

                part_ref_12 = y3[1]
                part_ref_6 = self.AP_6x6(y3[1])
                part_ref_3 = self.AP_3x3(y3[1])

                bs, c, h, w = part_ref_12.shape

                # 归一化查询向量和参考张量，以便进行余弦相似性计算
                query_vector_pano = global_pano / global_pano.norm(dim=1, keepdim=True)  # (bs, 1024)
                query_vector_bev = global_bev / global_bev.norm(dim=1, keepdim=True)  # (bs, 1024)

                reference_tensor_12 = part_ref_12 / part_ref_12.norm(dim=1, keepdim=True)  # (bs, 1024, 12, 12)

                reference_tensor_6 = part_ref_6 / part_ref_6.norm(dim=1, keepdim=True)  # (bs, 1024, 6, 6)

                reference_tensor_3 = part_ref_3 / part_ref_3.norm(dim=1, keepdim=True)  # (bs, 1024, 3, 3)

                # 扩展查询向量的维度以便进行广播
                query_vector_pano = query_vector_pano.unsqueeze(-1).unsqueeze(-1)  # (bs, 1024, 1, 1)
                query_vector_bev = query_vector_bev.unsqueeze(-1).unsqueeze(-1)  # (bs, 1024, 1, 1)

                # 计算余弦相似性
                similarity_pano_12 = torch.sum(query_vector_pano * reference_tensor_12, dim=1)  # (bs, 12, 12)
                similarity_bev_12 = torch.sum(query_vector_bev * reference_tensor_12, dim=1)  # (bs, 12, 12)

                similarity_pano_6 = torch.sum(query_vector_pano * reference_tensor_6, dim=1)  # (bs, 6, 6)
                similarity_bev_6 = torch.sum(query_vector_bev * reference_tensor_6, dim=1)  # (bs, 6, 6)

                similarity_pano_3 = torch.sum(query_vector_pano * reference_tensor_3, dim=1)  # (bs, 3, 3)
                similarity_bev_3 = torch.sum(query_vector_bev * reference_tensor_3, dim=1)  # (bs, 3, 3)

                # vis_tp1_12 = similarity_pano_12[0].detach().cpu().numpy()
                # vis_tp1_6 = similarity_pano_6[0].detach().cpu().numpy()
                # vis_tp1_3 = similarity_pano_3[0].detach().cpu().numpy()

                # 使用一个非常低温度的softmax来近似最大值选择
                temperature = 5e-2
                softmax_similarity_pano_12 = torch.nn.functional.softmax(similarity_pano_12.view(bs, -1) / temperature,
                                                                         dim=1).view(bs, 12, 12)
                softmax_similarity_bev_12 = torch.nn.functional.softmax(similarity_bev_12.view(bs, -1) / temperature,
                                                                        dim=1).view(bs, 12, 12)

                softmax_similarity_pano_6 = torch.nn.functional.softmax(similarity_pano_6.view(bs, -1) / temperature,
                                                                        dim=1).view(bs, 6, 6)
                softmax_similarity_bev_6 = torch.nn.functional.softmax(similarity_bev_6.view(bs, -1) / temperature,
                                                                       dim=1).view(bs, 6, 6)

                softmax_similarity_pano_3 = torch.nn.functional.softmax(similarity_pano_3.view(bs, -1) / temperature,
                                                                        dim=1).view(bs, 3, 3)
                softmax_similarity_bev_3 = torch.nn.functional.softmax(similarity_bev_3.view(bs, -1) / temperature,
                                                                       dim=1).view(bs, 3, 3)

                # vis_tp2_12 = similarity_pano_12[0].detach().cpu().numpy()
                # vis_tp2_6 = similarity_pano_6[0].detach().cpu().numpy()
                # vis_tp2_3 = similarity_pano_3[0].detach().cpu().numpy()


                # 生成坐标网格
                x_coords_pano_12 = torch.arange(12, device='cuda').float().view(1, 1, 12).expand(bs, 12, 12)
                y_coords_pano_12 = torch.arange(12, device='cuda').float().view(1, 12, 1).expand(bs, 12, 12)
                x_coords_bev_12 = torch.arange(12, device='cuda').float().view(1, 1, 12).expand(bs, 12, 12)
                y_coords_bev_12 = torch.arange(12, device='cuda').float().view(1, 12, 1).expand(bs, 12, 12)

                x_coords_pano_6 = torch.arange(6, device='cuda').float().view(1, 1, 6).expand(bs, 6, 6)
                y_coords_pano_6 = torch.arange(6, device='cuda').float().view(1, 6, 1).expand(bs, 6, 6)
                x_coords_bev_6 = torch.arange(6, device='cuda').float().view(1, 1, 6).expand(bs, 6, 6)
                y_coords_bev_6 = torch.arange(6, device='cuda').float().view(1, 6, 1).expand(bs, 6, 6)

                x_coords_pano_3 = torch.arange(3, device='cuda').float().view(1, 1, 3).expand(bs, 3, 3)
                y_coords_pano_3 = torch.arange(3, device='cuda').float().view(1, 3, 1).expand(bs, 3, 3)
                x_coords_bev_3 = torch.arange(3, device='cuda').float().view(1, 1, 3).expand(bs, 3, 3)
                y_coords_bev_3 = torch.arange(3, device='cuda').float().view(1, 3, 1).expand(bs, 3, 3)

                # 计算预测位置
                x_pred_pano_12 = torch.sum(x_coords_pano_12 * softmax_similarity_pano_12, dim=(1, 2))
                y_pred_pano_12 = torch.sum(y_coords_pano_12 * softmax_similarity_pano_12, dim=(1, 2))
                x_pred_bev_12 = torch.sum(x_coords_bev_12 * softmax_similarity_bev_12, dim=(1, 2))
                y_pred_bev_12 = torch.sum(y_coords_bev_12 * softmax_similarity_bev_12, dim=(1, 2))

                x_pred_pano_6 = torch.sum(x_coords_pano_6 * softmax_similarity_pano_6, dim=(1, 2))
                y_pred_pano_6 = torch.sum(y_coords_pano_6 * softmax_similarity_pano_6, dim=(1, 2))
                x_pred_bev_6 = torch.sum(x_coords_bev_6 * softmax_similarity_bev_6, dim=(1, 2))
                y_pred_bev_6 = torch.sum(y_coords_bev_6 * softmax_similarity_bev_6, dim=(1, 2))

                x_pred_pano_3 = torch.sum(x_coords_pano_3 * softmax_similarity_pano_3, dim=(1, 2))
                y_pred_pano_3 = torch.sum(y_coords_pano_3 * softmax_similarity_pano_3, dim=(1, 2))
                x_pred_bev_3 = torch.sum(x_coords_bev_3 * softmax_similarity_bev_3, dim=(1, 2))
                y_pred_bev_3 = torch.sum(y_coords_bev_3 * softmax_similarity_bev_3, dim=(1, 2))

                #
                positions_pred_pano_12 = torch.stack([x_pred_pano_12, y_pred_pano_12], dim=1)  # (bs, 2)
                positions_pred_bev_12 = torch.stack([x_pred_bev_12, y_pred_bev_12], dim=1)  # (bs, 2)

                positions_pred_pano_6 = torch.stack([x_pred_pano_6, y_pred_pano_6], dim=1)  # (bs, 2)
                positions_pred_bev_6 = torch.stack([x_pred_bev_6, y_pred_bev_6], dim=1)  # (bs, 2)

                positions_pred_pano_3 = torch.stack([x_pred_pano_3, y_pred_pano_3], dim=1)  # (bs, 2)
                positions_pred_bev_3 = torch.stack([x_pred_bev_3, y_pred_bev_3], dim=1)  # (bs, 2)

                # 计算欧氏距离损失
                pos_loss_12 = torch.sqrt(
                    torch.sum((positions_gt_12 - positions_pred_pano_12) ** 2, dim=1)).mean() + torch.sqrt(
                    torch.sum((positions_gt_12 - positions_pred_bev_12) ** 2, dim=1)).mean()
                pos_loss_6 = torch.sqrt(
                    torch.sum((positions_gt_6 - positions_pred_pano_6) ** 2, dim=1)).mean() + torch.sqrt(
                    torch.sum((positions_gt_6 - positions_pred_bev_6) ** 2, dim=1)).mean()
                pos_loss_3 = torch.sqrt(
                    torch.sum((positions_gt_3 - positions_pred_pano_3) ** 2, dim=1)).mean() + torch.sqrt(
                    torch.sum((positions_gt_3 - positions_pred_bev_3) ** 2, dim=1)).mean()

                pos_loss = (pos_loss_12 + pos_loss_6 + pos_loss_3)

                return y1, y2, y3, pos_loss

            return y1, y2, y3

        elif x2 is not None:
            y1 = self.model(x1)  # query
            y2 = self.model(x2)  # ref

            return y1, y2

        else:
            # --
            y1 = self.model(x1)
            return y1



def make_model(opt):
    model = CVGL_net(resnet=False)
    return model
