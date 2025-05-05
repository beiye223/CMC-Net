import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import sys
import os
from models.crackmamba2.VSS_Encoder import VSSMEncoder, load_pretrained_ckpt
from models.crackmamba2.myVSS_Encoder import VSSMEncoder as my_VSSMEncoder
from models.crackmamba2.U_Encoder import Encoder
from models.crackmamba2.Decoder import Decoder as U_Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from thop import profile
from thop import clever_format
# from models.convnext.convnext_backbone import ConvNeXt
# from models.RS3mamba.RS3mamba import Decoder as RS3_Decoder
# from models.DECSNet.decoder import decoder as DECS_Decoder

from My_blocks.test_block import BiFusion_block_2d as test_fusion_block
from My_blocks.my_BiFusion import BiFusion_block_2d as my_biFusion


class My_Dual_Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(My_Dual_Model, self).__init__()

        params1 = {'in_chns': in_channels,
                   # 'feature_chns': [46,96, 192, 384, 768],
                   'feature_chns': [32, 64, 128, 256, 512],
                   # 'dropout': [0.05, 0.1, 0.2, 0.3, 0.5], #原模型的参数
                   'dropout': [0.05, 0.05, 0.05, 0.05, 0],
                   'class_num': out_channels,
                   'up_type': 1,
                   'acti_func': 'relu'}

        self.u_encoder = Encoder(params1)                                               # dims=[64, 128, 256, 512]
        # self.vss_encoder = VSSMEncoder(patch_size=4, depths=[2, 2, 9, 2], in_chans=48, dims=[64, 128, 256, 512], use_checkpoint=True)
                                                                                        # dims=[96, 192, 384, 768]
        self.vss_encoder = my_VSSMEncoder(patch_size=4, depths=[2, 2, 9, 2], in_chans=48, dims=[64, 128, 256, 512], use_checkpoint=True)

        self.U_Decoder = U_Decoder(params1)
        # self.RS3_Decoder = RS3_Decoder(encoder_channels=[64, 128, 256, 512])
        # self.Decoder = DECS_Decoder(num_class=1,dims=[64, 128, 256, 512])

        # self.kaiming_normal_init_weight()


        # self.FusionBlock0 =test_fusion_block(64)
        # self.FusionBlock1 =test_fusion_block(128)
        # self.FusionBlock2 =test_fusion_block(256)
        # self.FusionBlock3 =test_fusion_block(512)
        # self.FusionBlock = [self.FusionBlock0, self.FusionBlock1, self.FusionBlock2,
        #                     self.FusionBlock3]

        self.FusionBlock0 = my_biFusion(64)
        self.FusionBlock1 = my_biFusion(128)
        self.FusionBlock2 = my_biFusion(256)
        self.FusionBlock3 = my_biFusion(512)
        self.FusionBlock = [self.FusionBlock0, self.FusionBlock1, self.FusionBlock2,
                            self.FusionBlock3]


    def forward(self, x):

        feature1 = self.u_encoder(x)
        feature2 = self.vss_encoder(x)


        # fuse_feature = [a + b for a, b in zip(feature1, feature2)]
        # fuse_feature = [a for a in feature2]    # 只使用一个分支
        result_feature = [0] * 5


        for i in range(len(feature1)):
            # print(feature1[i].size(), feature2[i].size())
            # result_feature[i] = self.FusionBlock[i](cat_feature[i])
            # result_feature[i] = self.FusionBlock[i](fuse_feature[i])
            # result_feature[i] = self.FusionBlock[i](feature1[i])+ self.FusionBlock[i](feature2[i])
            result_feature[i] = self.FusionBlock[i](feature1[i], feature2[i])


        output = self.U_Decoder(result_feature[0],result_feature[1],result_feature[2],result_feature[3])
        # output = self.U_Decoder(fuse_feature[0], fuse_feature[1], fuse_feature[2], fuse_feature[3])

        # return output
        return torch.sigmoid(output)

    # def kaiming_normal_init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([1, 3, 256, 256]).to(device)
    net = My_Dual_Model(in_channels=3, out_channels=1).to(device)
    # 计算FLOPs和模型参数量
    flops, params = profile(net, inputs=(tensor,))
    flops, params = clever_format([flops, params], "%.3f")

    result = net(tensor)
    print(result)
    print(torch.min(result), torch.max(result))
    print('输入维度{}'.format(tensor.size()))
    print("输出维度{}".format(result.shape))
    print('#parameters:', params)
    print('FLOPs:', flops)