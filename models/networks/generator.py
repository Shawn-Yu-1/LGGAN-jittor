"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
import jittor.nn as nn
# import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

def if_all_zero(tensor):
    if jt.sum(tensor) == 0:
        return 0
    else:
        return 1

class LGGANGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadebatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf  # of gen filters in first conv layer

        self.sw, self.sh = self.compute_latent_vector_size(opt)
        # print(self.sw, self.sh) 8, 4

        if opt.use_vae:  # False
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)  # print(self.opt.semantic_nc) # 36

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':  # opt.num_upsampling_layers: more
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        # local branch
        self.conv1 = nn.Conv2d(29, 64, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv4_norm = nn.InstanceNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.conv5_norm = nn.InstanceNorm2d(1024)

        self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks1.weight_init(0, 0.02)
        self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks2.weight_init(0, 0.02)
        self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks3.weight_init(0, 0.02)
        self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks4.weight_init(0, 0.02)
        self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks5.weight_init(0, 0.02)
        self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks6.weight_init(0, 0.02)
        self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks7.weight_init(0, 0.02)
        self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks8.weight_init(0, 0.02)
        self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks9.weight_init(0, 0.02)

        # self.resnet_blocks10_0 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_0.weight_init(0, 0.02)
        # self.resnet_blocks10_1 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_1.weight_init(0, 0.02)
        # self.resnet_blocks10_2 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_2.weight_init(0, 0.02)
        # self.resnet_blocks10_3 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_3.weight_init(0, 0.02)
        # self.resnet_blocks10_4 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_4.weight_init(0, 0.02)
        # self.resnet_blocks10_5 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_5.weight_init(0, 0.02)
        # self.resnet_blocks10_6 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_6.weight_init(0, 0.02)
        # self.resnet_blocks10_7 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_7.weight_init(0, 0.02)
        # self.resnet_blocks10_8 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_8.weight_init(0, 0.02)
        # self.resnet_blocks10_9 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_9.weight_init(0, 0.02)
        # self.resnet_blocks10_10 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_10.weight_init(0, 0.02)
        # self.resnet_blocks10_11 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_11.weight_init(0, 0.02)
        # self.resnet_blocks10_12 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_12.weight_init(0, 0.02)
        # self.resnet_blocks10_13 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_13.weight_init(0, 0.02)
        # self.resnet_blocks10_14 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_14.weight_init(0, 0.02)
        # self.resnet_blocks10_15 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_15.weight_init(0, 0.02)
        # self.resnet_blocks10_16 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_16.weight_init(0, 0.02)
        # self.resnet_blocks10_17 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_17.weight_init(0, 0.02)
        # self.resnet_blocks10_18 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_18.weight_init(0, 0.02)
        # self.resnet_blocks10_19 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_19.weight_init(0, 0.02)
        # self.resnet_blocks10_20 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_20.weight_init(0, 0.02)
        # self.resnet_blocks10_21 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_21.weight_init(0, 0.02)
        # self.resnet_blocks10_22 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_22.weight_init(0, 0.02)
        # self.resnet_blocks10_23 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_23.weight_init(0, 0.02)
        # self.resnet_blocks10_24 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_24.weight_init(0, 0.02)
        # self.resnet_blocks10_25 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_25.weight_init(0, 0.02)
        # self.resnet_blocks10_26 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_26.weight_init(0, 0.02)
        # self.resnet_blocks10_27 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_27.weight_init(0, 0.02)
        # self.resnet_blocks10_28 = resnet_block_2(64,3,1,1)
        # self.resnet_blocks10_28.weight_init(0, 0.02)

        self.deconv3_local = nn.ConvTranspose(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_local = nn.InstanceNorm2d(128)
        self.deconv4_local = nn.ConvTranspose(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_local = nn.InstanceNorm2d(64)

        self.deconv9 = nn.Conv2d(3*29, 3, 3, 1, 1)

        self.deconv5_0 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_1 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_2 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_3 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_4 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_5 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_6 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_7 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_8 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_9 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_10 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_11 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_12 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_13 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_14 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_15 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_16 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_17 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_18 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_19 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_20 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_21 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_22 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_23 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_24 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_25 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_26 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_27 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_28 = nn.Conv2d(64, 3, 7, 1, 0)
        # self.deconv5_29 = nn.Conv2d(64, 3, 7, 1, 0)
        # self.deconv5_30 = nn.Conv2d(64, 3, 7, 1, 0)
        # self.deconv5_31 = nn.Conv2d(64, 3, 7, 1, 0)
        # self.deconv5_32 = nn.Conv2d(64, 3, 7, 1, 0)
        # self.deconv5_33 = nn.Conv2d(64, 3, 7, 1, 0)
        # self.deconv5_34 = nn.Conv2d(64, 3, 7, 1, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(64*256 * 512, 512)
        self.fc2 = nn.Linear(64, 29)

        self.deconv3_attention = nn.ConvTranspose(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_attention = nn.InstanceNorm2d(128)
        self.deconv4_attention = nn.ConvTranspose(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_attention = nn.InstanceNorm2d(64)
        self.deconv5_attention = nn.Conv2d(64, 2, 1, 1, 0)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = opt.crop_h // (2 ** num_up_layers)

        return sw, sh

    def execute(self, input, z=None):
        # global branch
        seg = input  # print(input.size()) # [1, 36, 256, 512]

        if self.opt.use_vae:  # use_vae: False
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = jt.randn(input.size(0), self.opt.z_dim,
                                dtype=jt.float32)
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            # x = F.interpolate(seg, size=(self.sh, self.sw))  # print(x.size()) [1, 36, 4, 8]
            # x = self.fc(x)  # print(x.size()) [1, 1024, 4, 8]

            x = nn.pad(seg, (3, 3, 3, 3), 'reflect')  # print(x.size()) [1, 3, 262, 518]
            x = nn.relu(self.conv1_norm(self.conv1(x)))  # print(x.size()) [1, 64, 256, 512]
            x = nn.relu(self.conv2_norm(self.conv2(x)))  # print(x.size()) [1, 128, 128, 256]
            x_encode = nn.relu(self.conv3_norm(self.conv3(x)))  # print(x.size()) [1, 256, 64, 128]
            x = nn.relu(self.conv4_norm(self.conv4(x_encode)))  # print(x.size()) [1, 512, 32, 64]
            x = nn.relu(self.conv5_norm(self.conv5(x)))  # print(x.size()) [1, 1024, 16, 32]
            x = nn.interpolate(x, size=(self.sh, self.sw))  # print(x.size()) [1, 1024, 4, 8]

        x = self.head_0(x, seg)  # print(x.size()) [1, 1024, 4, 8] seg [1, 36, 256, 512]

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':  # num_upsampling_layers: more
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(nn.leaky_relu(x, 2e-1))
        result_global = jt.tanh(x)

        ############################## local branch ##############################
        label = input[:, 0: self.opt.label_nc, :, :]  # print(label.size()) [1, 35, 256, 512]
        for i in range(self.opt.label_nc):
            globals()['label_' + str(i)] = label[:, i:i + 1, :, :]
            globals()['label_3_' + str(i)] = label[:, i:i + 1, :, :].repeat(1, 3, 1, 1)
            globals()['label_64_' + str(i)] = label[:, i:i + 1, :, :].repeat(1, 64, 1, 1)



        x = self.resnet_blocks1(x_encode) # print(x.size()) [1 ,256, 64, 128]
        x = self.resnet_blocks2(x)
        x = self.resnet_blocks3(x)
        x = self.resnet_blocks4(x)
        x = self.resnet_blocks5(x)
        x = self.resnet_blocks6(x)
        x = self.resnet_blocks7(x)
        x = self.resnet_blocks8(x) # print(x.size()) [1, 256, 64, 128]
        middle_x = self.resnet_blocks9(x) # print(middle_x.size()) [1, 256, 64, 128]
        x_local = nn.relu(self.deconv3_norm_local(self.deconv3_local(middle_x))) # print(x_local.size()) [1, 128, 128, 256]
        x_feature_local = nn.relu(self.deconv4_norm_local(self.deconv4_local(x_local))) # print(x_feature_local.size()) [1, 64, 256, 512]

        # for i in range(self.opt.label_nc):
        #     globals()['feature_' + str(i)] = x_feature_local * eval('label_64_%d'% (i))

        feature_0 = x_feature_local * label_64_0
        feature_1 = x_feature_local * label_64_1
        feature_2 = x_feature_local * label_64_2
        feature_3 = x_feature_local * label_64_3
        feature_4 = x_feature_local * label_64_4
        feature_5 = x_feature_local * label_64_5
        feature_6 = x_feature_local * label_64_6
        feature_7 = x_feature_local * label_64_7
        feature_8 = x_feature_local * label_64_8
        feature_9 = x_feature_local * label_64_9
        feature_10 = x_feature_local * label_64_10
        feature_11 = x_feature_local * label_64_11
        feature_12 = x_feature_local * label_64_12
        feature_13 = x_feature_local * label_64_13
        feature_14 = x_feature_local * label_64_14
        feature_15 = x_feature_local * label_64_15
        feature_16 = x_feature_local * label_64_16
        feature_17 = x_feature_local * label_64_17
        feature_18 = x_feature_local * label_64_18
        feature_19 = x_feature_local * label_64_19
        feature_20 = x_feature_local * label_64_20
        feature_21 = x_feature_local * label_64_21
        feature_22 = x_feature_local * label_64_22
        feature_23 = x_feature_local * label_64_23
        feature_24 = x_feature_local * label_64_24
        feature_25 = x_feature_local * label_64_25
        feature_26 = x_feature_local * label_64_26
        feature_27 = x_feature_local * label_64_27
        feature_28 = x_feature_local * label_64_28
        # feature_29 = x_feature_local * label_64_29
        # feature_30 = x_feature_local * label_64_30
        # feature_31 = x_feature_local * label_64_31
        # feature_32 = x_feature_local * label_64_32
        # feature_33 = x_feature_local * label_64_33
        # feature_34 = x_feature_local * label_64_34
        # print('before feature:', feature_0.size())

        # feature_0 = self.resnet_blocks10_0(feature_0)
        # feature_1 = self.resnet_blocks10_1(feature_1)
        # feature_2 = self.resnet_blocks10_2(feature_2)
        # feature_3 = self.resnet_blocks10_3(feature_3)
        # feature_4 = self.resnet_blocks10_4(feature_4)
        # feature_5 = self.resnet_blocks10_5(feature_5)
        # feature_6 = self.resnet_blocks10_6(feature_6)
        # feature_7 = self.resnet_blocks10_7(feature_7)
        # feature_8 = self.resnet_blocks10_8(feature_8)
        # feature_9 = self.resnet_blocks10_9(feature_9)
        # feature_10 = self.resnet_blocks10_10(feature_10)
        # feature_11 = self.resnet_blocks10_11(feature_11)
        # feature_12 = self.resnet_blocks10_12(feature_12)
        # feature_13 = self.resnet_blocks10_13(feature_13)
        # feature_14 = self.resnet_blocks10_14(feature_14)
        # feature_15 = self.resnet_blocks10_15(feature_15)
        # feature_16 = self.resnet_blocks10_16(feature_16)
        # feature_17 = self.resnet_blocks10_17(feature_17)
        # feature_18 = self.resnet_blocks10_18(feature_18)
        # feature_19 = self.resnet_blocks10_19(feature_19)
        # feature_20 = self.resnet_blocks10_20(feature_20)
        # feature_21 = self.resnet_blocks10_21(feature_21)
        # feature_22 = self.resnet_blocks10_22(feature_22)
        # feature_23 = self.resnet_blocks10_23(feature_23)
        # feature_24 = self.resnet_blocks10_24(feature_24)
        # feature_25 = self.resnet_blocks10_25(feature_25)
        # feature_26 = self.resnet_blocks10_26(feature_26)
        # feature_27 = self.resnet_blocks10_27(feature_27)
        # feature_28 = self.resnet_blocks10_28(feature_28)

        # print('after feature:', feature_0.size())
        # [1, 64, 256, 512]
        feature_combine= jt.concat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11,
                                    feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19,feature_20,feature_21,
                                    feature_22,feature_23, feature_24, feature_25, feature_26,feature_27,feature_28), 0)
        # print(feature_combine.size())
        # [35, 64, 256, 512]
        feature_combine = self.avgpool(feature_combine)
        # print(feature_combine.size())
        # [35, 64 , 1, 1]
        feature_combine_fc = jt.flatten(feature_combine, 1)
        # print(feature_combine_fc.size())
        # [35, 64]
        feature_score = self.fc2(feature_combine_fc)
        # print(feature_score.size()) [35, 35]
        target= jt.float32([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28])
        # print(target)
        # print(label_0)
        valid_index = jt.float32([if_all_zero(label_0), if_all_zero(label_1), if_all_zero(label_2),if_all_zero(label_3),if_all_zero(label_4), if_all_zero(label_5),
                                    if_all_zero(label_6), if_all_zero(label_7), if_all_zero(label_8),if_all_zero(label_9),if_all_zero(label_10),if_all_zero(label_11),
                                    if_all_zero(label_12),if_all_zero(label_13),if_all_zero(label_14),if_all_zero(label_15),if_all_zero(label_16),if_all_zero(label_17),
                                    if_all_zero(label_18),if_all_zero(label_19),if_all_zero(label_20),if_all_zero(label_21),if_all_zero(label_22),if_all_zero(label_23),
                                    if_all_zero(label_24),if_all_zero(label_25),if_all_zero(label_26),if_all_zero(label_27),if_all_zero(label_28)])
        # print(valid_index)


        # for i in range(self.opt.label_nc):
        #     globals()['feature_' + str(i)] = nn.pad(eval('feature_%d'% (i)), (3, 3, 3, 3), 'reflect') # print(label_1.size())  [1, 64, 262, 518]

        feature_0 = nn.pad(feature_0, (3, 3, 3, 3), 'reflect')
        feature_1 = nn.pad(feature_1, (3, 3, 3, 3), 'reflect')
        feature_2 = nn.pad(feature_2, (3, 3, 3, 3), 'reflect')
        feature_3 = nn.pad(feature_3, (3, 3, 3, 3), 'reflect')
        feature_4 = nn.pad(feature_4, (3, 3, 3, 3), 'reflect')
        feature_5 = nn.pad(feature_5, (3, 3, 3, 3), 'reflect')
        feature_6 = nn.pad(feature_6, (3, 3, 3, 3), 'reflect')
        feature_7 = nn.pad(feature_7, (3, 3, 3, 3), 'reflect')
        feature_8 = nn.pad(feature_8, (3, 3, 3, 3), 'reflect')
        feature_9 = nn.pad(feature_9, (3, 3, 3, 3), 'reflect')
        feature_10 = nn.pad(feature_10, (3, 3, 3, 3), 'reflect')
        feature_11 = nn.pad(feature_11, (3, 3, 3, 3), 'reflect')
        feature_12 = nn.pad(feature_12, (3, 3, 3, 3), 'reflect')
        feature_13 = nn.pad(feature_13, (3, 3, 3, 3), 'reflect')
        feature_14 = nn.pad(feature_14, (3, 3, 3, 3), 'reflect')
        feature_15 = nn.pad(feature_15, (3, 3, 3, 3), 'reflect')
        feature_16 = nn.pad(feature_16, (3, 3, 3, 3), 'reflect')
        feature_17 = nn.pad(feature_17, (3, 3, 3, 3), 'reflect')
        feature_18 = nn.pad(feature_18, (3, 3, 3, 3), 'reflect')
        feature_19 = nn.pad(feature_19, (3, 3, 3, 3), 'reflect')
        feature_20 = nn.pad(feature_20, (3, 3, 3, 3), 'reflect')
        feature_21 = nn.pad(feature_21, (3, 3, 3, 3), 'reflect')
        feature_22 = nn.pad(feature_22, (3, 3, 3, 3), 'reflect')
        feature_23 = nn.pad(feature_23, (3, 3, 3, 3), 'reflect')
        feature_24 = nn.pad(feature_24, (3, 3, 3, 3), 'reflect')
        feature_25 = nn.pad(feature_25, (3, 3, 3, 3), 'reflect')
        feature_26 = nn.pad(feature_26, (3, 3, 3, 3), 'reflect')
        feature_27 = nn.pad(feature_27, (3, 3, 3, 3), 'reflect')
        feature_28 = nn.pad(feature_28, (3, 3, 3, 3), 'reflect')
        # feature_29 = nn.pad(feature_29, (3, 3, 3, 3), 'reflect')
        # feature_30 = nn.pad(feature_30, (3, 3, 3, 3), 'reflect')
        # feature_31 = nn.pad(feature_31, (3, 3, 3, 3), 'reflect')
        # feature_32 = nn.pad(feature_32, (3, 3, 3, 3), 'reflect')
        # feature_33 = nn.pad(feature_33, (3, 3, 3, 3), 'reflect')
        # feature_34 = nn.pad(feature_34, (3, 3, 3, 3), 'reflect')


        result_0 = jt.tanh(self.deconv5_0(feature_0)) # print(result_1.size()) [1, 3, 256, 512]
        # print(result_0.size())
        result_1 = jt.tanh(self.deconv5_1(feature_1))
        result_2 = jt.tanh(self.deconv5_2(feature_2))
        result_3 = jt.tanh(self.deconv5_3(feature_3))
        result_4 = jt.tanh(self.deconv5_4(feature_4))
        result_5 = jt.tanh(self.deconv5_5(feature_5))
        result_6 = jt.tanh(self.deconv5_6(feature_6))
        result_7 = jt.tanh(self.deconv5_7(feature_7))
        result_8 = jt.tanh(self.deconv5_8(feature_8))
        result_9 = jt.tanh(self.deconv5_9(feature_9))
        result_10 = jt.tanh(self.deconv5_10(feature_10))
        result_11 = jt.tanh(self.deconv5_11(feature_11))
        result_12 = jt.tanh(self.deconv5_12(feature_12))
        result_13 = jt.tanh(self.deconv5_13(feature_13))
        result_14 = jt.tanh(self.deconv5_14(feature_14))
        result_15 = jt.tanh(self.deconv5_15(feature_15))
        result_16 = jt.tanh(self.deconv5_16(feature_16))
        result_17 = jt.tanh(self.deconv5_17(feature_17))
        result_18 = jt.tanh(self.deconv5_18(feature_18))
        result_19 = jt.tanh(self.deconv5_19(feature_19))
        result_20 = jt.tanh(self.deconv5_20(feature_20))
        result_21 = jt.tanh(self.deconv5_21(feature_21))
        result_22 = jt.tanh(self.deconv5_22(feature_22))
        result_23 = jt.tanh(self.deconv5_23(feature_23))
        result_24 = jt.tanh(self.deconv5_24(feature_24))
        result_25 = jt.tanh(self.deconv5_25(feature_25))
        result_26 = jt.tanh(self.deconv5_26(feature_26))
        result_27 = jt.tanh(self.deconv5_27(feature_27))
        result_28 = jt.tanh(self.deconv5_28(feature_28))
        # result_29 = jt.tanh(self.deconv5_29(feature_29))
        # result_30 = jt.tanh(self.deconv5_30(feature_30))
        # result_31 = jt.tanh(self.deconv5_31(feature_31))
        # result_32 = jt.tanh(self.deconv5_32(feature_32))
        # result_33 = jt.tanh(self.deconv5_33(feature_33))
        # result_34 = jt.tanh(self.deconv5_34(feature_34))


        # result_local = result_0 + result_1 + result_2 + result_3 + result_4 + result_5 + result_6 + result_7 + result_8 + result_9 + result_10 + \
        #         result_11 + result_12 + result_13 + result_14 + result_15 + result_16 + result_17 + result_18 + result_19 + result_20 + \
        #         result_21 + result_22 + result_23 + result_24 + result_25 + result_26 + result_27 + result_28 + result_29 + result_30 + \
        #                result_31 + result_32 + result_33 + result_34

        combine_local = jt.concat((result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
                                   result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
                                   result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 ), 1)
        result_local = jt.tanh(self.deconv9(combine_local))

        # print(result_local.size()) [1, 3, 262, 518]
        # print(result_global.size()) [1, 3, 256, 512]
        # final = (result_global + result_local) * 0.5

        # x_encode [1 ,256, 64, 128]
        x_attention = nn.relu(self.deconv3_norm_attention(self.deconv3_attention(x_encode)))
        # print(x_attention.size()) [1, 128, 128, 256]
        x_attention = nn.relu(self.deconv4_norm_attention(self.deconv4_attention(x_attention)))
        # print(x_attention.size()) [1, 64, 256, 512]
        result_attention = self.deconv5_attention(x_attention)
        # print(result_attention.size()) [1, 2, 256, 512]
        # softmax_ = nn.Softmax(dim=1)
        result_attention = nn.softmax(result_attention, dim=1)

        attention_local = result_attention[:, 0:1, :, :]
        attention_global = result_attention[:, 1:2, :, :]

        attention_local = attention_local.repeat(1, 3, 1, 1)
        attention_global = attention_global.repeat(1, 3, 1, 1)

        final = attention_local * result_local + attention_global * result_global

        # attention_global_v = (attention_global - 0.5)/0.5 # for visualization
        # attention_local_v =  (attention_local - 0.5)/0.5 # for visualization
        # final = (result_global + result_local) * 0.5


        return final, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_26, label_3_28,  result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28, feature_score, target, valid_index.float(), attention_global, attention_local


# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            init.gauss_(self._modules[m].weight, mean, std)

    def execute(self, input):
        x = nn.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = nn.relu(self.conv1_norm(self.conv1(x)))
        x = nn.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x
class resnet_block_2(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block_2, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        # self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        # self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def execute(self, input):
        x = nn.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = nn.relu(self.conv1_norm(self.conv1(x)))
        # x = nn.pad(x, (selnn.padding, selnn.padding, selnn.padding, selnn.padding), 'reflect')
        # x = self.conv2_norm(self.conv2(x))

        return input + x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadeinstance3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def execute(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = jt.randn(input.size(0), self.opt.z_dim,
                                dtype=jt.float32)
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = nn.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(nn.leaky_relu(x, 2e-1))
        x = jt.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def execute(self, input, z=None):
        return self.model(input)
