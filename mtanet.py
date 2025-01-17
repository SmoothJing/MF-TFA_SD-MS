"""
gya@stu.xju.edu.cn

MTANet_construction - model

This file contains the model_achievement

"""
# MTANet

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ACBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(ACBlock,self).__init__()

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel,
                kernel_size=(3, 1), stride=(1, 1), padding=(2 ** dilation, 0), dilation=(2 ** dilation, 1)))
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel,
                kernel_size=(1, 3), stride=(1, 1), padding=(0, 2 ** dilation), dilation=(1, 2 ** dilation)))

    def forward(self, input):

        output2 = self.conv2(input)
        output3 = self.conv3(input)
        output = output2 + output3

        return output


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, memory_efficient=False, dilation=1):
        super(_DenseLayer, self).__init__()

        self.conv_block = nn.Sequential(
            ACBlock(num_input_features, growth_rate, dilation),
            ACBlock(growth_rate, growth_rate, dilation+1),
            nn.Sequential(
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=2**(dilation+1), dilation=2**(dilation+1),
                        bias=False))
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(num_input_features, growth_rate, kernel_size=1),
            nn.BatchNorm2d(growth_rate),
        )

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x

    def forward(self, input):  # noqa: F811
        prev_features = input

        residual = self.conv_skip(prev_features)

        new_features = self._pad(self.conv_block(prev_features), prev_features) + residual

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        self.recurrence = 2
        self.num_input_features = num_input_features
        out_channel = num_input_features + num_layers * growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                dilation=i
            )
            self.layers['denselayer%d' % (i + 1)] = layer
            num_input_features = growth_rate

        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, growth_rate, 1)
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(self.num_input_features, growth_rate, kernel_size=1),
            nn.BatchNorm2d(growth_rate),
        )

    def forward(self, init_features):
        features_list = [init_features]
        residual = self.conv_skip(init_features)
        for name, layer in self.layers.items():
            init_features = layer(init_features)
            features_list.append(init_features)
        output = self.conv1x1(torch.cat(features_list, dim=1)) + residual
        return output


class Multiscale_Module(nn.Module):
    def __init__(self, input_channel=2,
                     first_channel=32,
                     first_kernel=(3, 3),
                     scale=3,
                     kl=[(16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)],
                     drop_rate=0.1,
                     hidden=None,
                     in_size=None):
        super(Multiscale_Module, self).__init__()
        self.first_channel = 32
        self.dual_rnn = []

        self.En1 = _DenseBlock(kl[0][1], self.first_channel, kl[0][0], drop_rate)
        self.pool1 = nn.Sequential(
            nn.Conv2d(kl[0][0], kl[0][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.En2 = _DenseBlock(kl[1][1], kl[0][0], kl[1][0], drop_rate)
        self.pool2 = nn.Sequential(
            nn.Conv2d(kl[1][0], kl[1][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.En3 = _DenseBlock(kl[2][1], kl[1][0], kl[2][0], drop_rate)
        self.pool3 = nn.Sequential(
            nn.Conv2d(kl[2][0], kl[2][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # enter part
        self.Enter = _DenseBlock(kl[3][1], kl[2][0], kl[3][0], drop_rate)

        # decoder part
        self.up3 = nn.ConvTranspose2d(kl[3][0], kl[3][0], kernel_size=2, stride=2)
        self.De3 = _DenseBlock(kl[-3][1], kl[3][0]+kl[2][0], kl[-3][0], drop_rate)

        self.up2 = nn.ConvTranspose2d(kl[-3][0], kl[-3][0], kernel_size=2, stride=2)
        self.De2 = _DenseBlock(kl[-2][1], kl[-3][0]+kl[1][0], kl[-2][0], drop_rate)

        self.up1 = nn.ConvTranspose2d(kl[-2][0], kl[-2][0], kernel_size=2, stride=2)
        self.De1 = _DenseBlock(kl[-1][1], kl[-2][0]+kl[0][0], kl[-1][0], drop_rate)

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x

    def forward(self, input):
        x0 = input
        x1 = self.En1(x0)

        x_1 = self.pool1(x1)
        x2 = self.En2(x_1)
        x_2 = self.pool2(x2)
        x3 = self.En3(x_2)
        x_3 = self.pool3(x3)

        xy_ = self.Enter(x_3)


        y3 = self.up3(xy_)
        y_3 = self.De3(torch.cat([self._pad(y3, x3), x3], dim=1))
        y2 = self.up2(y_3)
        y_2 = self.De2(torch.cat([self._pad(y2, x2), x2], dim=1))
        y1 = self.up1(y_2)
        y_1 = self.De1(torch.cat([self._pad(y1, x1), x1], dim=1))

        output = self._pad(y_1, input)
        return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, do_activation=True):
        super(Conv, self).__init__()
        if not do_activation:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2))
        else:
            self.model = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2))

    def forward(self, x):
        x = self.model(x)
        return x


class TFatten(nn.Module):
    def __init__(self):
        super(TFatten, self).__init__()
        self.bn = nn.BatchNorm2d(22)
        self.t_conv1 = nn.Sequential(
            nn.Conv1d(22, 22, 3, padding=1),
            nn.ReLU()
        )
        self.t_conv2 = nn.Sequential(
            nn.Conv1d(22, 22, 3, padding=1),
            nn.Sigmoid()
        )
        self.f_conv1 = nn.Sequential(
            nn.Conv1d(22, 22, 3, padding=1),
            nn.ReLU()
        )
        self.f_conv2 = nn.Sequential(
            nn.Conv1d(22, 22, 3, padding=1),
            nn.Sigmoid()
        )
        self.x_conv1 = nn.Sequential(
            nn.Conv2d(22, 22, (3, 3), padding=1),
            nn.ReLU()
        )
        self.x_conv2 = nn.Sequential(
            nn.Conv2d(22, 22, (3, 3), padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.bn(x)
        x_hat = self.x_conv1(x)
        x_hat = self.x_conv2(x_hat)
        a_t = torch.mean(x, dim=-2)
        a_f = torch.mean(x, dim=-1)
        a_t = self.t_conv1(a_t)
        a_t = self.t_conv2(a_t)
        a_t = a_t.unsqueeze(dim=-2)
        a_f = self.f_conv1(a_f)
        a_f = self.f_conv2(a_f)
        a_f = a_f.unsqueeze(dim=-1)
        a_tf = a_t * a_f
        x_attn = a_tf * x_hat
        return x_attn, a_f, a_t


class MTAnet(nn.Module):
    def __init__(self, input_channel, drop_rate=0.1):
        super(MTAnet, self).__init__()
        kl_low = [(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)]
        kl_high = [(10, 3), (10, 3), (10, 3), (10, 3), (10, 3), (10, 3), (16, 3)]
        kl_full = [(6, 2), (6, 2), (6, 2), (6, 4), (6, 2), (6, 2), (6, 2)]     # 4、3、2代表num_layers(文章中的HDC blocks)
        in_size = [31, 32, 63]
        hidden = [128, 32, 128]
        self.lowNet = Multiscale_Module(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3,
                                      kl=kl_low, drop_rate=drop_rate, hidden=hidden[0], in_size=in_size[0])
        self.highNet = Multiscale_Module(input_channel=input_channel, first_channel=32, first_kernel=(3, 3), scale=3,
                                       kl=kl_high, drop_rate=drop_rate, hidden=hidden[0], in_size=in_size[1])
        self.fullNet = Multiscale_Module(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3,
                                       kl=kl_full, drop_rate=drop_rate, hidden=hidden[0], in_size=in_size[2])
        last_channel = kl_low[-1][0] + kl_full[-1][0]
        self.out = nn.Sequential(
            TFatten())

        self.channel_up = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.SELU()
        )
        self.channel_down = nn.Sequential(
            nn.Conv2d(22, 16, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(16, 1, 5, padding=2),
            nn.SELU()
        )
        self.bm_layer = nn.Sequential(
            nn.Conv2d(3, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (3, 1), stride=(3, 1)),  # 3
            nn.SELU(),
            nn.Conv2d(16, 16, (6, 1), stride=(6, 1)),  # 6
            nn.SELU(),
            nn.Conv2d(16, 1, (5, 1), stride=(5, 1)),
            nn.SELU()
        )
        self.bn_layer = nn.BatchNorm2d(3)
        self.residual = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=1),
            nn.BatchNorm2d(1)
        )
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x

    def forward(self, input):
        input = self.bn_layer(input)
        bm = input
        bm = self.bm_layer(bm)

        low_input = input[:, :, :348, :]
        high_input = input[:, :, -17:-1, :]
        low_input = self.channel_up(low_input)
        high_input = self.channel_up(high_input)

        low = self.lowNet(low_input)

        high = self.highNet(high_input)

        lower = low[:, :, :344, :]
        higher = high[:, :, 4:, ]

        middle_low = low[:, :, -5:-1, :]
        middle_high = high[:, :, :4, :]

        beta = self.beta
        middle = beta * middle_low + (1 - beta) * middle_high
        output = torch.cat([lower, middle, higher], 2)

        full_input = self.channel_up(input)
        full_output = self.fullNet(full_input)

        output = torch.cat([output, full_output], 1)

        output, a_f, a_t = self.out(output)

        output_pre = self._pad(output, input)
        output_pre = self.channel_down(output_pre)

        output_pre = torch.cat([bm, output_pre], dim=2)
        output = nn.Softmax(dim=-2)(output_pre)
        return output, output_pre

