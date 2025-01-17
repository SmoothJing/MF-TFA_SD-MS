import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 3), stride=(1, 1),
                               padding=(2, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 1), pool_type='avg'):
        x_1 = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x_1)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class TFABlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFABlock,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),)

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),)

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channel*3, out_channel, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, input):

        b, c, f, t = input.size()
        feature = self.conv1(input)

        input_t = F.adaptive_avg_pool2d(input, (1, t)).squeeze(-2)
        input_f = F.adaptive_avg_pool2d(input, (f, 1)).squeeze(-1)

        out1 = self.conv2(input_t)
        out2 = self.conv3(out1)
        output2 = out2.unsqueeze(-2) * feature

        out3 = self.conv5(input_f)
        out4 = self.conv6(out3)
        output3 = out4.unsqueeze(-1) * feature

        output = torch.cat([output2, output3, feature], dim=1)
        output = self.conv7(output)

        return output


class MSC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.att_conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.att_conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim*3, dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.att_conv1(x)
        out2 = self.att_conv2(out1)
        out3 = self.att_conv3(out2)

        output = torch.cat([out1, out2, out3], dim=1)
        output = self.conv1(output)

        return output


class MF_TFA(nn.Module):
    def __init__(self, in_c, out_c, stride=1, scale=4, stype='normal'):
        super(MF_TFA, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convs = []
        for i in range(self.nums):
            convs.append(MSC(out_c // 4))
        self.dlk_layers = nn.ModuleList(convs)

        self.conv3 = nn.Conv2d(out_c, in_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_c)

        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale

        self.TFA = TFABlock(in_c, in_c)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.chunk(out, 4, dim=1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.dlk_layers[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.TFA(out)

        out = residual + out
        out = self.relu(out)
        return out


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
else:
    device = torch.device("cpu")
    print("Using cpu")


class BranchBottleNeck(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(BranchBottleNeck, self).__init__()

        middle_channel = channel_in // 4
        self.conv1_block = nn.Sequential(
            nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU()
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(middle_channel, middle_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU()
        )
        self.conv3_block = nn.Sequential(
            nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU()
        )
        self.conv4_block = nn.Sequential(
            nn.Conv2d(channel_out, 1, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x, channel='false'):
        out = self.conv1_block(x)
        out = self.conv2_block(out)
        out = self.conv3_block(out)
        out1 = None
        if channel == 'true':
            out1 = self.conv4_block(out)
        return out, out1


class MFT_FANet(nn.Module):
    def __init__(self, momentum=0.01):
        super(MFT_FANet, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=16, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=16, out_channels=32, momentum=momentum)

        self.bn_layer = nn.BatchNorm2d(3)
        self.bm_layer = nn.Sequential(
            nn.Conv2d(3, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (3, 1), stride=(3, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (6, 1), stride=(6, 1)),
            nn.SELU(),
            nn.Conv2d(16, 1, (5, 1), stride=(5, 1)),
            nn.SELU()
        )

        self.MF_TFA_1 = MF_TFA(32, 64)
        self.MF_TFA_2 = MF_TFA(32, 64)
        self.bottleneck1_1 = BranchBottleNeck(32, 2)
        self.middle_fc1 = nn.Linear(720, 361)

        self.MF_TFA_3 = MF_TFA(32, 64)
        self.MF_TFA_4 = MF_TFA(32, 64)
        self.bottleneck1_2 = BranchBottleNeck(32, 2)
        self.middle_fc2 = nn.Linear(720, 361)

        self.MF_TFA_5 = MF_TFA(32, 64)

        self.channel_down = nn.Sequential(
            nn.Conv2d(32, 16, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(16, 1, 5, padding=2),
            nn.SELU()
        )

    def forward(self, x):
        input = self.bn_layer(x)
        bm = input
        bm = self.bm_layer(bm)

        x = self.conv_block1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.MF_TFA_1(x)
        x = self.MF_TFA_2(x)
        middle1, feature1 = self.bottleneck1_1(x, channel='true')
        feature1 = feature1
        middle1 = middle1.transpose(2, 3).transpose(1, 2).flatten(2)
        middle1 = self.middle_fc1(middle1).transpose(1, 2)
        middle1 = nn.Softmax(dim=-2)(middle1)

        x = self.MF_TFA_3(x)
        x = self.MF_TFA_4(x)
        middle2, feature2 = self.bottleneck1_2(x, channel='true')
        feature2 = feature2
        middle2 = middle2.transpose(2, 3).transpose(1, 2).flatten(2)
        middle2 = self.middle_fc2(middle2).transpose(1, 2)
        middle2 = nn.Softmax(dim=-2)(middle2)

        x = self.MF_TFA_5(x)
        output = self.channel_down(x)
        feature = output

        output_pre = torch.cat([bm, output], dim=2)
        output = nn.Softmax(dim=-2)(output_pre)

        return output, middle1, middle2, feature1, feature2, feature, output_pre

