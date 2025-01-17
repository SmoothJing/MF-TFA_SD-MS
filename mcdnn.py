# MCDNN from https://github.com/LqNoob/MelodyExtraction-MCDNN/blob/master/MelodyExtraction_SCDNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class MCDNN(nn.Module):
    def __init__(self):
        super(MCDNN, self).__init__()

        self.mcdnn = nn.Sequential(
            nn.Linear(360 * 3, 2048),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(512, 360)
        )   # 以上定义主网络层
        self.bm_layer = nn.Sequential(
            nn.Linear(360 * 3, 512),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(128, 1),
            nn.SELU()
        )    # 以上定义辅助网络层

    def forward(self, x):
        # [bs, 3, f, t]    即输入 x 的形状为 [batch_size, 3, frequency, time_steps]
        x = x.view(x.shape[0], -1, x.shape[-1])  # 将输入张量 reshape 成 [batch_size, 3*f, t] 的形状
        x = x.permute(0,2,1)    # [bs, t, f * 3]
        output_pre = self.mcdnn(x)
        bm = self.bm_layer(x)
        output_pre = output_pre.permute(0,2,1)  # [batch_size, 360, t]
        output_pre = output_pre.unsqueeze(dim=1)  # [batch_size, 1, 360, t]
        bm = bm.permute(0,2,1)            # [batch_size, 1, t]
        bm = bm.unsqueeze(dim=1)          # [batch_size, 1, 1, t]
        output_pre = torch.cat((bm, output_pre), dim=2)       # [batch_size, 1, 361, t]
        output = nn.Softmax(dim=2)(output_pre)

        return output, output_pre

# if __name__ == '__main__':
#     x = torch.randn(12,3,360,128)
#     model = MCDNN()
#     y = model(x)

# model = MCDNN()
# weights = model.mcdnn[6].weight
# bias = model.mcdnn[6].bias
# print('ww',weights)
# print('ww',weights.shape)
# print('bias',bias)
# print('bias',bias.shape)
