'''https://blog.csdn.net/jiangpeng59/article/details/80189889'''
'''https://github.com/JavisPeng/u_net_liver/blob/master/unet.py'''

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch=1, channel_nums=[64, 128, 256, 512, 1024], activation=None):
        super(Unet, self).__init__()

        assert activation in ['sigmoid', 'softmax', None], 'activation function error.'

        self.conv1 = DoubleConv(in_ch, channel_nums[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(channel_nums[0], channel_nums[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(channel_nums[1], channel_nums[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(channel_nums[2], channel_nums[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(channel_nums[3], channel_nums[4])

        self.up6 = nn.ConvTranspose2d(channel_nums[4], channel_nums[3], 2, stride=2)
        self.conv6 = DoubleConv(channel_nums[4], channel_nums[3])
        self.up7 = nn.ConvTranspose2d(channel_nums[3], channel_nums[2], 2, stride=2)
        self.conv7 = DoubleConv(channel_nums[3], channel_nums[2])
        self.up8 = nn.ConvTranspose2d(channel_nums[2], channel_nums[1], 2, stride=2)
        self.conv8 = DoubleConv(channel_nums[2], channel_nums[1])
        self.up9 = nn.ConvTranspose2d(channel_nums[1], channel_nums[0], 2, stride=2)
        self.conv9 = DoubleConv(channel_nums[1], channel_nums[0])
        self.conv10 = nn.Conv2d(channel_nums[0], out_ch, kernel_size=1)

        self.activation = activation

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)


        if self.activation == 'sigmoid':
            out = nn.Sigmoid()(c10)
        elif self.activation == 'softmax':
            out = nn.Softmax()(c10)
        else:
            out = c10

        return out