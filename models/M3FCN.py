import torch
import torch.nn.functional as F
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCN1(nn.Module):

    def __init__(self, filters, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.filters = filters
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = nn.Conv2d(in_channels=inplanes, out_channels=filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.inbn = nn.BatchNorm2d(filters)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.down_module_list = nn.ModuleList()
        self.down_conv_list = nn.ModuleList()
        self.down_bn_list = nn.ModuleList()
        self.down_scale_conv_list = nn.ModuleList()
        self.down_scale_bn_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(filters * (2 ** i), filters * (2 ** i)))
            self.down_conv_list.append(nn.Conv2d(filters * 2 ** i, filters * 2 ** (i + 1), kernel_size=kernel, stride=2, padding=self.padding))
            self.down_bn_list.append(nn.BatchNorm2d(filters * 2 ** (i + 1)))
            if i != layers - 1:
                self.down_scale_conv_list.append(nn.Conv2d(inplanes, filters * 2 ** (i + 1), kernel_size=kernel, stride=1, padding=self.padding))
                self.down_scale_bn_list.append(nn.BatchNorm2d(filters * 2 ** (i + 1)))

        self.bottom = block(filters * (2 ** layers), filters * (2 ** layers))

        self.up_conv_list = nn.ModuleList()
        self.up_bn_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(nn.ConvTranspose2d(in_channels=filters * 2 ** (layers - i), out_channels=filters * 2 ** max(0, layers - i - 1),
                                                        kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
            self.up_bn_list.append(nn.BatchNorm2d(filters * 2 ** max(0, layers - i - 1)))
            self.up_dense_list.append(block(filters * 2 ** max(0, layers - i - 1), filters * 2 ** max(0, layers - i - 1)))

    def forward(self, x):

        x_ = x

        out = self.leakyRelu(self.inbn(self.inconv(x)))

        down_out = []
        for i in range(0, self.layers):
            out = self.down_module_list[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = self.down_bn_list[i](out)
            if i != self.layers - 1:
                x_ = F.avg_pool2d(x_, 2)
                input = self.down_scale_conv_list[i](x_)
                input = self.down_scale_bn_list[i](input)
                out = self.leakyRelu(out + input)
            else:
                out = self.leakyRelu(out)

        out = self.bottom(out)
        bottom = out

        up_out = []
        up_out.append(bottom)
        for j in range(0, self.layers):
            out = self.up_conv_list[j](out)
            out = self.up_bn_list[j](out)
            out += down_out[self.layers - j - 1]
            out = self.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class FCN2(nn.Module):

    def __init__(self, filters, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.filters = filters
        self.layers = layers
        self.kernel = kernel
        self.inplanes = inplanes

        self.padding = int((kernel - 1) / 2)
        self.inblock = block(filters, filters)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.down_module_list = nn.ModuleList()
        self.down_conv_list = nn.ModuleList()
        self.down_bn_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(filters * (2 ** i), filters * (2 ** i)))
            self.down_conv_list.append(nn.Conv2d(filters * 2 ** i, filters * 2 ** (i + 1), stride=2, kernel_size=kernel, padding=self.padding))
            self.down_bn_list.append(nn.BatchNorm2d(filters * 2 ** (i + 1)))

        self.bottom = block(filters * (2 ** layers), filters * (2 ** layers))

        self.up_conv_list = nn.ModuleList()
        self.up_bn_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        self.up_out_conv_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(nn.ConvTranspose2d(filters * 2 ** (layers - i), filters * 2 ** max(0, layers - i - 1), kernel_size=3,
                                                        stride=2, padding=1, output_padding=1, bias=True))
            self.up_bn_list.append(nn.BatchNorm2d(filters * 2 ** max(0, layers - i - 1)))
            self.up_dense_list.append(block(filters * 2 ** max(0, layers - i - 1), filters * 2 ** max(0, layers - i - 1)))

    def forward(self, x):
        out = self.inblock(x[-1])

        down_out = []
        for i in range(0, self.layers):
            out = out + x[-i - 1]
            out = self.down_module_list[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = self.down_bn_list[i](out)
            out = self.leakyRelu(out)

        out = self.bottom(out)

        up_out = []
        for j in range(0, self.layers):
            out = self.up_conv_list[j](out)
            out = self.up_bn_list[j](out)
            out += down_out[self.layers - j - 1]
            out = self.relu(out)
            out = self.up_dense_list[j](out)
            if j != self.layers - 1:
                up_out.append(F.interpolate(out, size=(out.shape[2] * 2 ** (self.layers - j - 1), out.shape[3] * 2 ** (self.layers - j - 1))))
            else:
                up_out.append(out)

        return up_out


class M3FCN(nn.Module):
    def __init__(self, layers=4, filters=10, num_classes=2, inplanes=1, block=BasicBlock):
        super().__init__()
        self.layers = layers
        self.left_block = FCN1(filters=filters, layers=layers, block=block, inplanes=inplanes)
        self.right_block = FCN2(filters=filters, layers=layers, block=block, inplanes=inplanes)
        self.out_conv = nn.ModuleList()
        for i in range(layers):
            self.out_conv.append(nn.Conv2d(in_channels=filters * 2 ** (self.layers - i - 1), out_channels=num_classes, kernel_size=1))

    def forward(self, x):
        left_out = self.left_block(x)
        right_out = self.right_block(left_out)
        out_final = None
        for i in range(self.layers):
            out = self.out_conv[i](right_out[i])
            if out_final is None:
                out_final = out
            else:
                out_final = out_final + out

        out_final /= self.layers
        return out_final

