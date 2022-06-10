from typing import overload

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module


class ResidualBlock2Layers(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.bn3(self.conv3(x))
        y += x
        return F.relu(y)


class ResidualBlock2LayerIntAct(ResidualBlock2Layers):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__(in_channels, out_channels, use_1x1conv, strides)

    def forward(self, x):
        y1 = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y1))
        if self.conv3 is not None:
            x = self.bn3(self.conv3(x))
        y += x
        return F.relu(y), y1


class CustomSequential(nn.Sequential):

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        activation_maps = []
        for module in self:
            input, input2 = module(input)
            activation_maps.append(input2)
            activation_maps.append(input)

        return input, activation_maps


class ResidualBlock3Layers(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                               padding=(1, 1), stride=(strides, strides), bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * ResidualBlock3Layers.expansion, kernel_size=(1, 1),
                               stride=(1, 1), bias=False)

        if use_1x1conv:
            self.conv4 = nn.Conv2d(in_channels, out_channels * ResidualBlock3Layers.expansion,
                                   kernel_size=(1, 1), stride=(strides, strides), bias=False)
            self.bn4 = nn.BatchNorm2d(out_channels * ResidualBlock3Layers.expansion)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * ResidualBlock3Layers.expansion)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.conv4 is not None:
            x = self.bn4(self.conv4(x))
        return F.relu(y + x)


def create_resnet_block(in_channels, out_channels, count_blocks, residual_block=ResidualBlock2Layers, use_1x1conv=True,
                        stride=2, expansion=1):
    block = []
    for i in range(count_blocks):
        if i == 0:
            block.append(residual_block(in_channels, out_channels, use_1x1conv, stride))
        else:
            block.append(residual_block(out_channels * expansion, out_channels))
    return block


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(*create_resnet_block(64, 64, 2, use_1x1conv=False, stride=1))
        self.conv3 = nn.Sequential(*create_resnet_block(64, 128, 2))
        self.conv4 = nn.Sequential(*create_resnet_block(128, 256, 2))
        self.conv5 = nn.Sequential(*create_resnet_block(256, 512, 2))
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        y = self.pooling(c5)
        return self.fc(self.flatten(y)), c5


class ResNet18IntAct(ResNet18):
    # use_activations is a list of tuples, one tuple for each group of residual blocks.
    # the 1st elem of the tuple is the index of the group, the second is the index of the activation within the group
    def __init__(self, use_activations, use_act_1st_layer=False, last_layer=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = CustomSequential(
            *create_resnet_block(64, 64, 2, use_1x1conv=False, stride=1, residual_block=ResidualBlock2LayerIntAct))
        self.conv3 = CustomSequential(*create_resnet_block(64, 128, 2, residual_block=ResidualBlock2LayerIntAct))
        self.conv4 = CustomSequential(*create_resnet_block(128, 256, 2, residual_block=ResidualBlock2LayerIntAct))
        self.conv5 = CustomSequential(*create_resnet_block(256, 512, 2, residual_block=ResidualBlock2LayerIntAct))
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, last_layer)
        self.use_activations = use_activations
        self.use_activation_1st_layer = use_act_1st_layer

    def forward(self, x):
        c1 = self.conv1(x)
        c2, activation_c2 = self.conv2(c1)
        c3, activation_c3 = self.conv3(c2)
        c4, activation_c4 = self.conv4(c3)
        c5, activation_c5 = self.conv5(c4)
        list_activations_maps = [activation_c2, activation_c3, activation_c4, activation_c5]
        y = self.pooling(c5)
        output_activations = []
        if self.use_activation_1st_layer:
            output_activations.append(c1)
        for tup in self.use_activations:
            output_activations.append(list_activations_maps[tup[0] - 2][tup[1] - 1])
        return self.fc(self.flatten(y)), output_activations


class ResNet18DiscFilters(ResNet18):
    def __init__(self, conv_layers):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        list_activations_maps = [c1, c2, c3, c4, c5]
        y = self.pooling(c5)
        return self.fc(self.flatten(y)), [list_activations_maps[i - 1] for i in self.conv_layers]


class ResNet50(nn.Module):
    def __init__(self, conv_layers, classes=100):
        super(ResNet50, self).__init__()
        self.conv_layers = conv_layers
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(*create_resnet_block(64, 64, 3, use_1x1conv=True,
                                                        stride=1, residual_block=ResidualBlock3Layers,
                                                        expansion=ResidualBlock3Layers.expansion))
        self.conv3 = nn.Sequential(*create_resnet_block(256, 128, 4, residual_block=ResidualBlock3Layers,
                                                        expansion=ResidualBlock3Layers.expansion))
        self.conv4 = nn.Sequential(*create_resnet_block(512, 256, 6, residual_block=ResidualBlock3Layers,
                                                        expansion=ResidualBlock3Layers.expansion))
        self.conv5 = nn.Sequential(*create_resnet_block(1024, 512, 3, residual_block=ResidualBlock3Layers,
                                                        expansion=ResidualBlock3Layers.expansion))
        self.dropout = nn.Dropout(0.4)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, classes)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        list_activations_maps = [c1, c2, c3, c4, c5]
        flatten = self.flatten(self.pooling(c5))
        return self.fc(self.dropout(flatten)), [list_activations_maps[i - 1] for i in self.conv_layers]
