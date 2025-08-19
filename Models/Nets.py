#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np


class BasicBlock(nn.Module):
	expansion = 1
	
	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(
			in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
		                       stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)
	
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4
	
	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
		                       stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion *
		                       planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion * planes)
		
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)
	
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 64
		
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512 * block.expansion, num_classes)
	
	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)
	
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


# def NarrowResNet18():
#     return NarrowResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18():
	return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
	return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
	return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
	return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
	return ResNet(Bottleneck, [3, 8, 36, 3])


# class NarrowResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(NarrowResNet, self).__init__()
#         self.in_planes = 1

#         self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.layer1 = self._make_layer(block, 1, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 1, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 1, num_blocks[2], stride=2)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         return out

class NarrowResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(NarrowResNet, self).__init__()
		self.in_planes = 1
		
		self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(1)
		self.layer1 = self._make_layer(block, 1, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 1, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 1, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 1, num_blocks[3], stride=2)
	
	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)
	
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		return out


def NarrowResNet18():
	return NarrowResNet(BasicBlock, [2, 2, 2, 2])


# class narrow_ResNet(nn.Module):
#     # by default : block = BasicBlock
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(narrow_ResNet, self).__init__()

#         self.in_planes = 1 # one channel chain

#         self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False) # original num_channel = 16
#         self.bn1 = nn.BatchNorm2d(1) # bn1
#         # => 1 x 32 x 32

#         self.layer1 = self._make_layer(block, 1, num_blocks[0], stride=1) # original num_channel = 16
#         # => 1 x 32 x 32

#         self.layer2 = self._make_layer(block, 1, num_blocks[1], stride=2) # original num_channel = 32
#         # => 1 x 16 x 16

#         self.layer3 = self._make_layer(block, 1, num_blocks[2], stride=2) # original num_channel = 64
#         # => 1 x 8 x 8

#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         return out

__all__ = [
	'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
	'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
	'''
	VGG model
	'''
	
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Linear(512, 10),
		)
		# Initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()
	
	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
	      512, 512, 512, 512, 'M'],
}


def vgg11():
	"""VGG 11-layer model (configuration "A")"""
	return VGG(make_layers(cfg['A']))


def vgg11_bn():
	"""VGG 11-layer model (configuration "A") with batch normalization"""
	return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
	"""VGG 13-layer model (configuration "B")"""
	return VGG(make_layers(cfg['B']))


def vgg13_bn():
	"""VGG 13-layer model (configuration "B") with batch normalization"""
	return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
	"""VGG 16-layer model (configuration "D")"""
	return VGG(make_layers(cfg['D']))


def vgg16_bn():
	"""VGG 16-layer model (configuration "D") with batch normalization"""
	return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
	"""VGG 19-layer model (configuration "E")"""
	return VGG(make_layers(cfg['E']))


def vgg19_bn():
	"""VGG 19-layer model (configuration 'E') with batch normalization"""
	return VGG(make_layers(cfg['E'], batch_norm=True))


def get_model(data):
	if data == 'fmnist' or data == 'fedemnist':
		return CNN_MNIST()
	elif data == 'mnist':
		return CNN_MNIST()
	elif data == 'emnist':
		return CNN_EMNIST()
	elif data == 'gtsrb':
		return CNN_GTSRB()
	elif data == 'cifar10':
		return vgg19()
	elif data == 'cifar100':
		return preactresnet18()
	elif data == 'loan':
		return LoanNet()


"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_GTSRB(nn.Module):
	def __init__(self, num_classes=43):
		super(CNN_GTSRB, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout = nn.Dropout(0.5)
		
		# 修改 fc1 输入维度为 8192（匹配保存的权重文件）
		self.fc1 = nn.Linear(8192, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, num_classes)
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = x.view(x.size(0), -1)  # 展平
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)
		return x


class PreActBasic(nn.Module):
	expansion = 1
	
	def __init__(self, in_channels, out_channels, stride):
		super().__init__()
		self.residual = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
		)
		
		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
			self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)
	
	def forward(self, x):
		res = self.residual(x)
		shortcut = self.shortcut(x)
		
		return res + shortcut


class PreActBottleNeck(nn.Module):
	expansion = 4
	
	def __init__(self, in_channels, out_channels, stride):
		super().__init__()
		
		self.residual = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, out_channels, 1, stride=stride),
			
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, padding=1),
			
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
		)
		
		self.shortcut = nn.Sequential()
		
		if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
			self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)
	
	def forward(self, x):
		res = self.residual(x)
		shortcut = self.shortcut(x)
		
		return res + shortcut


class PreActResNet(nn.Module):
	
	def __init__(self, block, num_block, class_num=100):
		super().__init__()
		self.input_channels = 64
		
		self.pre = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True)
		)
		
		self.stage1 = self._make_layers(block, num_block[0], 64, 1)
		self.stage2 = self._make_layers(block, num_block[1], 128, 2)
		self.stage3 = self._make_layers(block, num_block[2], 256, 2)
		self.stage4 = self._make_layers(block, num_block[3], 512, 2)
		
		self.linear = nn.Linear(self.input_channels, class_num)
	
	def _make_layers(self, block, block_num, out_channels, stride):
		layers = []
		
		layers.append(block(self.input_channels, out_channels, stride))
		self.input_channels = out_channels * block.expansion
		
		while block_num - 1:
			layers.append(block(self.input_channels, out_channels, 1))
			self.input_channels = out_channels * block.expansion
			block_num -= 1
		
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.pre(x)
		
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		
		x = F.adaptive_avg_pool2d(x, 1)
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		
		return x


def preactresnet18():
	return PreActResNet(PreActBasic, [2, 2, 2, 2])


def preactresnet34():
	return PreActResNet(PreActBasic, [3, 4, 6, 3])


def preactresnet50():
	return PreActResNet(PreActBottleNeck, [3, 4, 6, 3])


def preactresnet101():
	return PreActResNet(PreActBottleNeck, [3, 4, 23, 3])


def preactresnet152():
	return PreActResNet(PreActBottleNeck, [3, 8, 36, 3])


class LoanNet(nn.Module):
	def __init__(self, in_dim=92, n_hidden_1=46, n_hidden_2=23, out_dim=9):
		super(LoanNet, self).__init__()
		self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
		                            nn.Dropout(0.5),  # drop 50% of the neuron to avoid over-fitting
		                            nn.ReLU())
		self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
		                            nn.Dropout(0.5),
		                            nn.ReLU())
		self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
	
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		
		if np.isnan(np.sum(x.data.cpu().numpy())):
			raise ValueError()
		
		return x


class CNN_EMNIST(nn.Module):
	def __init__(self, num_classes=47):  # EMNIST 默认有 62 个类别（byclass 分支）
		super(CNN_EMNIST, self).__init__()
		# 卷积层
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))  # 输入通道为 1（灰度图），输出 32 通道
		self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))  # 输出 64 通道
		self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))  # 最大池化层
		self.drop1 = nn.Dropout(p=0.5)  # Dropout 防止过拟合
		
		# 全连接层
		self.fc1 = nn.Linear(9216, 128)  # 根据特征图大小调整输入尺寸
		self.drop2 = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(128, num_classes)  # 输出类别为 EMNIST 的类别数（默认为 62）
	
	def forward(self, x):
		# 卷积 + 激活
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		# 池化
		x = self.max_pool(x)
		# 展平
		x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
		# 全连接 + 激活
		x = self.drop1(x)
		x = F.relu(self.fc1(x))
		x = self.drop2(x)
		x = self.fc2(x)  # 最后一层输出 logits
		return x


class CNN_CIFAR(nn.Module):
	def __int__(self):
		super(CNN_CIFAR, self).__int__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=3,
			          out_channels=16,
			          kernel_size=5,
			          stride=1,
			          padding=2, ),
			nn.ReLU(),
			nn.Dropout(0.2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=16,
				out_channels=16,
				kernel_size=3,
				stride=1,
				padding=1,
			),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.25)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(
				in_channels=16,
				out_channels=32,
				kernel_size=5,
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.Dropout(0.5)
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,
				out_channels=32,
				kernel_size=3,
				stride=2,
				padding=1,
			),
			nn.ReLU(),
			nn.Dropout(0.5)
		)
		
		self.fc = nn.Sequential(
			nn.BatchNorm1d(2048),
			nn.Linear(2048, 218),
			nn.ReLU(),
			nn.Linear(218, 10)
		)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x


class CNN_MNIST(nn.Module):
	def __init__(self):
		super(CNN_MNIST, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
		self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
		self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
		self.drop1 = nn.Dropout(p=0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.drop2 = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(128, 10)
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.max_pool(x)
		x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
		x = self.drop1(x)
		x = F.relu(self.fc1(x))
		x = self.drop2(x)
		x = self.fc2(x)
		return x
