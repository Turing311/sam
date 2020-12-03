import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class LiveModel(nn.Module):

    
    def __init__(self, weight_file):
        super(LiveModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.conv1 = self.__conv(2, name='conv1', in_channels=1, out_channels=8, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.conv2a = self.__conv(2, name='conv2a', in_channels=8, out_channels=12, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2 = self.__conv(2, name='conv2', in_channels=12, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3a = self.__conv(2, name='conv3a', in_channels=12, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv3 = self.__conv(2, name='conv3', in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4a = self.__conv(2, name='conv4a', in_channels=24, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv4 = self.__conv(2, name='conv4', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.fc1_1 = self.__dense(name = 'fc1_1', in_features = 2048, out_features = 128, bias = True)
        self.fc2_1 = self.__dense(name = 'fc2_1', in_features = 128, out_features = 2, bias = True)

    def forward(self, x):
        conv1_pad       = F.pad(x, (2, 2, 2, 2))
        conv1           = self.conv1(conv1_pad)
        ReLU1           = F.relu(conv1)
        pool1_pad       = F.pad(ReLU1, (0, 1, 0, 1), value=float('-inf'))
        pool1, pool1_idx = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2a          = self.conv2a(pool1)
        ReLU1_1         = F.relu(conv2a)
        conv2_pad       = F.pad(ReLU1_1, (1, 1, 1, 1))
        conv2           = self.conv2(conv2_pad)
        ReLU1_2         = F.relu(conv2)
        pool2_pad       = F.pad(ReLU1_2, (0, 1, 0, 1), value=float('-inf'))
        pool2, pool2_idx = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv3a          = self.conv3a(pool2)
        ReLU1_3         = F.relu(conv3a)
        conv3_pad       = F.pad(ReLU1_3, (1, 1, 1, 1))
        conv3           = self.conv3(conv3_pad)
        ReLU1_4         = F.relu(conv3)
        pool3_pad       = F.pad(ReLU1_4, (0, 1, 0, 1), value=float('-inf'))
        pool3, pool3_idx = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv4a          = self.conv4a(pool3)
        ReLU1_5         = F.relu(conv4a)
        conv4_pad       = F.pad(ReLU1_5, (1, 1, 1, 1))
        conv4           = self.conv4(conv4_pad)
        ReLU1_6         = F.relu(conv4)
        pool4_pad       = F.pad(ReLU1_6, (0, 1, 0, 1), value=float('-inf'))
        pool4, pool4_idx = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        fc1_0           = pool4.view(pool4.size(0), -1)
        fc1_1           = self.fc1_1(fc1_0)
        relufc1         = F.relu(fc1_1)
        fc2_0           = relufc1.view(relufc1.size(0), -1)
        fc2_1           = self.fc2_1(fc2_0)
        prov            = F.softmax(fc2_1)
        return prov


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

