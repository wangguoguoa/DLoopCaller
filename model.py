# -*- coding: utf8 -*-
import torch
import torch.nn as nn


import numpy as np
import sys


class DeepCNN(nn.Module):
    """FPN for semantic segmentation"""
    def __init__(self, motiflen=7):
        super(DeepCNN, self).__init__()
        # encode process
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(5, 5), padding=2, bias=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        # classifier head
        c_in = 128 # 384
        self.bn = nn.BatchNorm2d(32)
        self.linear1 = nn.Linear(c_in, 64)
        self.drop = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(64, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _, _ = data.size()
        # encode process
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = self.bn(out1)
        skip4 = out1
        # classifier
        out2 = skip4.view(b, -1)
        out2 = self.linear1(out2)
        out2 = self.relu(out2)
        out2 = self.drop(out2)
        out2 = self.linear2(out2)
        out_class = self.sigmoid(out2)

        return out_class


class DanQ(nn.Module):
    """FPN for semantic segmentation"""
    def __init__(self):
        super(DanQ, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.pool1 = nn.MaxPool1d(kernel_size=13, stride=13)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size=320, hidden_size=320, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.5)
        # classifier head
        c_in = 896 # 384
        self.linear1 = nn.Linear(c_in, 925)
        self.linear2 = nn.Linear(925, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout1(out1)
        out1 = out1.permute(0, 2, 1)
        out1, _ = self.lstm(out1)
        out1 = self.dropout2(out1)
        skip4 = out1
        # classifier
        out2 = skip4.view(b, -1)
        out2 = self.linear1(out2)
        out2 = self.relu(out2)
        out2 = self.linear2(out2)
        out_class = self.sigmoid(out2)

        return out_class

