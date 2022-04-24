#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemLoss(nn.Module):
    def __init__(self):
        super(OhemLoss, self).__init__()
        self.criteria = nn.BCELoss()

    def forward(self, label_p, label_t):
        label_p = label_p.view(-1)
        label_t = label_t.view(-1)
        loss = self.criteria(label_p, label_t)
        return loss


