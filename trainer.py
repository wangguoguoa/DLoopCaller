import os
import math
import datetime
import numpy as np
import os.path as osp
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,recall_score

import torch


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch,
                 train_loader, test_loader, lr_policy):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        if not osp.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        self.LR_policy = lr_policy
        self.epoch = 0
        self.r = 0.5
        self.f1 = 0
        self.state_best = None

    def train(self):
        """training the model"""
        self.model.to(self.device)
        self.criterion.to(self.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            # set training mode during the training process
            self.model.train()
            self.epoch = epoch
            # self.LR_policy.step() # for cosine learning strategy
            for i_batch, sample_batch in enumerate(self.train_loader):
                X_data = sample_batch["data"].float().to(self.device)
                label = sample_batch["label"].float().to(self.device)
                self.optimizer.zero_grad()
                label_p = self.model(X_data)
                loss = self.criterion(label_p, label)
                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')
                loss.backward()
                self.optimizer.step()
                print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.5f}".format(self.epoch, i_batch,
                                                    loss.item(), self.optimizer.param_groups[0]['lr']))
            # validation and save the model with higher accuracy
            self.test()

        return self.r, self.f1, self.state_best

    def test(self):
        """validate the performance of the trained model."""
        self.model.eval()
        label_p_all = []
        label_t_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            X_data = sample_batch["data"].float().to(self.device)
            label = sample_batch["label"].float().to(self.device)
            with torch.no_grad():
                label_p = self.model(X_data)
            label_p_all.append(label_p.view(-1).data.cpu().numpy()[0])
            label_t_all.append(label.view(-1).data.cpu().numpy()[0])

        r = recall_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        f1 = f1_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        if (self.r + self.f1) < (r + f1):
            self.f1 = f1
            self.r = r
            self.state_best = self.model.state_dict()
        print("r: {:.3f}\tf1: {:.3f}\n".format(r, f1))

