#!/usr/bin/python

import os
import sys
import time
import argparse
import random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# custom functions defined by user
from model import DeepCNN, DanQ
from datasets import EPIDataSetTrain, EPIDataSetTest
from trainer import Trainer
from loss import OhemLoss
from utils import *
from sklearn.metrics import f1_score,precision_recall_curve,auc
from sklearn.metrics import average_precision_score, roc_auc_score


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCN for motif location")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")

    files = os.listdir(args.data_dir)
    chromnames = []
    for file in files:
        ccname = file.split('_')[0]
        if ccname not in chromnames:
            chromnames.append(ccname)
    f = open('/home/sc3/wsg/deeploop/three_data_models/GM12878_h3k27ac/record_test'
             '.txt', 'w')
    f.write('chromname,f1-score,prauc\n')
    for key in chromnames:
        Data = np.load(osp.join(args.data_dir, '%s_negative.npz' % key))
        seqs_neg, atat_neg, histone1_neg,label_neg = Data['data'], Data['atac'],Data['histone1'], Data['label']

        Data = np.load(osp.join(args.data_dir, '%s_positive.npz' % key))
        seqs_pos, atac_pos,histone1_pos, label_pos = Data['data'], Data['atac'],Data['histone1'], Data['label']

        seqs = np.concatenate((seqs_pos, seqs_neg), axis=0)
        atac = np.concatenate((atac_pos, atat_neg), axis=0)
        histone1 = np.concatenate((histone1_pos,histone1_neg),axis=0)
        for i in range(len(atac)):
            atac[i] = np.log10(1 + atac[i]*10)
            histone1[i] = np.log10(1 + histone1[i]*10)
            seqs[i] = np.log10(1 + seqs[i]*10)

            atac[i] = atac[i]/np.max(atac[i]+1)
            histone1[i] = histone1[i]/np.max(histone1[i]+1)
            seqs[i] = seqs[i]/np.max(seqs[i]+1)
        seqs = np.concatenate((seqs,histone1), axis=1)
        seqs = np.concatenate((seqs, atac), axis=1)
        label = np.concatenate((label_pos, label_neg), axis=0)
        # build test data generator
        data_te = seqs
        label_te = label
        test_data = EPIDataSetTest(data_te, label_te)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
        # Load weights
        checkpoint_file = osp.join(args.checkpoint, '{}_model_best.pth'.format(key))
        chk = torch.load(checkpoint_file)
        state_dict = chk['model_state_dict']
        model = DeepCNN()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        p_all = []
        t_all = []
        for i_batch, sample_batch in enumerate(test_loader):
            X_data = sample_batch["data"].float().to(device)
            signal = sample_batch["label"].float()
            with torch.no_grad():
                pred = model(X_data)
            p_all.append(pred.view(-1).data.cpu().numpy()[0])
            t_all.append(signal.view(-1).data.numpy()[0])
        f1 = f1_score(t_all, [int(x > 0.5) for x in p_all])
        precision,recall,_ = precision_recall_curve(t_all, [int(x > 0.5) for x in p_all])
        prauc = auc(recall,precision)

        f.write("chrom: {}\tf1: {:.3f}\tprauc: {:.3f}\n".format(key, f1, prauc))
    f.close()


if __name__ == "__main__":
    main()

