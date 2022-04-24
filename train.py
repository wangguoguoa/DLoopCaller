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
import torch.optim as optim
from torch.utils.data import DataLoader

# custom functions defined by user
from model import DeepCNN
from datasets import EPIDataSetTrain, EPIDataSetTest
from trainer import Trainer
from loss import OhemLoss



def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" DLoopCaller train model for chromatin loops")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    # parser.add_argument("-n", dest="name", type=str, default=None,
    #                     help="The name of a specified data.")

    parser.add_argument("-g", dest="gpu", type=str, default='1',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=1,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.01,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-m", dest="momentum", type=float, default=0.9,
                        help="Momentum for the SGD optimizer.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=30,
                        help="Number of training steps.")
    parser.add_argument("-w", dest="weight_decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("-p", dest="power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")

    parser.add_argument("-r", dest="restore", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)
    # motifLen = motifLen_dict[args.name]Trainer
    ######在单个chromsome上训练和测试
    # Data_negative = np.load(osp.join(args.data_dir, '%s_negative.npz' % args.name))
    # Data_positive = np.load(osp.join(args.data_dir, '%s_positive.npz' % args.name))
    #
    # seqs = np.concatenate((np.array(Data_negative['data']), np.array(Data_positive['data'])),axis=0)
    # # for i in range(len(seqs)):
    # #     seqs[i] = (seqs[i] > np.quantile(seqs[i],0.5)).astype(np.int_)
    # # ret3, th3 = cv2.threshold(seqs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # seqs = seqs.reshape((seqs.shape[0], 1, seqs.shape[1], seqs.shape[2]))
    # label = np.concatenate((np.array(Data_negative['label']),np.array(Data_positive['label'])),axis=0)
    # label = label.reshape((label.shape[0],1))




    # #######在全基因组上######
    files = os.listdir(args.data_dir)
    chromnames = []
    for file in files:
        ccname = file.split('_')[0]
        if ccname not in chromnames:
            chromnames.append(ccname)

    for chromname in chromnames:
        seqs = []
        label = []
        atac_info = []
        histone1 = []
        histone2 = []
        for filename in files:
            if filename.split('_')[0] != chromname:
                Data = np.load(osp.join(args.data_dir, filename))
                seqs.extend(Data['data'])
                atac_info.extend(Data['atac'])
                # histone1.extend(Data['histone1'])
                # histone2.extend(Data['histone2'])
                label.extend(Data['label'])
        seqs = np.array(seqs)
        label = np.array(label)
        atac_info = np.array(atac_info)
        label = label.reshape((label.shape[0], 1))
        for i in range(len(atac_info)):
            atac_info[i] = np.log10(1 + atac_info[i]*10)
            atac_info[i] = atac_info[i]/np.max(atac_info[i]+1)

            seqs[i] = np.log10(1 + seqs[i] * 10)
            seqs[i] = seqs[i]/np.max(seqs[i]+1)

        seqs = np.concatenate((seqs, atac_info), axis=1)
        ratio = 0.2
        number_t = int(len(seqs) * ratio)
        index = list(range(len(seqs)))
        index_test = random.sample(index, number_t)
        index_train = list(set(index) - set(index_test))


        f = open(osp.join(args.checkpoint, 'record.txt'), 'w')
        f.write('f1-score\n')
        # build training data generator
        data_tr = seqs[index_train]
        label_tr = label[index_train]
        train_data = EPIDataSetTrain(data_tr, label_tr)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
        # build test data generator
        data_te = seqs[index_test]
        label_te = label[index_test]
        test_data = EPIDataSetTest(data_te, label_te)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # implement
        f1_best = 0
        r_best = 0
        for _ in range(5):
            model = DeepCNN()
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.learning_rate)
            criterion = OhemLoss()
            start_epoch = 0

            if args.restore:
                print("Resume it from {}.".format(args.restore_from))
                checkpoint = torch.load(args.restore)
                state_dict = checkpoint["model_state_dict"]
                model.load_state_dict(state_dict, strict=False)

            # if there exists multiple GPUs, using DataParallel
            if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

            executor = Trainer(model=model,
                               optimizer=optimizer,
                               criterion=criterion,
                               device=device,
                               checkpoint=args.checkpoint,
                               start_epoch=start_epoch,
                               max_epoch=args.max_epoch,
                               train_loader=train_loader,
                               test_loader=test_loader,
                               lr_policy=None)

            r, f1, state_dict = executor.train()
            if (f1_best) < (f1):
                print("Store the weights of the model in the current run.\n")
                r_best = r
                f1_best = f1
                checkpoint_file = osp.join(args.checkpoint, '{}_model_best.pth'.format(chromname))
                torch.save({
                    'model_state_dict': state_dict
                }, checkpoint_file)

        f.write("{:.3f}\n".format(f1_best))
        # f.write("r: {:.3f}\tf1: {:.3f}\n".format(r_best, f1_best))
        f.close()

if __name__ == "__main__":
    main()
