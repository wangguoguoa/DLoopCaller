import os
import gc
import pathlib
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
import pyBigWig


class Chromosome():
    def __init__(self, coomatrix, model, ATAC,lower=1, upper=500, cname='chrm', res=10000, width=11 ):
        # cLen = coomatrix.shape[0] # seems useless
        R, C = coomatrix.nonzero()
        validmask = np.isfinite(coomatrix.data) & (
            C-R+1 > lower) & (C-R < upper)
        R, C, data = R[validmask], C[validmask], coomatrix.data[validmask]
        self.M = sparse.csr_matrix((data, (R, C)), shape=coomatrix.shape)
        self.ridx, self.cidx = R, C
        self.ATAC = ATAC
        self.chromname = cname
        self.r = res
        self.w = width
        self.model = model

    def getwindow(self, coords):
        """
        Generate training set
        :param Matrix: single chromosome dense array
        :param coords: List of tuples containing coord bins
        :param width: Distance added to center. width=5 makes 11x11 windows
        :return: yields paired positive/negative samples for training
        """
        out_dir = '/home/sc3/wsg/deeploop/models/'
        bw = pyBigWig.open(self.ATAC)
        seq,clist,atac = [],[],[]
        width = self.w
        for i, c in enumerate(coords):
            if (i+1) % 1000 == 0:
                print("The current iteration is {}".format(i+1))
            # if i == 100:
            #     break
            x, y = c[0], c[1]
            try:
                window = self.M[x-width:x+width+1,
                                y-width:y+width+1].toarray()
            except:
                continue
            if np.count_nonzero(window) < window.size*.2:
                pass
            if np.isfinite(window).all() and window.shape == (2*width+1,2*width+1):
                try:
                    window_x = np.array(bw.values(self.chromname, (x - width) * self.r, (x + width + 1) * self.r))
                    window_x[np.isnan(window_x)] = 0
                    window_x = window_x.reshape(2*width+1, self.r)
                    window_x = [window_x.mean(axis=1)]
                    window_y = np.array(bw.values(self.chromname, (y - width) * self.r, (y + width + 1) * self.r))
                    window_y[np.isnan(window_y)] = 0
                    window_y = window_y.reshape(2*width+1, self.r)
                    window_y = [window_y.mean(axis=1)]
                    window_atac = np.dot(np.transpose(window_x), window_y)
                    seq.append(window)
                    clist.append(c)
                    atac.append(window_atac)
                except:
                    continue
        seq = np.array(seq)
        atac = np.array(atac)
        seq = seq.reshape((seq.shape[0], 1, seq.shape[1], seq.shape[2]))
        atac = atac.reshape((atac.shape[0], 1, atac.shape[1], atac.shape[2]))
        for i in range(len(seq)):
            seq[i] = seq[i] / np.max(seq[i]+1)
            atac[i] = np.log10(1 + atac[i] * 10)
            atac[i] = atac[i] / np.max(atac[i]+1)
        fts = np.concatenate((seq, atac), axis=1)
        # np.savez(out_dir + '/{}_sample.npz'.format(self.chromname), data=fts, clist=clist)

        return fts, clist

    def test(self, fts):
        num_total = len(fts)
        batch = 20
        iteration = int(np.ceil(num_total/batch))
        preds = np.array([])
        for i in range(iteration):
            segment = fts[i*batch:(i+1)*batch]
            segment = torch.from_numpy(segment)
            segment = segment.float().to(torch.device("cuda:0"))
            with torch.no_grad():
                label_p = self.model(segment)
            probas = label_p.view(-1).data.cpu().numpy()
            preds = np.concatenate((preds, probas))
            print("The current number is {}".format((i+1)*batch))

        return preds




    def score(self, thre=0.5):
        print('scoring matrix {}'.format(self.chromname))
        print('num candidates {}'.format(self.M.data.size))
        coords = [(r, c) for r, c in zip(self.ridx, self.cidx)]
        fts, clist = self.getwindow(coords)
        p = self.test(fts)
        clist = np.r_[clist]
        pfilter = p > thre
        ri = clist[:, 0][pfilter]
        ci = clist[:, 1][pfilter]
        result = sparse.csr_matrix((p[pfilter], (ri, ci)), shape=self.M.shape)
        data = np.array(self.M[ri, ci]).ravel()
        self.M = sparse.csr_matrix((data, (ri, ci)), shape=self.M.shape)

        return result, self.M

    def writeBed(self, out, prob_csr, raw_csr):
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        with open(out + '/' + self.chromname + '.bed', 'w') as output_bed:
            r, c = prob_csr.nonzero()
            for i in range(r.size):
                line = [self.chromname, r[i]*self.r, (r[i]+1)*self.r,
                        self.chromname, c[i]*self.r, (c[i]+1)*self.r,
                        prob_csr[r[i],c[i]], raw_csr[r[i],c[i]]]
                output_bed.write('\t'.join(list(map(str, line)))+'\n')
