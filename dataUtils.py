#!/usr/bin/env python
# Program to train classifier given a cooler file and
# paired bedfile containing ChIA-PET peaks
# Author: Tarik Salameh

import numpy as np
from random import randint
from sklearn.ensemble import RandomForestClassifier as forest
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from collections import defaultdict, Counter
from scipy import stats
import gc
import random
import pyBigWig



def buildmatrix(Matrix, coords, width=11, lower=1, positive=True, stop=5000):
    """
    Generate training set
    :param coords: List of tuples containing coord bins
    :param width: Distance added to center. width=5 makes 11x11 windows
    :return: yields paired positive/negative samples for training
    """
    w2 = int(width//2)
    negcount = 0
    for c in coords:
        x, y = c[0], c[1]
        if y-x < lower:
            pass
        else:
            try:
                window = Matrix[x-width:x+width+1,
                                y-width:y+width+1].toarray()
                if window.size!= 23*23:
                    continue
                if np.count_nonzero(window) < window.size*.1:
                    pass
                else:
                    center = window[width, width]
                    ls = window.shape[0]
                    p2LL = center/np.mean(window[ls-1-ls//4:ls, :1+ls//4])
                    if positive and p2LL < 0.1:
                        pass
                    else:
                        if np.all(np.isfinite(window)):
                            if not positive:
                                negcount += 1
                            if negcount >= stop:
                                raise StopIteration

                            yield window
            except:
                pass


def trainRF(X, F, nproc=1):
    """
    :param X: training set from buildmatrix
    :param distances:
    """
    print('input data {} peaks and {} background'.format(
        X.shape[0], F.shape[0]))
    gc.collect()#清理内存
    params = {}
    params['class_weight'] = ['balanced', None]
    #params['class_weight'] += [{1: w} for w in range(5,10000,500)]
    params['n_estimators'] = [100]
    params['n_jobs'] = [1]
    params['max_features'] = ['auto']
    params['max_depth'] = [20]
    params['random_state'] = [42]
    #from hellinger_distance_criterion import HellingerDistanceCriterion as hdc
    #h = hdc(1,np.array([2],dtype='int64'))
    params['criterion'] = ['gini']
    #model = forest(**params)
    mcc = metrics.make_scorer(metrics.matthews_corrcoef)#创建一个计分标准
    model = GridSearchCV(forest(), param_grid=params,
                         scoring=mcc, verbose=2, n_jobs=1, cv=3)
    y = np.array([1]*X.shape[0] + [0]*F.shape[0])
    x = np.vstack((X, F))
    model.fit(x, y)
    fts = model.best_estimator_.feature_importances_[:]
    params = model.best_params_
    print(params)
    print(model.best_score_)
    fts = fts.tolist()
    print('{} peaks {} controls'.format(X.shape[0], F.shape[0]))
    return model.best_estimator_

#在正样本每条染色体中取起始点，然后对每对进行排序
#res和upper代表什么？
def parsebed(chiafile, res=10000, lower=1, upper=5000000):

    coords = defaultdict(set)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a //= res ##为什么这么做？
            b //= res
            # all chromosomes including X and Y
            if (b-a > lower) and (b-a < upper) and 'M' not in s[0]:
                # always has prefix "chr", avoid potential bugs
                chrom = 'chr' + s[0].lstrip('chr')# lstrip() 方法用于截掉字符串左边的空格或指定字符
                coords[chrom].add((a, b))

    for c in coords:
        coords[c] = sorted(coords[c])

    return coords


def learn_distri_kde(coords):

    dis = []
    for c in coords:
        for a, b in coords[c]:
            dis.append(b-a)

    lower = min(dis)

    # part 1: same distance distribution as the positive input
    #估计每对loop距离的概率密度函数，有6343对loop组成的一维向量包含6343个距离
    kde = stats.gaussian_kde(dis)#核密度估计是一种以非参数方式估计随机变量的概率密度函数

    # part 2: random long-range interactions
    counts, bins = np.histogram(dis, bins=100)#将距离数组分成100个柱状，counts指每个柱状里面的数据个数，bins指统计的区间
    long_end = int(bins[-1])#一对loop中，两个start距离最远的距离，即最后bin区间的位置
    tp = np.where(np.diff(counts) >= 0)[0] + 2 #输出后面柱状比前面柱状数据多的位置
    long_start = int(bins[tp[0]])#取第一个柱状比前一个柱状多的位置的bin区间的位置作为long_start

    return kde, lower, long_start, long_end


def negative_generating(M, kde, positives, lower, long_start, long_end):

    positives = set(positives)
    N = 3 * len(positives)
    # part 1: kde trained from positive input
    part1 = kde.resample(N).astype(int).ravel()
    part1 = part1[(part1 >= lower) & (part1 <= long_end)]

    # part 2: random long-range interactions
    part2 = []
    pool = np.arange(long_start, long_end+1)
    #print(M.shape[0])
    tmp = np.cumsum(M.shape[0]-pool)#???
    ref = tmp / tmp[-1]
    for i in range(N):
        r = np.random.random()
        ii = np.searchsorted(ref, r)
        part2.append(pool[ii])

    sample_dis = Counter(list(part1) + part2)

    neg_coords = []
    midx = np.arange(M.shape[0])
    for i in sorted(sample_dis):  # i cannot be zero
        n_d = sample_dis[i]
        R, C = midx[:-i], midx[i:]
        tmp = np.array(M[R, C]).ravel()
        tmp[np.isnan(tmp)] = 0
        mask = tmp > 0
        R, C = R[mask], C[mask]
        pool = set(zip(R, C)) - positives
        sub = random.sample(pool, n_d)
        neg_coords.extend(sub)

    random.shuffle(neg_coords)

    return neg_coords

def getbigwig(file,chrome,start,end):
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrome,start,end))
    sample[np.isnan(sample)] = 0
    bw.close()
    return sample

def generateATAC(Matrix, coords, chromname, files, resou,  width=11, lower=1, positive=True, stop=5000):
    """
    Generate training set
    :param coords: List of tuples containing coord bins
    :param width: Distance added to center. width=5 makes 11x11 windows
    :return: yields paired positive/negative samples for training
    """
    w2 = int(width//2)
    negcount = 0
    for c in coords:
        x, y = c[0], c[1]
        if y-x < lower:
            pass
        else:
            try:
                window = Matrix[x-width:x+width+1,
                                y-width:y+width+1].toarray()
                if window.size!= 23*23:
                    continue
                if np.count_nonzero(window) < window.size*.1:
                    pass
                else:
                    center = window[width, width]
                    ls = window.shape[0]
                    p2LL = center/np.mean(window[ls-1-ls//4:ls, :1+ls//4])
                    if positive and p2LL < 0.1:
                        pass
                    else:
                        if np.all(np.isfinite(window)):
                            ##########求x的atac
                            window_x = getbigwig(files, chromname, (x - width) * resou, (x + width + 1) * resou)
                            window_x = window_x.reshape(23, 10000)
                            window_x = [window_x.mean(axis=1)]  # 行
                            #####求y的atac
                            window_y = getbigwig(files, chromname, (y - width) * resou, (y + width + 1) * resou)
                            window_y = window_y.reshape(23, 10000)
                            window_y = [window_y.mean(axis=1)]
                            window_atac = np.dot(np.transpose(window_x), window_y)
                            if not positive:
                                negcount += 1
                            if negcount >= stop:
                                raise StopIteration
                            yield window
            except:
                pass

def generateATAC1(Matrix, coords, chromname, files, resou,  width=11, lower=1, positive=True, stop=5000):
    """
    Generate training set
    :param coords: List of tuples containing coord bins
    :param width: Distance added to center. width=5 makes 11x11 windows
    :return: yields paired positive/negative samples for training
    """
    w2 = int(width//2)
    negcount = 0
    for c in coords:
        x, y = c[0], c[1]
        if y-x < lower:
            pass
        else:
            try:
                window = Matrix[x-width:x+width+1,
                                y-width:y+width+1].toarray()
                if window.size!= 23*23:
                    continue
                if np.count_nonzero(window) < window.size*.1:
                    pass
                else:
                    center = window[width, width]
                    ls = window.shape[0]
                    p2LL = center/np.mean(window[ls-1-ls//4:ls, :1+ls//4])
                    if positive and p2LL < 0.1:
                        pass
                    else:
                        if np.all(np.isfinite(window)):
                            ##########求x的atac
                            window_x = getbigwig(files, chromname, (x - width) * resou, (x + width + 1) * resou)
                            window_x = window_x.reshape(23, 10000)
                            window_x = [window_x.mean(axis=1)]  # 行
                            #####求y的atac
                            window_y = getbigwig(files, chromname, (y - width) * resou, (y + width + 1) * resou)
                            window_y = window_y.reshape(23, 10000)
                            window_y = [window_y.mean(axis=1)]
                            window_atac = np.dot(np.transpose(window_x), window_y)
                            if not positive:
                                negcount += 1
                            if negcount >= stop:
                                raise StopIteration
                            yield window_atac
            except:
                pass


def generateHistone(Matrix, coords, chromname, files, resou,  width=11, lower=1, positive=True, stop=5000):
    """
    Generate training set
    :param coords: List of tuples containing coord bins
    :param width: Distance added to center. width=5 makes 11x11 windows
    :return: yields paired positive/negative samples for training
    """
    w2 = int(width//2)
    negcount = 0
    for c in coords:
        x, y = c[0], c[1]
        if y-x < lower:
            pass
        else:
            try:
                window = Matrix[x-width:x+width+1,
                                y-width:y+width+1].toarray()
                if window.size!= 23*23:
                    continue
                if np.count_nonzero(window) < window.size*.1:
                    pass
                else:
                    center = window[width, width]
                    ls = window.shape[0]
                    p2LL = center/np.mean(window[ls-1-ls//4:ls, :1+ls//4])
                    if positive and p2LL < 0.1:
                        pass
                    else:
                        if np.all(np.isfinite(window)):
                            ##########求x的histone
                            window_x = getbigwig(files, chromname, (x - width) * resou, (x + width + 1) * resou)
                            window_x = window_x.reshape(23, 10000)
                            window_x = [window_x.mean(axis=1)]  # 行
                            #####求y的histone
                            window_y = getbigwig(files, chromname, (y - width) * resou, (y + width + 1) * resou)
                            window_y = window_y.reshape(23, 10000)
                            window_y = [window_y.mean(axis=1)]
                            window_histone = np.dot(np.transpose(window_x), window_y)

                            if not positive:
                                negcount += 1
                            if negcount >= stop:
                                raise StopIteration
                            yield window_histone
            except:
                pass
