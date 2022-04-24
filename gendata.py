#!/usr/bin/env python
import pathlib
import straw
import argparse
import numpy as np
from dataUtils import *
from utils import *

def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="generate positive and negative samples ")

    parser.add_argument("-p", dest="path", type=str, default=None,
                        help="Path to a .cool URI string or a .hic file.")
    parser.add_argument("--balance", dest="balance", default=True,
                        help="Whether or not using the ICE/KR-balanced matrix.")
    parser.add_argument('-b', '--bedpe',
                          help='''Path to the bedpe file containing positive training set.''')
    parser.add_argument("-a", dest="bigwig", type=str, default=None,
                        help="Path to the chromatin accessibility data which is a bigwig file ")
    parser.add_argument("-o", dest="out_dir", default='./data/', help="Folder path to store results.")
    parser.add_argument('-l', '--lower', type=int, default=2,
                        help='''Lower bound of distance between loci in bins (default 2).''')
    parser.add_argument('-u', '--upper', type=int, default=300,
                        help='''Upper bound of distance between loci in bins (default 300).''')
    parser.add_argument('-w', '--width', type=int, default=11,
                        help='''Number of bins added to center of window. 
                                default width=11 corresponds to 23*23 windows''')
    parser.add_argument('-r', '--resolution',
                        help='Resolution in bp, default 10000',
                        type=int, default=10000)


    return parser.parse_args()


def main():
    args = get_args()
    np.seterr(divide='ignore', invalid='ignore')

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)#创建目录

    # more robust to check if a file is .hic
    hic_info = read_hic_header(args.path)

    if hic_info is None:
        hic = False
    else:
        hic = True

    coords = parsebed(args.bedpe, lower=2, res=args.resolution)#取标记的每对loop的start位置，并在每条染色体上进行排序，生成一个字典包含23条染色体上的所有正样本的两个start位置（除以了10000）
    kde, lower, long_start, long_end = learn_distri_kde(coords)
    # ATAC_path =

    if not hic:
        import cooler
        Lib = cooler.Cooler(args.path)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = get_hic_chromosomes(args.path, args.resolution)

    # train model per chromosome
    positive_class = {}
    positive_atac = {}
    negative_atac = {}
    negative_class = {}
    positive_labels = {}
    negative_labels = {}
    for key in chromosomes:
        # if key != '17':
        #     continue
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        print('collecting from {}'.format(key))
        if not hic:
            X = Lib.matrix(balance=True,
                           sparse=True).fetch(key).tocsr()
        else:
            if args.balance:
                X = csr_contact_matrix(
                    'KR', args.path, key, key, 'BP', args.resolution)
            else:
                X = csr_contact_matrix(
                    'NONE', args.path, key, key, 'BP', args.resolution)
        clist = coords[chromname]

        try:
            #####generate positive samples
            positive_class[chromname] = np.array([f.tolist() for f in generateATAC(
                X, clist, chromname, files=args.bigwig, resou=args.resolution,  width=args.width)])
            print(len(positive_class[chromname]))
            positive_class[chromname] = positive_class[chromname].reshape((positive_class[chromname].shape[0], 1,
                                                                               positive_class[chromname].shape[1],
                                                                               positive_class[chromname].shape[2]))
            positive_atac[chromname] = np.array([f.tolist() for f in generateATAC1(
                X, clist, chromname, files=args.bigwig, resou=args.resolution,  width=args.width)])
            print(len(positive_atac[chromname]))
            positive_atac[chromname] = positive_atac[chromname].reshape((positive_atac[chromname].shape[0], 1,
                                                                         positive_atac[chromname].shape[1],
                                                                         positive_atac[chromname].shape[2]))
            positive_num = len((positive_class[chromname]))
            positive_labels[chromname] = np.ones(positive_num).tolist()

            neg_coords = negative_generating(X, kde, clist, lower, long_start, long_end)
            stop = len(clist)
            negative_class[chromname] = np.array([f.tolist() for f in generateATAC(
                X, neg_coords, chromname, files=args.bigwig, resou=args.resolution, width=args.width, positive=False,stop=stop)])
            print(len(negative_class[chromname]))
            negative_class[chromname] = negative_class[chromname].reshape((negative_class[chromname].shape[0], 1,
                                                                           negative_class[chromname].shape[1], negative_class[chromname].shape[2]))
            negative_atac[chromname] = np.array([f.tolist() for f in generateATAC1(
                X, neg_coords, chromname, files=args.bigwig, resou=args.resolution, width=args.width, positive=False,stop=stop)])
            print(len(negative_atac[chromname]))
            negative_atac[chromname] = negative_atac[chromname].reshape((negative_atac[chromname].shape[0], 1,
                                                                         negative_atac[chromname].shape[1],
                                                                         negative_atac[chromname].shape[2]))

            negative_num = len(negative_class[chromname])
            negative_labels[chromname] = np.zeros(negative_num).tolist()

            np.savez(args.out_dir+'%s_positive.npz' % chromname, data=positive_class[chromname],
                     atac=positive_atac[chromname],label=positive_labels[chromname])
            np.savez(args.out_dir+'%s_negative.npz' % chromname, data=negative_class[chromname],
                     atac=negative_atac[chromname],label=negative_labels[chromname])
        except:
            print(chromname, ' failed to gather fts')

if __name__ == "__main__":
    main()
