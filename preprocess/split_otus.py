import io
import os
import numpy as np
from config import cfg


def split_otus():
    splited_otus = None
    with open(os.path.join(cfg.DATA_DIR, cfg.ORIGIN_FA_FILE), 'r') as f:
        otus = f.read()
        splited_otus = otus.split('>')
    splited_otus_dir = os.path.join(cfg.DATA_DIR, 'split_otus')
    if not os.path.isdir(splited_otus_dir):
        os.makedirs(splited_otus_dir)
    with open(os.path.join(splited_otus_dir, 'otus_path.txt'), 'w') as otu_path:
        for otu in splited_otus:
            if otu == '':
                continue
            single_otu_path = os.path.join(splited_otus_dir, (otu.split()[0] + '.fasta'))
            otu_path.write(single_otu_path + '\n')
            with open(single_otu_path, 'w') as wf:
                wf.write('>' + otu)


def write_kmer_out(k):
    for idx, line in enumerate(open(os.path.join(cfg.DATA_DIR, 'split_otus', 'otus_path.txt')).readlines()):
        line = line.strip()
        kmers_dir = os.path.join(cfg.DATA_DIR, 'kmers')
        if not os.path.isdir(kmers_dir):
            os.makedirs(kmers_dir)
        outpath = os.path.join(kmers_dir, ('otu_kmer_' + str(idx + 1) + '.txt'))
        if os.path.exists(outpath):
            os.remove(outpath)
        if os.path.exists(outpath):
            os.remove(outpath)
        command = "/home/keyan/threepoors/dna-utils-master/kmer_total_count -i {} -k {} -l -n > {}".format(
            line, k, outpath)
        os.system(command)


def kmer_seq_to_idx(seq):
    k = len(seq)
    seq_dict = {'A': 0,
                'C': 1,
                'G': 2,
                'T': 3}
    idx = 0
    for i, c in enumerate(seq):
        idx += seq_dict[c] * 4 ** (k - i - 1)
    return idx


def calc_kmer_feat(k):
    kmer_out_dir = os.path.join(cfg.DATA_DIR, 'kmers')
    kmer_out_path = os.listdir(kmer_out_dir)
    kmer_feat = {}
    for path in kmer_out_path:
        path = os.path.join(kmer_out_dir, path)
        otu_idx = path.split('_')[-1].split('.')[0]
        otu_idx = int(otu_idx)
        key = 'OTU_'+str(otu_idx)
        feat = [0]*(4**k)
        with open(path, 'r') as kmer_otu:
            for line in kmer_otu.readlines():
                seq = line.split()[0]
                num = int(line.split()[1].strip())
                idx = kmer_seq_to_idx(seq)
                feat[idx] = num
            kmer_feat[key] = feat
    return kmer_feat


def calc_kmer_feat_merged(k):
    kmer_feat = {}
    for graph_idx in cfg.USE_GRAPH:
        cfg.set_graph_idx(graph_idx)
        kmer_out_dir = os.path.join(cfg.DATA_DIR, 'kmers')
        kmer_out_path = os.listdir(kmer_out_dir)
        for path in kmer_out_path:
            path = os.path.join(kmer_out_dir, path)
            otu_idx = path.split('_')[-1].split('.')[0]
            otu_idx = int(otu_idx)
            key = 'Graph{}_OTU_{}'.format(graph_idx, otu_idx)
            feat = [0]*(4**k)
            with open(path, 'r') as kmer_otu:
                for line in kmer_otu.readlines():
                    seq = line.split()[0]
                    num = int(line.split()[1].strip())
                    idx = kmer_seq_to_idx(seq)
                    feat[idx] = num
                kmer_feat[key] = feat
    return kmer_feat

# def read_kmer_feat():
#     kmer_dir='/home/keyan/NewDisk/kmer_counting_7'
#     kmer_path=os.listdir(kmer_dir)
#     kmer_feats={}
#     for path in kmer_path:
#         otu_idx = path.split('_')[-1].split('.')[0]
#         otu_idx = int(otu_idx)
#         key = 'OTU_' + str(otu_idx)
#         path = os.path.join(kmer_dir, path)
#         feat=[]
#         with open(path,'r') as kmer:
#             for line in kmer.readlines():
#                 feat.append(int(line))
#         kmer_feats[key]=feat
#     return kmer_feats


if __name__ == '__main__':
    cfg.set_graph_idx(11)
    split_otus()
    write_kmer_out(cfg.KMER_LENGH)
    calc_kmer_feat(cfg.KMER_LENGH)
