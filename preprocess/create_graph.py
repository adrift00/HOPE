import os
import json
import random
import pandas as pd
import numpy as np
from split_otus import split_otus, write_kmer_out, calc_kmer_feat
from config import cfg


def create_lsa_from_sparcc(sparcc_thresh):
    path = os.path.join(cfg.DATA_DIR, cfg.SPARCC_FILE)
    sparcc_data = pd.read_csv(path, sep='\t', index_col=0)
    cols = sparcc_data.columns.values.tolist()
    rows = sparcc_data.index.values.tolist()
    col_otu_idx = list(map(lambda x: int(x[4:]), cols))
    row_otu_idx = list(map(lambda x: int(x[4:]), rows))
    sparcc_mat = sparcc_data.to_numpy()
    edge_idx = np.where((sparcc_mat >= sparcc_thresh)|(sparcc_mat<=-sparcc_thresh) )
    row_otu_idx = np.array(row_otu_idx)
    col_otu_idx = np.array(col_otu_idx)
    otu_src_idx = row_otu_idx[edge_idx[0]]
    otu_dst_idx = col_otu_idx[edge_idx[1]]
    # write like lsa format
    edge_data = pd.DataFrame({'index1': otu_src_idx, 'index2': otu_dst_idx})
    edge_data.to_csv(os.path.join(cfg.DATA_DIR, cfg.LSA_EDGE_FILE), sep='\t')


def create_id_map():
    # to avoid otu not in kmer feats
    feats = calc_kmer_feat(cfg.KMER_LENGH)
    path = os.path.join(cfg.DATA_DIR, cfg.SPARCC_FILE)
    sparcc_data = pd.read_csv(path, sep='\t', index_col=0)
    cols = sparcc_data.columns.values.tolist()
    otu_idx = list(map(lambda x: int(x[4:]), cols))
    nodes=list(set(otu_idx))

    random.shuffle(nodes)
    id_map = {}
    i = 0
    for idx in nodes:
        key = 'OTU_'+str(idx)
        if key not in feats.keys():
            continue
        id_map[key] = i
        i += 1
    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    with open(os.path.join(cfg.OUTPUT_DIR, cfg.ID_MAP_FILE), 'w') as f:
        json.dump(id_map, f)


def load_id_map():
    with open(os.path.join(cfg.OUTPUT_DIR, cfg.ID_MAP_FILE), 'r') as f:
        map = json.load(f)
    return map


def create_edge_from_lsa():
    path = os.path.join(cfg.DATA_DIR, cfg.LSA_EDGE_FILE)
    lsa_data = pd.read_csv(path, sep='\t')
    src_nodes = lsa_data.loc[:, 'index1'].values.tolist()
    dst_nodes = lsa_data.loc[:, 'index2'].values.tolist()
    edges = []
    id_map = load_id_map()
    for src_node, dst_node in zip(src_nodes, dst_nodes):
        src_node = 'OTU_' + str(src_node)
        dst_node = 'OTU_'+str(dst_node)
        # to avoid not in
        if src_node not in id_map.keys() or dst_node not in id_map.keys():
            continue
        src_idx = id_map[src_node]
        dst_idx = id_map[dst_node]
        edges.append([src_idx, dst_idx])
    return edges


def create_graph(id2idx, edges):
    graph = {}
    graph["directed"] = False
    graph['graph'] = {"name": "disjoint_union( ,  )"}
    nodes = []
    train_num = int(len(id2idx) * 0.7)
    val_num = int(len(id2idx) * 0.2)
    # features=read_kmer_feat()
    # TODO: the label is not used?
    # labels=create_class_map()
    for i, (id, idx) in enumerate(id2idx.items()):
        is_test = False
        is_val = False
        if i > train_num and i < (train_num + val_num):
            is_val = True
        elif i >= (train_num + val_num):
            is_test = True
        # feature = features[id]
        # label=labels[id]
        node = {"test": is_test,
                'id': id,
                'feature': None,
                'val': is_val,
                'label': None
                }
        nodes.append(node)
    links = []
    for src, dst in edges:
        link = {'test_removed': False,
                'train_removed': False,
                'target': dst,
                'source': src
                }
        links.append(link)
    graph['nodes'] = nodes
    graph['links'] = links
    graph['multigraph'] = False
    with open(os.path.join(cfg.OUTPUT_DIR, cfg.GRAPH_FILE), 'w') as f:
        json.dump(graph, f)


def create_features():
    features = calc_kmer_feat(cfg.KMER_LENGH)
    id_map = load_id_map()
    filter_features = []
    for id in id_map.keys():
        filter_features.append(features[id])
    np.save(os.path.join(cfg.OUTPUT_DIR, cfg.FEATURE_FILE), filter_features)


def create_class_map(low_thresh=None,high_thresh=None):
    id_map = load_id_map()
    path = os.path.join(cfg.DATA_DIR, cfg.KO_PREDICTED_FILE)
    # need index_col parameter
    label_data = pd.read_csv(path, sep='\t', index_col=0)
    ids = list(id_map.keys())
    # change Zotu to OTU
    indexs = label_data.index.values
    new_indexs = list(map(lambda x: 'OTU_'+x[4:], indexs))
    label_data.index = new_indexs

    labels = label_data.loc[ids, :]
    labels = labels.loc[:, ~((labels == 0).all())]
    #save ko_name
    ko_names=labels.columns.values
    #
    labels = np.array(labels)
    labels[labels > 1] = 1

    if low_thresh is not None:
        label_sum = labels.sum(axis=0)
        idx = np.where(label_sum > low_thresh)[0]
        labels = labels[:, idx]
    #
    ko_names=ko_names[idx]
    ko_names=pd.DataFrame(ko_names)
    ko_names.to_csv('Graph10_ko_names.csv')
    # filted_labels=pd.DataFrame(labels,index=new_indexs,columns=ko_names)
    # filted_labels.to_csv('filted_ko.csv')
    #
    if high_thresh is not None:
        label_sum = labels.sum(axis=0)
        idx = np.where(label_sum < cfg.FILTED_THRESH[cfg.GRAPH_IDX-1])[0]
        labels = labels[:, idx]

    class_map = {}
    for (i, id) in enumerate(ids):
        class_map[id] = labels[i].tolist()
    with open(os.path.join(cfg.OUTPUT_DIR, cfg.CLASS_MAP_FILE), 'w') as f:
        json.dump(class_map, f)
    return class_map


def preprocess_data():
    create_lsa_from_sparcc()
    create_id_map()
    id2idx = load_id_map()
    edges = create_edge_from_lsa()
    create_graph(id2idx, edges)
    create_class_map()
    create_features()


if __name__ == '__main__':
    # for graph_idx in range(1,8):
    #     cfg.set_graph_idx(graph_idx)
    #     create_id_map()
    #     create_features()
    #     for sparcc_thresh in cfg.SPARCC_THRESH:
    #         cfg.set_graph_suffix(sparcc_thresh)
    #         create_lsa_from_sparcc(sparcc_thresh=sparcc_thresh)
    #         id2idx=load_id_map()
    #         edges=create_edge_from_lsa()
    #         create_graph(id2idx,edges)
    #     # 4 class map type
    #     cfg.set_class_map_suffix(cfg.CLASS_MAP_TYPE[0])
    #     create_class_map(20,None)
    #     cfg.set_class_map_suffix(cfg.CLASS_MAP_TYPE[1])
    #     create_class_map(20,cfg.FILTED_THRESH[graph_idx-1])
    #     cfg.set_class_map_suffix(cfg.CLASS_MAP_TYPE[2])
    #     create_class_map(None,None)
    #     cfg.set_class_map_suffix(cfg.CLASS_MAP_TYPE[3])
    #     create_class_map(None,cfg.FILTED_THRESH[graph_idx-1])

    cfg.set_graph_idx(10)
    create_id_map()
    print('create id map finished')
    create_features()
    print('create feature finished')
    cfg.set_graph_suffix(0.9)
    create_lsa_from_sparcc(0.9)
    id2idx = load_id_map()
    edges = create_edge_from_lsa()
    create_graph(id2idx, edges)
    print('create graph finished')
    cfg.set_class_map_suffix('20_nothresh')
    create_class_map(20,None)
    print('create class map finished')
    


        

            