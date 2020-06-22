import os
import json
import random
import pandas as pd
import numpy as np
from split_otus import calc_kmer_feat, calc_kmer_feat_merged, split_otus, write_kmer_out
from config import cfg


def create_lsa_from_sparcc(sparcc_thresh):
    begin_idx = 0
    for graph_idx in cfg.USE_GRAPH:
        cfg.set_graph_idx(graph_idx)
        path = os.path.join(cfg.DATA_DIR, cfg.SPARCC_FILE)
        sparcc_data = pd.read_csv(path, sep='\t', index_col=0)
        cols = sparcc_data.columns.values.tolist()
        rows = sparcc_data.index.values.tolist()
        col_otu_idx = list(map(lambda x: int(x[4:]), cols))
        row_otu_idx = list(map(lambda x: int(x[4:]), rows))
        sparcc_mat = sparcc_data.to_numpy()
        edge_idx = np.where(sparcc_mat >= sparcc_thresh)  # 边选择
        # edge_idx = np.where((sparcc_mat >= sparcc_thresh)|(sparcc_mat<=-sparcc_thresh) )
        row_otu_idx = np.array(row_otu_idx)
        col_otu_idx = np.array(col_otu_idx)
        otu_src_idx = ['Graph{}_{}'.format(graph_idx, idx) for idx in row_otu_idx[edge_idx[0]]]
        otu_dst_idx = ['Graph{}_{}'.format(graph_idx, idx) for idx in row_otu_idx[edge_idx[1]]]
        begin_idx += len(cols)
        if graph_idx == 1:
            edge_data = pd.DataFrame.from_dict({'index1': otu_src_idx, 'index2': otu_dst_idx})
        else:
            new_edge_data = pd.DataFrame.from_dict({'index1': otu_src_idx, 'index2': otu_dst_idx})
            edge_data = pd.concat([edge_data, new_edge_data])
    # write like lsa format
    edge_data.to_csv(os.path.join(cfg.MERGED_OUTPUT_DIR, cfg.LSA_EDGE_FILE), sep='\t')


def create_id_map():
    # to avoid otu not in kmer feats
    id_map = {}
    i = 0
    for graph_idx in cfg.USE_GRAPH:
        cfg.set_graph_idx(graph_idx)
        feats = calc_kmer_feat(cfg.KMER_LENGH)
        path = os.path.join(cfg.DATA_DIR, cfg.SPARCC_FILE)
        sparcc_data = pd.read_csv(path, sep='\t', index_col=0)
        cols = sparcc_data.columns.values.tolist()
        otu_idx = list(map(lambda x: int(x[4:]), cols))
        nodes = list(set(otu_idx))
        for idx in nodes:
            key = 'Graph{}_OTU_{}'.format(cfg.GRAPH_IDX, idx)
            splited_key = key.split('_')
            otu_name = splited_key[1]+'_'+splited_key[2]
            if otu_name not in feats.keys():  # delete the Graph{}_ to compare
                continue
            id_map[key] = i
            i += 1
    with open(os.path.join(cfg.MERGED_OUTPUT_DIR, cfg.ID_MAP_FILE), 'w') as f:
        json.dump(id_map, f)


def load_id_map():
    with open(os.path.join(cfg.MERGED_OUTPUT_DIR, cfg.ID_MAP_FILE), 'r') as f:
        map = json.load(f)
    return map


def load_old_id_map():
    with open(os.path.join(cfg.OUTPUT_DIR, cfg.ID_MAP_FILE), 'r') as f:
        map = json.load(f)
    return map


def create_edge_from_lsa():
    path = os.path.join(cfg.MERGED_OUTPUT_DIR, cfg.LSA_EDGE_FILE)
    lsa_data = pd.read_csv(path, sep='\t')
    src_nodes = lsa_data.loc[:, 'index1'].values.tolist()
    dst_nodes = lsa_data.loc[:, 'index2'].values.tolist()
    edges = []
    id_map = load_id_map()
    for src_node, dst_node in zip(src_nodes, dst_nodes):
        graph, src = src_node.split('_')
        src_node = '{}_OTU_{}'.format(graph, src)
        graph, dst = dst_node.split('_')
        dst_node = '{}_OTU_{}'.format(graph, dst)
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
    for i, (id, idx) in enumerate(id2idx.items()):  # id：OTU1 idx：1
        is_test = False
        is_val = False
        graph_name = id.split('_')[0]
        if graph_name == 'Graph'+str(cfg.VAL_GRAPH):
            is_val = True
        elif graph_name == 'Graph'+str(cfg.TEST_GRAPH):
            is_test = True
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
    with open(os.path.join(cfg.MERGED_OUTPUT_DIR, cfg.GRAPH_FILE), 'w') as f:
        json.dump(graph, f)


def create_features():
    id_map = load_id_map()
    features = calc_kmer_feat_merged(cfg.KMER_LENGH)
    filter_features = []
    for id in id_map.keys():
        filter_features.append(features[id])
    np.save(os.path.join(cfg.MERGED_OUTPUT_DIR, cfg.FEATURE_FILE), filter_features)


def create_class_map(class_type):
    class_map = {}
    merged_id_map = load_id_map()
    ko_names = []
    for graph_idx in cfg.USE_GRAPH:
        cfg.set_graph_idx(graph_idx)
        old_id_map = load_old_id_map()
        path = os.path.join(cfg.DATA_DIR, cfg.KO_PREDICTED_FILE)
        # need index_col parameter
        label_data = pd.read_csv(path, sep='\t', index_col=0)
        ids = list(old_id_map.keys())
        # change Zotu to OTU
        indexs = label_data.index.values
        new_indexs = list(map(lambda x: 'OTU_'+x[4:], indexs))
        label_data.index = new_indexs

        labels = label_data.loc[ids, :]
        labels = labels.loc[:, ~((labels == 0).all())]
        ko_name = labels.columns.values
        labels = np.array(labels)
        labels[labels > 1] = 1

        if 'filted' not in class_type:
            label_sum = labels.sum(axis=0)
            idx = np.where(label_sum > 20)[0]
            labels = labels[:, idx]
            ko_name = ko_name[idx]
        ko_names.append(ko_name)
    ko_union = []
    for i in range(len(cfg.USE_GRAPH)):
        if i == 0:
            ko_union = ko_names[0]
        else:
            ko_union = list(set(ko_union) & set(ko_names[i]))

    for graph_idx in cfg.USE_GRAPH:
        cfg.set_graph_idx(graph_idx)
        old_id_map = load_old_id_map()
        path = os.path.join(cfg.DATA_DIR, cfg.KO_PREDICTED_FILE)
        # need index_col parameter
        label_data = pd.read_csv(path, sep='\t', index_col=0)
        ids = list(old_id_map.keys())
        # change Zotu to OTU
        indexs = label_data.index.values
        new_indexs = list(map(lambda x: 'OTU_'+x[4:], indexs))
        label_data.index = new_indexs

        labels = label_data.loc[ids, :]
        labels = labels.loc[:, ~((labels == 0).all())]
        labels = labels.loc[:, ko_union]  # get the union of all dataset
        labels = np.array(labels)
        print(graph_idx)
        labels[labels > 1] = 1

        for (i, id) in enumerate(ids):
            id = 'Graph{}_{}'.format(graph_idx, id)
            class_map[id] = labels[i].tolist()

    with open(os.path.join(cfg.MERGED_OUTPUT_DIR, cfg.CLASS_MAP_FILE), 'w') as f:
        json.dump(class_map, f, indent=4)
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
    if not os.path.isdir(cfg.MERGED_OUTPUT_DIR):
        os.makedirs(cfg.MERGED_OUTPUT_DIR)
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
    create_class_map('20_nothresh')
    print('create class map finished')
