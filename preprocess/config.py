class Config(object):
    def __init__(self):
        self.GRAPH_IDX = 7
        self.USE_GRAPH = [1, 6, 10]
        self.VAL_GRAPH = 1
        self.TEST_GRAPH = 10
        # for split otus
        self.KMER_LENGH = 7
        self.ORIGIN_FA_FILE = 'zotus{}.fa'.format(self.GRAPH_IDX)

        self.DATA_DIR = '/home/keyan/NewDisk/gut_data/graphsage/graph{}'.format(self.GRAPH_IDX)
        # self.DATA_DIR = '/home/keyan/threepoors/MyGraphSAGE/example_data/new_graph/graph{}'.format(self.GRAPH_IDX)
        self.SPARCC_FILE = 'zotutable{}.out'.format(self.GRAPH_IDX)
        self.KO_PREDICTED_FILE = 'KO_predicted{}.tsv'.format(self.GRAPH_IDX)
        self.LSA_EDGE_FILE = 'OTUs_edge_lsa.txt'

        self.SPARCC_THRESH = [0.3, 0.7, 0.9]
        self.CLASS_MAP_TYPE = ['20_nothresh', '20_thresh', 'filted_nothresh', 'filted_thresh']
        # for output
        self.OUTPUT_DIR = '/home/keyan/threepoors/MyGraphSAGE/example_data/new_graph/graph{}'.format(self.GRAPH_IDX)
        self.ID_MAP_FILE = 'otu-id_map.json'
        self.CLASS_MAP_FILE = 'otu-class_map.json'
        self.GRAPH_FILE = 'otu-G.json'
        self.FEATURE_FILE = 'otu-feats.npy'

        self.FILTED_THRESH = [340, 400, 1100, 950, 950, 2050, 1700]
        self.MERGED_OUTPUT_DIR = '/home/keyan/threepoors/MyGraphSAGE/example_data/train6_val1_test10'

    def set_graph_idx(self, graph_idx):
        self.GRAPH_IDX = graph_idx
        self.ORIGIN_FA_FILE = 'zotus{}.fa'.format(self.GRAPH_IDX)
        self.DATA_DIR = '/home/keyan/NewDisk/gut_data/graphsage/graph{}'.format(self.GRAPH_IDX)
        self.SPARCC_FILE = 'zotutable{}.out'.format(self.GRAPH_IDX)
        self.KO_PREDICTED_FILE = 'KO_predicted{}.tsv'.format(self.GRAPH_IDX)
        self.LSA_EDGE_FILE = 'OTUs_edge_lsa.txt'
        self.OUTPUT_DIR = '/home/keyan/threepoors/MyGraphSAGE/example_data/new_graph/graph{}'.format(self.GRAPH_IDX)

    def set_class_map_suffix(self, suffix):
        if 'filted' in suffix:
            self.KO_PREDICTED_FILE = 'KO_predicted{}_filted.tsv'.format(self.GRAPH_IDX)
        else:
            self.KO_PREDICTED_FILE = 'KO_predicted{}.tsv'.format(self.GRAPH_IDX)
        self.CLASS_MAP_FILE = 'otu-class_map_{}.json'.format(suffix)

    def set_graph_suffix(self, suffix):
        self.GRAPH_FILE = 'otu-G_{}.json'.format(suffix)


cfg = Config()
