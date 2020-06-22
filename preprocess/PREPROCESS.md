# preprocess

### Construct single graph
If you want to construct a graph for a dataset, you can do as follows:
* Modify the configs in the config.py, especially the DATA_DIR, OUTPUT_DIR property.
* Run the split_otus.py for split otus and calculate kmer counting.
* Run the create_graph.py for create the graph files.

*Note:* The code also support construct graph for several graphs in a batch. You can place the origin data in the directories as graph1, graph2 and so on. And use a loop for constructing all the graphs.  


### Construct multi graphs
If you want to consturct a big graph for several dataset, you can run the merged_graph_new.py. The output file of the code are as follows:
* Modify the configs in config.py, especially the MERGED_OUTPUT_DIR, USE_GRAPH, VAL_GRAPH, TEST_GRAPH.
* Run the create_graph.py for each dataset to create the id_map.
* Run the merge_graphs.py for merged the graphs.

