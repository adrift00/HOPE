## HOPE: Hierarchical micrObial functions Prediction by graph aggregated Embedding

### Overview

This directory contains code necessary to run the HOPE algorithm.
HOPE (Hierarchical micrObial functions Prediction by graph aggregated Embedding) is a novel method that incorporates k-mer compositions of microbial sequences and topology of microbial networks to predict microbial functions. See our [paper]() for details on the algorithm.

If you make use of this code or the HOPE algorithm in your work, please cite the following paper:


### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn, and networkx are required (but networkx must be <=1.11). You can install all the required packages using the following command:

	$ pip install -r requirements.txt


### Running the code

The example_unsupervised.sh and example_supervised.sh files contain example usages of the code, which use the unsupervised and supervised variants of GraphSage, respectively.

If your benchmark/task does not require generalizing to unseen data, we recommend you try setting the "--identity_dim" flag to a value in the range [64,256].
This flag will make the model embed unique node ids as attributes, which will increase the runtime and number of parameters but also potentially increase the performance.
Note that you should set this flag and *not* try to pass dense one-hot vectors as features (due to sparsity).
The "dimension" of identity features specifies how many parameters there are per node in the sparse identity-feature lookup table.

Note that example_unsupervised.sh sets a very small max iteration number, which can be increased to improve performance.
We generally found that performance continued to improve even after the loss was very near convergence (i.e., even when the loss was decreasing at a very slow rate).

*Note:* For the PPI data, and any other multi-ouput dataset that allows individual nodes to belong to multiple classes, it is necessary to set the `--sigmoid` flag during supervised training. By default the model assumes that the dataset is in the "one-hot" categorical setting.


#### Construct graph
The code of consturct graph is in the preprocess directory.
It take the data files in the  as the input. See the [PREPROCESS.md](preprocess/PREPROCESS.md) for detail use. The output file are as follows:

* otu-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* otu-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* otu-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* otu-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.

To run the model on a new dataset, you need to make data files in the format described above and run the code in the proprocess 
directory. If you just want to run the model, there are constructed graph file in the , The example_data subdirectory contains a small example of , you can skip the preprocess step and train the model directly.

#### Train and eval model

The experimental dataset is divided into three distinct parts, training set, validation set, and test set. We split 20% of all OTUs into a validation set and design an 8-fold cross validation on the remaining 80% of all OTUs. The cross validation is applied to learn the appropriate parameters in the weight matrices and select the appropriate hyperparameters. 
To avoid over fitting, dropout is set as = 0.4 and is adopted. All of the experiments use rectified linear units (ReLU) as activation functions. The experiments are run on a single machine with 4 NVIDIA GeForce GTX1080 TI with CUDA Version 10.2, Intel(R) Xeon(R) CPU (E5-2620 v4 @ 2.10GHz) and 128Gb of RAM.


#### Model variants
The user must also specify a --model, the variants of which are described in detail in the paper:
* graphsage_mean -- GraphSage with mean-based aggregator
* graphsage_seq -- GraphSage with LSTM-based aggregator
* graphsage_maxpool -- GraphSage with max-pooling aggregator (as described in the NIPS 2017 paper)
* graphsage_meanpool -- GraphSage with mean-pooling aggregator (a variant of the pooling aggregator, where the element-wie mean replaces the element-wise max).
* gcn -- GraphSage with GCN-based aggregator
* n2v -- an implementation of [DeepWalk](https://arxiv.org/abs/1403.6652) (called n2v for short in the code.)

#### Logging directory
Finally, a --base_log_dir should be specified (it defaults to the current directory).
The output of the model and log files will be stored in a subdirectory of the base_log_dir.
The path to the logged data will be of the form `<sup/unsup>-<data_prefix>/graphsage-<model_description>/`.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored in a numpy formated file named val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data when running with the unsupervised variant).

### Acknowledgements

The original version of this code base was originally forked from https://github.com/williamleif/GraphSAGE, and we owe many thanks to  William L. Hamilton for making his code available.
Please see the [paper](https://arxiv.org/pdf/1706.02216.pdf) for funding details and additional (non-code related) acknowledgements.