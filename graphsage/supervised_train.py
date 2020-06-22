from __future__ import division
from __future__ import print_function

import os
import time
import json
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics
from tensorflow.python import debug as tf_dbg
import tensorboard
from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data
from graphsage.inits import glorot
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from preprocess import generate_jsons

# for shap
import shap
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

# for construct data
from preprocess.create_jsons import create_class_map
from preprocess.create_jsons import create_id_map_lsa
from preprocess.create_jsons import load_id_map
from preprocess.create_jsons import create_edge_lsa
from preprocess.create_jsons import create_graph
from preprocess.create_jsons import create_features
from preprocess.create_jsons import generate_jsons

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')  # 必须要填的

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 3, 'number of epochs to train.')
flags.DEFINE_integer('epochs2', 5, 'number of epochs to train in hierarchy2.')
flags.DEFINE_integer('epochs3', 10, 'number of epochs to train in hierarchy3.')
flags.DEFINE_integer('epochs4', 15, 'number of epochs to train in hierarchy4.')
flags.DEFINE_float('dropout', 0., 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 128, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')  # 命令行参数也有
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 128, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 1, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")
flags.DEFINE_integer('hierarchy', 3, "The degree of hierarchical classes.")
#os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
#os.environ["CUDA_DEVICE_ORDER"] = "0"
GPU_MEM_FRACTION = 0.95

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    with open('y_pred.txt', 'w') as f:
       np.savetxt(f, y_pred, fmt='%d', delimiter=",")
    with open('y_true.txt', 'w') as f:
       np.savetxt(f, y_true, fmt='%d', delimiter=",")
    f1_spilt = metrics.f1_score(y_true, y_pred, average=None)
    #"""
    f = open('./f1_score.txt', 'a')
    f.write(str(f1_spilt)+'\n')
    f.close()
    #"""
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro"), metrics.f1_score(y_true, y_pred, average=None)

def calc_precision(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true.tolist(), y_pred.tolist())

def incremental_evaluate_for_each(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num,
                                                                                                 test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)

    def calc_f1_each(y_true, y_pred):
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        return metrics.f1_score(y_true, y_pred, average=None)
    f1_scores = calc_f1_each(labels, val_preds)
    with open('f1_scores.txt', 'w') as f:
        np.savetxt(f, f1_scores, fmt='%0.5f')
    return f1_scores

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                             feed_dict=feed_dict_val)
    mic, mac, ko_none = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num,
                                                                                                 test=test)
        node_outs_val = sess.run([model.preds, model.loss, model.node_preds],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    lis_preds = val_preds.tolist()
    lis_label = labels.tolist()
    OTU_mi = []
    for i in range(val_preds.shape[0]):
        OTU_f1 = calc_precision(labels[i, ...], val_preds[i, ...])
        OTU_mi.append(float(OTU_f1))

    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test), OTU_mi, f1_scores[2]

def y_ture_pre(sess, model, minibatch_iter, size, test=False):

    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num,
                                                                                                 test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    return  val_preds



# placeholder 占位符，可以理解成形参，由用户提供
def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),  # none代表可以是任意数
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def construct_class_numpy(class_map):
    count = 0
    for key in class_map:
        if count == 0:
            seq_class = class_map[key]
            count += 1
        else:
            c = np.array(class_map[key])
            seq_class = np.vstack((seq_class, c))
            count += 1
    return seq_class

def train(train_data, test_data=None):
    G = train_data[0]  # G 是一个Networkx里的对象，这几个都是经过load_data()处理过的
    features = train_data[1]
    id_map = train_data[2]
    class_map1 = train_data[4]
    class_map2 = train_data[5]
    class_map3 = train_data[6]
    dict_classmap = {0: class_map1, 1: class_map2, 2: class_map3, 3: class_map3}
    hierarchy = FLAGS.hierarchy
    features_shape1 = None
    a_class = construct_class_numpy(class_map1)
    b_class = construct_class_numpy(class_map2)
    c_class = construct_class_numpy(class_map3)
    a_class = tf.cast(a_class, tf.float32)
    b_class = tf.cast(b_class, tf.float32)
    c_class = tf.cast(c_class, tf.float32)

    num_class = []
#    for key in class_map.keys():
#        num_class = num_class.append(sum(class_map[key]))

    for hi_num in range(hierarchy):
        #tf.reset_default_graph()
        if hi_num == 0:
            class_map = class_map1
            features = features
            features_shape1 = features.shape[1]
            if features is not None:
                # pad with dummy zero vector
                features = np.vstack([features, np.zeros((features.shape[1],))])
            features = tf.cast(features, tf.float32)

        else:
            print("hierarchy %d finished" % (hi_num), end='\n\n')
            class_map = dict_classmap[hi_num]
            features = features2
            features = tf.cast(features, tf.float32)
            features = tf.concat([features, tf.zeros([1, features_shape1+num_classes])], axis=0)
            features_shape1 = features.shape[1]

        if hi_num == 0:
            if isinstance(list(class_map.values())[0], list):
                num_classes = len(list(class_map.values())[0])
            else:
                num_classes = len(set(class_map.values()))
        else:
            if isinstance(list(dict_classmap[hi_num].values())[0], list):
                num_classes = len(list(dict_classmap[hi_num].values())[0])
            else:
                num_classes = len(set(dict_classmap[hi_num].values()))

        """"" 
        if features is not None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])
        """""

        # features = tf.cast(features, tf.float32)
        # embeding_weight=tf.get_variable('emb_weights', [50, 128], initializer=tf.random_normal_initializer(),dtype=tf.float32)
        # features=tf.matmul(features,embeding_weight)
        context_pairs = train_data[3] if FLAGS.random_context else None
        placeholders = construct_placeholders(num_classes)
        minibatch = NodeMinibatchIterator(G,
                                          id_map,
                                          placeholders,
                                          class_map,
                                          num_classes,
                                          batch_size=FLAGS.batch_size,
                                          max_degree=FLAGS.max_degree,
                                          context_pairs=context_pairs)
        ##########
        with open('test_nodes.txt', 'w') as f:
            json.dump(minibatch.test_nodes, f)
        ###########
        if hi_num == 0:
            adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape, name='adj_info_ph')

        # 把adj_info设成Variable应该是因为在训练和测试时会改变adj_info的值，所以
        # 用Varible然后用tf.assign()赋值。
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        shap.initjs()
        if FLAGS.model == 'graphsage_mean':
            # Create model
            sampler = UniformNeighborSampler(adj_info)

            if FLAGS.samples_3 != 0:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                               SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]


            elif FLAGS.samples_2 != 0:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]


            else:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]


            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,  # 每一个的度
                                        layer_infos,
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True,
                                        concat=True,
                                        )


        elif FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="gcn",
                                         model_size=FLAGS.model_size,
                                        concat=False,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="seq",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True,
                                        concat=True)

        elif FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="maxpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True,
                                        concat=True)

        elif FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="meanpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True,
                                        concat=True)
        elif FLAGS.model == 'gat':
            sampler = UniformNeighborSampler(adj_info)
            # 建立两层网络 采样邻居、邻居个数、输出维度
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders, features,
                                        adj_info,
                                        minibatch.deg,
                                        concat=True,
                                        layer_infos=layer_infos,
                                        aggregator_type="gat",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True,
                                        )
        else:
            raise Exception('Error: model name unrecognized.')

        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
        config.allow_soft_placement = True

        # Initialize session

        sess = tf.Session(config=config)
        # sess = tf_dbg.LocalCLIDebugWrapperSession(sess)
        #merged = tf.summary.merge_all()  # 将所有东西保存到磁盘，可视化会用到
        #summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)  # 记录信息,可视化，可以用tensorboard查看

        # Init variables

        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
        #sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph2: minibatch2.adj})

        # Train model
        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []
        epoch_val_costs2 = []
        # 这里minibatch.adj和minibathc.test_adj的大小是一样的，只不过adj里面把不是train的值都变成一样
        # val在这里是validation的意思，验证
        train_adj_info = tf.assign(adj_info, minibatch.adj)  # tf.assign()是为一个tf.Variable赋值，返回值是一个Variable,是赋值后的值
        val_adj_info = tf.assign(adj_info, minibatch.test_adj)  # assign()是一个Opration，要用sess.run()才能执行

        it = 0
        train_loss = []
        val_loss = []
        train_f1_mics = []
        val_f1_mics = []
        loss_plt = []
        loss_plt2 = []
        trainf1mi = []
        trainf1ma = []
        valf1mi = []
        valf1ma = []
        iter_num = 0

        if hi_num == 0:
            epochs = FLAGS.epochs
        elif hi_num == 1:
            epochs = FLAGS.epochs2
        elif hi_num == 2:
            epochs = FLAGS.epochs3
        else:
            epochs = FLAGS.epochs4

        for epoch in range(epochs+1):
            if epoch < epochs:
                minibatch.shuffle()
                iter = 0
                print('Epoch: %04d' % (epoch + 1))
                epoch_val_costs.append(0)
                while not minibatch.end():
                    # Construct feed dictionary
                    # 通过改变feed_dict来改变每次minibatch的节点
                    feed_dict, labels = minibatch.next_minibatch_feed_dict()  # feed_dict是mibatch修改过的placeholder
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    t = time.time()
                    # Training step
                    outs = sess.run([model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
                    train_cost = outs[1]
                    iter_num = iter_num + 1
                    loss_plt.append(float(train_cost))
                    if iter % FLAGS.print_every == 0:
                        # Validation 验证集
                        sess.run(val_adj_info.op)  # sess.run()  fetch参数是一个Opration，代表执行这个操作。
                        if FLAGS.validate_batch_size == -1:
                            val_cost, val_f1_mic, val_f1_mac, duration, otu_lazy, _ = incremental_evaluate(sess, model, minibatch,
                                                                                                        FLAGS.batch_size)
                        else:
                            val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch,
                                                                                  FLAGS.validate_batch_size)
                        sess.run(train_adj_info.op)  # 每一个tensor都有op属性，代表产生这个张量的opration。
                        epoch_val_costs[-1] += val_cost

                    #if iter % FLAGS.print_every == 0:
                        #summary_writer.add_summary(outs[0], total_steps)


                    # Print results
                    avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
                    loss_plt2.append(float(val_cost))
                    valf1mi.append(float(val_f1_mic))
                    valf1ma.append(float(val_f1_mac))

                    if iter % FLAGS.print_every == 0:
                        train_f1_mic, train_f1_mac, train_f1_none = calc_f1(labels, outs[-1])
                        trainf1mi.append(float(train_f1_mic))
                        trainf1ma.append(float(train_f1_mac))

                        print("Iter:", '%04d' % iter,
                              # 训练集上的损失函数等信息
                              "train_loss=", "{:.5f}".format(train_cost),
                              "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                              "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                              # 在测试集上的损失函数值等信息
                              "val_loss=", "{:.5f}".format(val_cost),
                              "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                              "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                              "time=", "{:.5f}".format(avg_time))
                        train_loss.append(train_cost)
                        val_loss.append(val_cost)
                        train_f1_mics.append(train_f1_mic)
                        val_f1_mics.append(val_f1_mic)
                    iter += 1
                    total_steps += 1
                    if total_steps > FLAGS.max_total_steps:
                        break
                if total_steps > FLAGS.max_total_steps:
                    break

               # concat features
            elif hi_num == FLAGS.hierarchy-1:
                print("the last outputs")
            else:
                iter = 0
                minibatch.shuffle()
                while not minibatch.end():
                    print("Iter:", '%04d' % iter, "concat")
                    feed_dict, labels = minibatch.next_minibatch_feed_dict()  # feed_dict是mibatch修改过的placeholder
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    x = feed_dict[placeholders['batch']]
                    outs = sess.run([model.opt_op, model.loss, model.preds, model.node_preds], feed_dict=feed_dict)
                    features_tail = outs[3]
                    features_tail = tf.cast(features_tail, tf.float32)

                    """""
                    if hi_num == 0:
                        features_tail = tf.nn.embedding_lookup(a_class, feed_dict[placeholders["batch"]])
                    elif hi_num == 1:
                        features_tail = tf.nn.embedding_lookup(b_class, feed_dict[placeholders["batch"]])
                    else:
                        features_tail = tf.nn.embedding_lookup(c_class, feed_dict[placeholders["batch"]])
                    """""
                    hidden = tf.nn.embedding_lookup(features, feed_dict[placeholders["batch"]])
                    features_inter = tf.concat([hidden, features_tail], axis=1)

                    if iter == 0:
                        features2 = features_inter
                    else:
                        features2 = tf.concat([features2, features_inter], axis=0)
                    iter += 1

                # val features & test features
                iter_num = 0
                finished = False
                while not finished:
                    feed_dict_val, batch_labels, finished, _ = minibatch.incremental_node_val_feed_dict(FLAGS.batch_size,
                                                                                                             iter_num,
                                                                                                             test=False)
                    node_outs_val = sess.run([model.preds, model.loss, model.node_preds], feed_dict=feed_dict_val)
                    tail_val = tf.cast(node_outs_val[2], tf.float32)
                    hidden_val = tf.nn.embedding_lookup(features, feed_dict_val[placeholders["batch"]])
                    features_inter_val = tf.concat([hidden_val, tail_val], axis=1)
                    iter_num += 1
                    features2 = tf.concat([features2, features_inter_val], axis=0)
                print("val features finished")
                iter_num = 0
                finished = False
                while not finished:
                    feed_dict_test, batch_labels, finished, _ = minibatch.incremental_node_val_feed_dict(
                                                                                    FLAGS.batch_size, iter_num, test=True)
                    node_outs_test = sess.run([model.preds, model.loss, model.node_preds], feed_dict=feed_dict_test)
                    tail_test = tf.cast(node_outs_test[2], tf.float32)
                    hidden_test = tf.nn.embedding_lookup(features, feed_dict_test[placeholders["batch"]])
                    features_inter_test = tf.concat([hidden_test, tail_test], axis=1)
                    iter_num += 1
                    features2 = tf.concat([features2, features_inter_test], axis=0)
                print("test features finished")



                print("finish features concat")
                #features2 = sess.run(features2)

    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    val_cost, val_f1_mic, val_f1_mac, duration, otu_f1, ko_none = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True)
    print("Full validation stats:",
          "loss=", "{:.5f}".format(val_cost),
          "f1_micro=", "{:.5f}".format(val_f1_mic),
          "f1_macro=", "{:.5f}".format(val_f1_mac),
          "time=", "{:.5f}".format(duration))
    pred = y_ture_pre(sess, model, minibatch, FLAGS.batch_size)
    for i in range(pred.shape[0]):
        sum = 0
        for l in range(pred.shape[1]):
            sum = sum + pred[i, l]
        for m in range(pred.shape[1]):
            pred[i, m] = pred[i, m]/sum
    id = json.load(open(FLAGS.train_prefix + "-id_map.json"))
    # x_train = np.empty([pred.shape[0], array.s)
    num = 0
    session = tf.Session()
    array = session.run(features)
    x_test = np.empty([pred.shape[0], array.shape[1]])
    x_train = np.empty([len(minibatch.train_nodes), array.shape[1]])
    for node in minibatch.val_nodes:
        x_test[num] = array[id[node]]
        num = num + 1
    num1 = 0
    for node in minibatch.train_nodes:
        x_train[num1] = array[id[node]]
        num1 = num1 + 1

    with open(log_dir() + "val_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                 format(val_cost, val_f1_mic, val_f1_mac, duration))

    print("Writing test set stats to file (don't peak!)")
    val_cost, val_f1_mic, val_f1_mac, duration, otu_lazy, ko_none = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size,
                                                                                test=True)
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                 format(val_cost, val_f1_mic, val_f1_mac))

    incremental_evaluate_for_each(sess, model, minibatch, FLAGS.batch_size,
                                  test=True)


##################################################################################################################
    # plot loss
    plt.figure()
    plt.plot(loss_plt, label='train_loss')
    plt.plot(loss_plt2, label='val_loss')
    plt.legend(loc=0)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.title('Loss plot')
    plt.grid(True)
    plt.axis('tight')
    #plt.savefig("./graph/HMC12_loss.png")
    # plt.show()

    # plot f1 score
    plt.figure()
    plt.subplot(211)
    plt.plot(trainf1mi, label='train_f1_micro')
    plt.plot(valf1mi, label='val_f1_micro')
    plt.legend(loc=0)
    plt.xlabel('Iterations')
    plt.ylabel('f1_micro')
    plt.title('train_val_f1_score')
    plt.grid(True)
    plt.axis('tight')

    plt.subplot(212)
    plt.plot(trainf1ma, label='train_f1_macro')
    plt.plot(valf1ma, label='val_f1_macro')
    plt.legend(loc=0)
    plt.xlabel('Iteration')
    plt.ylabel('f1_macro')
    plt.grid(True)
    plt.axis('tight')
    #plt.savefig("./graph/HMC123_f1.png")
    # plt.show()

    plt.figure()
    plt.plot(np.arange(len(train_loss)) + 1, train_loss, label='train')
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, label='val')
    plt.legend()
    plt.savefig('loss.png')
    plt.figure()
    plt.plot(np.arange(len(train_f1_mics)) + 1, train_f1_mics, label='train')
    plt.plot(np.arange(len(val_f1_mics)) + 1, val_f1_mics, label='val')
    plt.legend()
    plt.savefig('f1.png')

    # OTU f1
    plt.figure()
    plt.plot(otu_f1, label='otu_f1')
    plt.legend(loc=0)
    plt.xlabel('OTU')
    plt.ylabel('f1_score')
    plt.title('OTU f1 plot')
    plt.grid(True)
    plt.axis('tight')
    #plt.savefig("./graph/HMC123_otu_f1.png")
    # plt.show()

    #Ko f1 score
    plt.figure()
    plt.plot(ko_none, label='Ko f1 score')
    plt.legend(loc=0)
    plt.xlabel('Ko')
    plt.ylabel('f1_score')
    plt.grid(True)
    plt.axis('tight')
    #plt.savefig("./graph/HMC123_ko_f1.png")

    bad_ko = []
    b02 = 0
    b05 = 0
    b07 = 0
    for i in range(len(ko_none)):
        if ko_none[i] < 0.2:
            bad_ko.append(i)
            b02 += 1
            bad_ko = np.array(bad_ko)
        elif ko_none[i] < 0.5:
            b05 += 1
        elif ko_none[i] < 0.7:
            b07 += 1
    print("ko f1 below 0.2:", b02)
    print("ko f1 below 0.5:", b05)
    print("ko f1 below 0.7:", b07)
    #with open('HMC123 ko below zero point two .txt', 'w') as f:
        #np.savetxt(f, bad_ko, fmt='%d', delimiter=",")

def main(self):
    #print(device_lib.list_local_devices())
    #gpu_device_name = tf.test.gpu_device_name()
    #print(gpu_device_name)
    #if tf.test.is_gpu_available():
        #print("gpu on loading")
    #else:
        #print("no avilable gpu")

    print("Loading training data..")
    # 下载toy-ppi数据集，该数据集的json文件里有test，val等属性，相当于分好训练集、测试集、验证集了，
    # 然后通过load_data()处理，就可以使用了
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data.. ")
    train(train_data)

if __name__ == '__main__':
    tf.app.run()
