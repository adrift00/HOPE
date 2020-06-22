from __future__ import division
from __future__ import print_function
import math
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
import warnings
from tensorflow.python.client import device_lib
from preprocess import generate_jsons

warnings.filterwarnings('ignore')
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
flags.DEFINE_float('learning_rate', 0.005, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')  # 必须要填的

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.4, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.00, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 32, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 15, 'number of samples in layer 2')
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
flags.DEFINE_integer('gpu', 2, "which gpu to use.")
flags.DEFINE_integer('print_every', 1, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")

flags.DEFINE_integer('hierarchy', 3, "The degree of hierarchical classes.")
flags.DEFINE_integer('ko_threshold', 950, "The threshold of kos which are thought the long tails.")
flags.DEFINE_integer('ko_threshold2', 480, "The threshold of kos which are thought the long tails.")
flags.DEFINE_float('beta1', 0.99, "Beta of CB in loss.")
flags.DEFINE_float('beta2', 0.995, "Beta of CB in f1.")


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
GPU_MEM_FRACTION = 0.95




def calc_f1(y_true, y_pred, f1_par0):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    else:
        #y_pred[y_pred > 0.3] = 1
        #y_pred[y_pred <= 0.3] = 0

        for i in range(len(y_pred)):
            for l in range(y_pred.shape[1]):
                if y_pred[i][l] > (f1_par0[l]*0.3):
                    y_pred[i][l] = 1
                else:
                    y_pred[i][l] = 0

    with open('y_pred.txt', 'w') as f:
       np.savetxt(f, y_pred, fmt='%d', delimiter=",")
    with open('y_true.txt', 'w') as f:
       np.savetxt(f, y_true, fmt='%f', delimiter=",")
    #f1_spilt = metrics.f1_score(y_true, y_pred, average=None)
    total_sum = 0
    for i in range(y_true.shape[0]):
        acc = metrics.accuracy_score(y_true[i, ...], y_pred[i, ...])
        total_sum = total_sum + acc
    accuracy = total_sum/y_true.shape[0]
    mi_roc_auc = metrics.roc_auc_score(y_true, y_pred, average="micro")
    #ma_roc_auc = metrics.roc_auc_score(y_true, y_pred, average='macro')
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro"), metrics.f1_score(y_true, y_pred, average=None), accuracy, mi_roc_auc

def calc_precision(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.3] = 1
        y_pred[y_pred <= 0.3] = 0
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
        y_pred[y_pred > 0.3] = 1
        y_pred[y_pred <= 0.3] = 0
        return metrics.f1_score(y_true, y_pred, average=None)
    f1_scores = calc_f1_each(labels, val_preds)
    with open('f1_scores.txt', 'w') as f:
        np.savetxt(f, f1_scores, fmt='%0.5f')
    return f1_scores

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, f1_par0, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                             feed_dict=feed_dict_val)
    mic, mac, ko_none, acc, mi_roc = calc_f1(labels, node_outs_val[0], f1_par0)
    return node_outs_val[1], mic, mac, (time.time() - t_test), acc, mi_roc


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter, f1_par0, size, test=False):
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

    lis_preds = val_preds.tolist()
    lis_label = labels.tolist()
    OTU_mi = []
    for i in range(val_preds.shape[0]):
        OTU_f1 = calc_precision(labels[i, ...], val_preds[i, ...])
        OTU_mi.append(float(OTU_f1))

    f1_scores = calc_f1(labels, val_preds, f1_par0)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test), OTU_mi, f1_scores[2], val_preds, labels, f1_scores[3], f1_scores[4]

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

def construct_class_para(class_map_ko, pd, beta0):
    ny = class_map_ko.sum(axis=0)
    beta = beta0
    ko_cb = []
    for ko_val in ny:
        x = math.pow(beta, ko_val)
        if pd == 0:
            cb = (1 - x) / (1 - beta)
        else:
            cb = (1 - beta) / (1 - x)
        ko_cb.append(cb)
    ko_cb = np.array(ko_cb)
    sumko = 0
    for ko in ko_cb:
        sumko = sumko + ko
    y = class_map_ko.shape[1] / sumko
    ko_cb = ko_cb * y
    return ko_cb

def train(train_data, test_data=None):
    G = train_data[0]  # G 是一个Networkx里的对象，这几个都是经过load_data()处理过的
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]
    class_map2 = train_data[5]
    class_map3 = train_data[6]
    #class_map = class_map
    hierarchy = FLAGS.hierarchy
    ko_threshold = FLAGS.ko_threshold
    ko_threshold2 = FLAGS.ko_threshold2
    if features is not None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])
    features = tf.cast(features, tf.float32)
    for hi_num in range(hierarchy):
        if hi_num == 0:
            class_map_ko_0 = construct_class_numpy(class_map)
            class_map_ko = construct_class_numpy(class_map)
            a = class_map_ko.sum(axis=0)
            count = 0
            list_del = []
            for i in a:
                if i < ko_threshold:
                    list_del.append(count)
                    count += 1
                else:
                    count += 1
            class_map_ko = np.delete(class_map_ko, list_del, axis=1)
            count = 0
            for key in class_map:
                arr = class_map_ko[count, :]
                class_map[key] = arr.tolist()
                count += 1
            num_classes = class_map_ko.shape[1]

        elif hi_num == 1:
            class_map = class_map2
            class_map_ko_1 = construct_class_numpy(class_map)
            class_map_ko = construct_class_numpy(class_map)
            a = class_map_ko.sum(axis=0)
            count = 0
            list_del = []
            for i in a:
                if i < ko_threshold2:
                    list_del.append(count)
                    count += 1
                else:
                    count += 1
            class_map_ko = np.delete(class_map_ko, list_del, axis=1)
            count = 0
            for key in class_map:
                arr = class_map_ko[count, :]
                class_map[key] = arr.tolist()
                count += 1
            num_classes = class_map_ko.shape[1]

        elif hi_num == 2:
            class_map = class_map3
            class_map_ko_2 = construct_class_numpy(class_map)
            class_map_ko = construct_class_numpy(class_map)
            a = class_map_ko.sum(axis=0)
            count = 0
            list_del = []
            for i in a:
                if i > ko_threshold2:
                    list_del.append(count)
                    count += 1
                else:
                    count += 1
            class_map_ko = np.delete(class_map_ko, list_del, axis=1)
            count = 0
            for key in class_map:
                arr = class_map_ko[count, :]
                class_map[key] = arr.tolist()
                count += 1
            num_classes = class_map_ko.shape[1]


        OTU_ko_num = class_map_ko.sum(axis=1)
        count = 0
        for num in OTU_ko_num:
            if num < 100:
                count += 1
        ko_cb = construct_class_para(class_map_ko, 0, FLAGS.beta1)
        ko_cb = tf.cast(ko_cb, tf.float32)
        f1_par = construct_class_para(class_map_ko, 1, FLAGS.beta2)


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


        with open('test_nodes.txt', 'w') as f:
            json.dump(minibatch.test_nodes, f)
    ###########
        list_node = minibatch.nodes
        for otu in minibatch.train_nodes:
            if otu in list_node:
                list_node.remove(otu)
        for otu in minibatch.val_nodes:
            if otu in list_node:
                list_node.remove(otu)
        for otu in minibatch.test_nodes:
            if otu in list_node:
                list_node.remove(otu)
    ###########
        if hi_num == 0:
            adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        # 把adj_info设成Variable应该是因为在训练和测试时会改变adj_info的值，所以
        # 用Varible然后用tf.assign()赋值。
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

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


            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,  # 每一个的度
                                        layer_infos,
                                        ko_cb, hi_num,
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True,
                                        concat=False
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

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
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

        for epoch in range(FLAGS.epochs*2):
            if epoch < FLAGS.epochs:
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
                            val_cost, val_f1_mic, val_f1_mac, duration, otu_lazy, _, val_preds, __, val_accuracy, val_mi_roc_auc = incremental_evaluate(sess, model, minibatch, f1_par,
                                                                                                        FLAGS.batch_size)
                        else:
                            val_cost, val_f1_mic, val_f1_mac, duration, val_accuracy, val_mi_roc_auc = evaluate(sess, model, minibatch, f1_par,
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
                        train_f1_mic, train_f1_mac, train_f1_none, train_accuracy, train_mi_roc_auc = calc_f1(labels, outs[-1], f1_par)
                        trainf1mi.append(float(train_f1_mic))
                        trainf1ma.append(float(train_f1_mac))

                        print("Iter:", '%04d' % iter,
                              # 训练集上的损失函数等信息
                              "train_loss=", "{:.5f}".format(train_cost),
                              "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                              "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                              "train_accuracy=", "{:.5f}".format(train_accuracy),
                              "train_ra_mi=", "{:.5f}".format(train_mi_roc_auc),

                              # 在测试集上的损失函数值等信息
                              "val_loss=", "{:.5f}".format(val_cost),
                              "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                              "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                              "val_accuracy=", "{:.5f}".format(val_accuracy),
                              "val_ra_mi=", "{:.5f}".format(val_mi_roc_auc),

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
    ###################################################################################################################
            # begin second degree training
    ###################################################################################################################
            """""
            else:
                minibatch2.shuffle()
                iter = 0
                print('Epoch2: %04d' % (epoch + 1))
                epoch_val_costs2.append(0)
                while not minibatch2.end():
                # Construct feed dictionary
                # 通过改变feed_dict来改变每次minibatch的节点
                    feed_dict, labels = minibatch2.next_minibatch_feed_dict()  # feed_dict是mibatch修改过的placeholder
                    feed_dict.update({placeholders2['dropout']: FLAGS.dropout})
    
                    t = time.time()
                # Training step
                    #global model2
                    outs = sess.run([merged, model2.opt_op, model2.loss, model2.preds], feed_dict=feed_dict)
    
                    train_cost = outs[2]
                    iter_num = iter_num + 1
                    loss_plt.append(float(train_cost))
                    if iter % FLAGS.print_every == 0:
                    # Validation 验证集
                        sess.run(val_adj_info2.op)  # sess.run()  fetch参数是一个Opration，代表执行这个操作。
                        if FLAGS.validate_batch_size == -1:
                            val_cost, val_f1_mic, val_f1_mac, duration, otu_lazy = incremental_evaluate(sess, model2, minibatch2,
                                                                                                    FLAGS.batch_size)
                        else:
                            val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model2, minibatch2,
                                                                              FLAGS.validate_batch_size)
                        sess.run(train_adj_info2.op)  # 每一个tensor都有op属性，代表产生这个张量的opration。
                        epoch_val_costs2[-1] += val_cost
    
                    if iter % FLAGS.print_every == 0:
                        summary_writer.add_summary(outs[0], total_steps)
    
                # Print results
                    avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
                    loss_plt2.append(float(val_cost))
                    valf1mi.append(float(val_f1_mic))
                    valf1ma.append(float(val_f1_mac))
    
                    if iter % FLAGS.print_every == 0:
                        train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
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
            """

        print("Optimization Finished!")
        sess.run(val_adj_info.op)
        if hi_num == 1:
            last_preds = test_preds
            last_labels = test_labels
        val_cost, val_f1_mic, val_f1_mac, duration, otu_f1, ko_none, test_preds, test_labels, test_accuracy, test_mi_roc_auc = incremental_evaluate(sess, model, minibatch, f1_par, FLAGS.batch_size, test=True)
        print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "f1_micro=", "{:.5f}".format(val_f1_mic),
              "f1_macro=", "{:.5f}".format(val_f1_mac),
              "accuracy=", "{:.5f}".format(test_accuracy),
              "roc_auc_mi=", "{:.5f}".format(test_mi_roc_auc),

              "time=", "{:.5f}".format(duration),)

        if hi_num == 1:
            # update test preds
            """
            ab_ko = json.load(open(FLAGS.train_prefix + "-below1500_ko_idx.json"))
            #ab_ko = construct_class_numpy(ab_ko)
            f1_par = construct_class_para(class_map_ko_0, 1, FLAGS.beta2)
            i = 0
            for col in ab_ko:
                last_preds[..., col] = test_preds[..., i]
                i += 1
            f1_scores = calc_f1(last_preds, last_labels, f1_par)
            """
            f1_par = construct_class_para(class_map_ko_0, 1, FLAGS.beta2)
            final_preds = np.hstack((last_preds, test_preds))
            final_labels = np.hstack((last_labels, test_labels))

        elif hi_num == 2:
            f1_par = construct_class_para(class_map_ko_0, 1, FLAGS.beta2)
            final_preds = np.hstack((final_preds, test_preds))
            final_labels = np.hstack((final_labels, test_labels))
            f1_scores = calc_f1(final_preds, final_labels, f1_par)
            print('\n', 'Hierarchy combination f1 score:')
            print("f1_micro=", "{:.5f}".format(f1_scores[0]),
              "f1_macro=", "{:.5f}".format(f1_scores[1]),
              "accuracy=", "{:.5f}".format(f1_scores[3]),
              "roc_auc_mi=", "{:.5f}".format(f1_scores[4])
             )

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
        val_cost, val_f1_mic, val_f1_mac, duration, otu_lazy, ko_none, _, __, test_accuracy, test_mi_roc_auc = incremental_evaluate(sess, model, minibatch, f1_par, FLAGS.batch_size,
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
    #plt.savefig("./graph/HMC_SAGE_CB_loss.png")
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
   # plt.savefig("./graph/HMC_SAGE_CB_f1.png")
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
    #plt.savefig('f1.png')

    # OTU f1
    plt.figure()
    plt.plot(otu_f1, label='otu_f1')
    plt.legend(loc=0)
    plt.xlabel('OTU')
    plt.ylabel('f1_score')
    plt.title('OTU f1 plot')
    plt.grid(True)
    plt.axis('tight')
    #plt.savefig("./graph/below_1500_CECB15_otu_f1.png")
    # plt.show()

    ko_none = f1_scores[2]
    # Ko f1 score
    plt.figure()
    plt.plot(ko_none, label='Ko f1 score')
    plt.legend(loc=0)
    plt.xlabel('Ko')
    plt.ylabel('f1_score')
    plt.grid(True)
    plt.axis('tight')
    #plt.savefig("./graph/below1500_CECB15_ko_f1.png")
    bad_ko = []
    b02 = 0
    b05 = 0
    b07 = 0
    for i in range(len(ko_none)):
        if ko_none[i] < 0.2:
            bad_ko.append(i)
            b02 += 1
        elif ko_none[i] < 0.5:
            b05 += 1
        elif ko_none[i] < 0.7:
            b07 += 1
    print("ko f1 below 0.2:", b02)
    print("ko f1 below 0.5:", b05)
    print("ko f1 below 0.7:", b07)
    print("ko f1 over 0.7:", len(ko_none)-b02-b05-b07)
    bad_ko = np.array(bad_ko)
    with open('./new_data_badko/graph7 ko below zero point two .txt', 'w') as f:
        np.savetxt(f, bad_ko, fmt='%d', delimiter=",")

def main(self):
    print("Loading training data..")
    # 下载toy-ppi数据集，该数据集的json文件里有test，val等属性，相当于分好训练集、测试集、验证集了，
    # 然后通过load_data()处理，就可以使用了
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data.. ")
    train(train_data)
if __name__ == '__main__':
    tf.app.run()
