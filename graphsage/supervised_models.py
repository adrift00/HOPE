import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, \
    GCNAggregator, GatAggregator, MLPAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
                  placeholders, features, adj, degrees,
                 layer_infos, cb, hi_num, concat=True, aggregator_type="mean",
                 model_size="small", sigmoid_loss=False, identity_dim=0,
                 **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "mlp":
            self.aggregator_cls = MLPAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type =='gat':
            self.aggregator_cls = GatAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]  # 改变placeholders来改变每次的节点，placeholder很重要
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = features  # tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False) #为什么要用Variable？
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1].value) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])  # 输出维度128
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.cb = cb
        self.hi = hi_num
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()



    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        # add samples
        samples2, support_sizes2 = self.sample(self.inputs1, self.layer_infos)
        samples3, support_sizes3 = self.sample(self.inputs1, self.layer_infos)
        samples4, support_sizes4 = self.sample(self.inputs1, self.layer_infos)
        samples5, support_sizes5 = self.sample(self.inputs1, self.layer_infos)

        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                                                      support_sizes1, concat=self.concat, model_size=self.model_size)
        #""""
        # add outputs
        self.outputs2, self.aggregators = self.aggregate(samples2, [self.features], self.dims, num_samples,
                                                         support_sizes2, concat=self.concat, model_size=self.model_size)
        self.outputs3, self.aggregators = self.aggregate(samples3, [self.features], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs4, self.aggregators = self.aggregate(samples4, [self.features], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs5, self.aggregators = self.aggregate(samples5, [self.features], self.dims, num_samples,
                                                        support_sizes1, concat=self.concat, model_size=self.model_size)
        #"""
        # conv3
        """"
        self.outputs1 = tf.stack([self.outputs1, self.outputs2, self.outputs3], 2)
        self.outputs1 = tf.expand_dims(self.outputs1, 0)
        self.outputs1 = tf.layers.conv2d(
            inputs=self.outputs1, filters=1, kernel_size=[1, 1],  activation='relu'
        )
        self.outputs1 = tf.squeeze(self.outputs1)
        """

        # concat3
        #self.outputs1 = tf.concat([self.outputs1, self.outputs2, self.outputs3], 1)

        dim_mult = 1*2 if self.concat else 1

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)  # 这个其实是embedding之后出来的向量。


        self.node_pred = layers.Dense(int((dim_mult * self.dims[-1]+self.num_classes)*2/3),
                                      self.num_classes,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x)
        self.hidden_layer1 = layers.Dense(dim_mult * self.dims[-1],
                                         int((dim_mult * self.dims[-1]+self.num_classes)*2/3),
                                         dropout=self.placeholders['dropout'],
                                         act=lambda x: x)
        self.hidden_layer2 = layers.Dense(int((dim_mult * self.dims[-1]+self.num_classes)*2/3),
                                          int((dim_mult * self.dims[-1] + self.num_classes) * 2 / 3),
                                          dropout=self.placeholders['dropout'],
                                          act=lambda x: x)

        # TF graph management
        self.node_preds = self.hidden_layer1(self.outputs1)
        #self.node_preds = self.hidden_layer2(self.node_preds)
        #self.node_preds = self.hidden_layer2(self.node_preds)
        self.node_preds = self.node_pred(self.node_preds)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()



    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # add_loss

        ''' 
        y_true = self.placeholders['labels']
        y_pred = self.node_preds
        sub = tf.constant(1, shape = [1, 5526])
        tp = tf.reduce_sum(tf.cast(tf.multiply(y_true, y_pred), tf.float32), axis=0)
        tn = tf.reduce_sum(tf.cast(tf.multiply(tf.subtract(sub, y_true), tf.subtract(sub, y_pred)), tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast(tf.multiply(tf.subtract(sub, y_true), y_pred), tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(tf.multiply(y_true, tf.subtract(sub, y_pred)), tf.float32), axis=0)

        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())

        f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1) 
        '''

        # classification loss
        if self.sigmoid_loss:
            '''''
            beta = tf.constant(0.999, dtype=tf.float32)
            a1 = tf.pow(beta, tf.reduce_sum(self.placeholders['labels'], 0, keepdims=True))
            a2 = tf.subtract(tf.constant(1, dtype=tf.float32, shape=[1, a1.shape[1]]), a1)
            a3 = tf.subtract(tf.constant(1, dtype=tf.float32, shape=[1, a1.shape[1]]), beta)
            a4 = a3/a2
            a5 = tf.constant(self.num_classes, dtype=tf.float32)/tf.reduce_sum(a4, 1, keepdims=True)
            a6 = tf.multiply(a4, a5)
            '''''
            if self.hi == 0 or 1 or 2:
                self.loss += tf.reduce_mean(tf.multiply(self.cb, tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels'])))

                #self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    #logits=self.node_preds,
                    #labels=self.placeholders['labels']))
            else:
                # margin loss
                #"""""
                self.level = tf.constant(0.35, shape=[1, self.num_classes])
                self.all1 = tf.constant(1, dtype=tf.float32, shape=[1, self.num_classes])
                self.all0 = tf.constant(0, dtype=tf.float32, shape=[1, self.num_classes])
                self.level1 = tf.constant(0.25, shape=[1, self.num_classes])
                self.level = tf.multiply(self.placeholders['labels'], self.level)
                self.neglabel = tf.subtract(self.all1, self.placeholders['labels'])
                self.level2 = tf.multiply(self.neglabel, self.level1)
                #"""""

                self.level = tf.multiply(self.placeholders['labels'], self.cb)
                self.level2 = tf.multiply(self.neglabel, self.cb)
                #self.loss += ((tf.reduce_sum(tf.nn.relu(tf.negative(tf.subtract(tf.multiply(self.node_preds, self.placeholders['labels']), self.level)))) + tf.reduce_sum(tf.nn.relu(tf.subtract(tf.multiply(self.node_preds, tf.subtract(self.all1, self.placeholders['labels'])), self.level2)))))

                self.loss += tf.reduce_sum(tf.multiply(self.cb, tf.add(tf.nn.relu(tf.negative(tf.subtract(tf.multiply(self.node_preds, self.placeholders['labels']), self.level))), tf.nn.relu(tf.subtract(tf.multiply(self.node_preds, tf.subtract(self.all1, self.placeholders['labels'])), self.level2)))))

                #self.loss += -tf.log(1/(1+(tf.reduce_sum(tf.subtract(self.all1, tf.multiply(self.node_preds, self.placeholders['labels']))+tf.multiply(self.node_preds, self.neglabel)))))

        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.node_preds,
            abels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def f1_loss(y_true, y_pred):

        tp = tf.reduce_sum(tf.cast(tf.multiply(y_true, y_pred, 'float')), axis=0)
        tn = tf.reduce_sum(tf.cast(tf.multiply(tf.subtract(1, y_true), tf.subtract(1, y_pred)), 'float'), axis=0)
        fp = tf.reduce_sum(tf.cast(tf.multiply(tf.subtract(1, y_true),  y_pred), 'float'), axis=0)
        fn = tf.reduce_sum(tf.cast(tf.multiply(y_true, tf.subtract(1, y_pred)), 'float'), axis=0)

        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())

        f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - tf.reduce_mean(f1)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
