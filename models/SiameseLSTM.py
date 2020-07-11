# -*- coding: utf-8 -*-
"""
 @Time : 2020/6/17 22:27
 @Author : huangkai
 @File : SiameseLSTM.py
 @Software: PyCharm

"""

import tensorflow as tf

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def __init__(
            self, embedding1,embedding2, hidden_units,dropout_keep_prob):

        # Placeholders for input, output and dropout
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")


        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            out1 = self.BiRNN(embedding1, dropout_keep_prob, "side1", hidden_units)
            out2 = self.BiRNN(embedding2, dropout_keep_prob, "side2",  hidden_units)
            # self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            # self.distance = tf.div(self.distance,
            #                        tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
            #                               tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            # pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(out1), 1))
            # pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(out2), 1))
            # pooled_mul_12 = tf.reduce_sum(out1 *out2, 1)
            self.distance=self.get_cos_distance(out1,out2)
           # self.distance = tf.reshape(distance, [-1], name="distance")
            print('distance_shape:',self.distance.shape)

    def getLogits(self):
        return self.distance
        # with tf.name_scope("loss"):
        #     self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
        # #### Accuracy computation is outside of this class.
        # with tf.name_scope("accuracy"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
        #                                 name="temp_sim")  # auto threshold 0.5
        #     correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def BiRNN(self, x, dropout, scope,  hidden_units):
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        #x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=x,
                dtype=tf.float32
            )
        with tf.name_scope("context"):
            shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
            c_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
            c_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word_representation"):
            y2 = tf.concat([c_left, x, c_right], axis=2, name="word_representation")

        # max_pooling层
        with tf.name_scope("max_pooling"):
            fc = tf.layers.dense(y2, hidden_units, activation=tf.nn.relu)
            self.output = tf.reduce_max(fc, axis=1)
            print("output shape:",self.output.shape)
        return self.output
    def get_cos_distance(self,X1, X2):
        # calculate cos distance between two sets
        # more similar more big
        (k, n) = X1.shape
        (m, n) = X2.shape
        # 求模
        X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
        X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
        # 内积
        X1_X2 = tf.reduce_sum(tf.multiply(X1, X2),axis=1)
        print('x1_x2 ',X1_X2.shape)
        print('x1_norm ', X1_norm.shape)
        print('x2_norm ', X2_norm.shape)
        X1_X2_norm = tf.multiply(tf.reshape(X1_norm, [k,1]), tf.reshape(X2_norm, [m, 1]))
        # 计算余弦距离
        cos = tf.reshape(X1_X2, [k,-1])/ X1_X2_norm
        return cos
