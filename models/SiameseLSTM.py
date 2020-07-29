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


        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            out1 = self.BiRNN(embedding1, dropout_keep_prob, "side1", hidden_units)
            out2 = self.BiRNN(embedding2, dropout_keep_prob, "side2",  hidden_units)
            print(out1.shape,out2.shape)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(out1, out2)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(out1), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(out2), 1, keep_dims=True))))
            # pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(out1), 1))
            # pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(out2), 1))
            # pooled_mul_12 = tf.reduce_sum(out1 *out2, 1)
            # self.output = tf.concat([tf.multiply(out1, out2), tf.abs(tf.subtract(out1, out2))], axis=-1)
            self.distance = tf.reshape(self.distance, [-1], name="distance")
            print('distance_shape:',self.distance.shape)


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
            outputs,_= tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw_cell_m,
                cell_bw=lstm_bw_cell_m,
                inputs=x,
                dtype=tf.float32
            )
            #print('1',outputs[0].shape)# shape :[batch_size,max_length,hiddensize]
            outputs=tf.concat(outputs,2)
            print('1',outputs.shape)
        outputs=tf.reduce_mean(outputs,axis=1) #temporal average
        print('2',outputs.shape)
        with tf.name_scope("feedforward_128"):
            intermediate_output = tf.layers.dense(outputs, 128, activation=tf.nn.relu)
            intermediate_output = tf.nn.dropout(intermediate_output, keep_prob=0.9)
        outputs=tf.reshape(intermediate_output,(-1,128))
        print(outputs.shape)
        return  outputs

    def get_cos_distance(self,X1, X2):
        # calculate cos distance between two sets
        # more similar more big
        (k, n) = X1.shape
        (m, n) = X2.shape
        # 求模
        X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
        X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
        # 内积
        X1_X2 = tf.matmul(X1, tf.transpose(X2))
        X1_X2_norm = tf.matmul(tf.reshape(X1_norm, [k, 1]), tf.reshape(X2_norm, [1, m]))
        # 计算余弦距离
        cos = X1_X2 / X1_X2_norm
        return cos
