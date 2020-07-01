# -*- coding: utf-8 -*-
"""
 @Time : 2020/6/30 22:13
 @Author : huangkai
 @File : RCNNATT.py
 @Software: PyCharm 
 
"""
import tensorflow as tf
class RCNNATT():
	def __init__(self, embedding, context_dim, hidden_dim, dropout_keep_prob):
		"""
		:param embedding: bert生成的embedding
		:param context_dim: lstm隐藏层维度
		:param hidden_dim:全连接层隐藏层维度
		:param dropout_keep_prob:lstm  keep_prob
		:return 调用getLogits返回logits
		"""
		self.embedding_dim = embedding.shape[-1].value
		self.embedding = embedding
		self.dropout_keep_prob = dropout_keep_prob
		self.context_dim = context_dim
		self.hidden_dim = hidden_dim

		with tf.name_scope("bi_rnn"):
			fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.context_dim)
			fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, self.dropout_keep_prob)
			bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.context_dim)
			bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, self.dropout_keep_prob)
			(output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=fw_cell,
				cell_bw=bw_cell,
				inputs=self.embedding,
				dtype=tf.float32
			)

		with tf.name_scope("context"):
			shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
			c_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
			c_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

		with tf.name_scope("word_representation"):
			y2 = tf.concat([c_left, self.embedding, c_right], axis=2, name="word_representation")

		# max_pooling层修改为Attention，将BERT的输出向量X输入BiLstm，得到一个特征向量H，最后将X和H 拼接送入Attention
		with tf.name_scope("attention"):
			hidden_size = y2.shape[2].value
			u_omega = tf.get_variable("u_omega", [hidden_size])
			with tf.name_scope('v'):
				v = tf.tanh(y2)
			vu = tf.tensordot(v, u_omega, axes=1, name='vu')
			alphas = tf.nn.softmax(vu, name='alphas')
			output = tf.reduce_sum(y2 * tf.expand_dims(alphas, -1), 1)
		self.output = tf.tanh(output)

	def getLogits(self):
		return self.output
