# -*- coding: utf-8 -*-
"""
 @Time : 2020/6/23 21:36
 @Author : huangkai
 @File : simaese_network_args.py
 @Software: PyCharm

"""


import os
import sys
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# path setting
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
print(curdir)
bert_model_dir = curdir + "/chinese_wwm_ext_L-12_H-768_A-12/"

## Required parameters

tf.flags.DEFINE_boolean("is_char_based",
                        True,
                        "is character based syntactic similarity. "
                        "if false then word embedding based semantic similarity is used."
                        "(default: True)")


tf.flags.DEFINE_string("word2vec_model",
                       "wiki.simple.vec",
                       "word2vec pre-trained embeddings file (default: None)")


tf.flags.DEFINE_string("word2vec_format",
                       "text",
                       "word2vec pre-trained embeddings file format (bin/text/textgz)(default: None)")

tf.flags.DEFINE_integer("embedding_dim",
                        300,
                        "Dimensionality of character embedding (default: 300)")


tf.flags.DEFINE_float("dropout_keep_prob",
                      1.0,
                      "Dropout keep probability (default: 1.0)")


tf.flags.DEFINE_float("l2_reg_lambda",
                      0.0,
                      "L2 regularizaion lambda (default: 0.0)")


tf.flags.DEFINE_string("training_files",
                       "person_match.train2",
                       "training file (default: None)")  #for sentence semantic similarity use "train_snli.txt"


tf.flags.DEFINE_integer("hidden_units",
                        50,
                        "Number of hidden units (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size",
                        64,
                        "Batch Size (default: 64)")


tf.flags.DEFINE_integer("num_epochs",
                        300,
                        "Number of training epochs (default: 200)")


tf.flags.DEFINE_integer("evaluate_every",
                        1000,
                        "Evaluate models on dev set after this many steps (default: 100)")


tf.flags.DEFINE_integer("checkpoint_every",
                        1000,
                        "Save models after this many steps (default: 100)")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement",
                        True,
                        "Allow device soft device placement")


tf.flags.DEFINE_boolean("log_device_placement",
                        False,
                        "Log placement of ops on devices")

