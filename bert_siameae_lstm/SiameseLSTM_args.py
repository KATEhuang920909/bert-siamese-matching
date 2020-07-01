# -*- coding: utf-8 -*-
"""
 @Time : 2020/6/23 21:36
 @Author : huangkai
 @File : SiameseLSTM_args.py
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
bert_model_dir = curdir + "/chinese_L-12_H-768_A-12/"

## Required parameters

flags.DEFINE_string(
	"model_type",
	"SiameseLSTM",
	"The name of the layer to train after bert selected in the list")

flags.DEFINE_string(
    "data_dir", curdir +'/data/',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

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
                      0.8,
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

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 50,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
