# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pandas as pd
import os
import siamese_bert_model
from embedding.bert import optimization
from embedding.bert import tokenization
import tensorflow as tf
import shutil
from models.SiameseLSTM import SiameseLSTM
from args import FLAGS

flags = tf.flags


## Required parameters
# Parameters
# ==================================================


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids1, input_mask1, segment_ids1,
                 input_ids2, input_mask2, segment_ids2,
                 label_id):
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.segment_ids1 = segment_ids1

        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2

        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data=pd.read_csv(input_file)

        return data


class MyProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for index,lines in data.iterrows():
            guid = "%s-%s" % (set_type, index)

            text_a = tokenization.convert_to_unicode(lines[0])
            text_b = tokenization.convert_to_unicode(lines[1])
            label = tokenization.convert_to_unicode(str(lines[2]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_single_sentence(tokens_input, max_seq_length, tokenizer):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_input:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # if ex_index < 5:
    #     tf.logging.info("*** Example ***")
    #     tf.logging.info("*** %s ***" % tag)
    #     tf.logging.info("tokens: %s" % " ".join(
    #         [tokenization.printable_text(x) for x in tokens]))
    #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     tf.logging.info("label:(id = %d)" % (label_id))

    return input_ids, input_mask, segment_ids


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    # tokens=["[CLS]",0,0,...,"[SEP",1,1,1,...,1,"[SEP"]
    label_id = label_map[example.label]
    input_ids1, input_mask1, segment_ids1 = convert_single_sentence(tokens_a, max_seq_length, tokenizer)
    input_ids2, input_mask2, segment_ids2 = convert_single_sentence(tokens_b, max_seq_length, tokenizer)

    # ex_index为 example的编号，从0开始
    # 返回前五个样本
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens_a: %s" % " ".join([tokenization.printable_text(x) for x in tokens_a]))
        tf.logging.info("tokens_b: %s" % " ".join([tokenization.printable_text(x) for x in tokens_b]))
        tf.logging.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))
        tf.logging.info("input_ids2: %s" % " ".join([str(x) for x in input_ids2]))
        tf.logging.info("input_mask1: %s" % " ".join([str(x) for x in input_mask1]))
        tf.logging.info("input_mask2: %s" % " ".join([str(x) for x in input_mask2]))
        tf.logging.info("segment_ids1: %s" % " ".join([str(x) for x in segment_ids1]))
        tf.logging.info("segment_ids2: %s" % " ".join([str(x) for x in segment_ids2]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    # input_ids:样本的word索引列表
    # input_mask:掩码列表
    # segment_ids:隔断句编号
    # label_id:标签编号
    feature = InputFeatures(
        input_ids1=input_ids1,
        input_mask1=input_mask1,
        segment_ids1=segment_ids1,
        input_ids2=input_ids2,
        input_mask2=input_mask2,
        segment_ids2=segment_ids2,
        label_id=label_id)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids1"] = create_int_feature(feature.input_ids1)
        features["input_mask1"] = create_int_feature(feature.input_mask1)
        features["segment_ids1"] = create_int_feature(feature.segment_ids1)
        features["input_ids2"] = create_int_feature(feature.input_ids2)
        features["input_mask2"] = create_int_feature(feature.input_mask2)
        features["segment_ids2"] = create_int_feature(feature.segment_ids2)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids1": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask1": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids1": tf.FixedLenFeature([seq_length], tf.int64),
        "input_ids2": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask2": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids2": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # print('arrive input_fn 1')
        batch_size = params["batch_size"]
        # print('arrive input_fn 2')
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        # print('arrive input_fn 3')
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            # print('arrive input_fn 4')

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        # print('finish input_fn ')

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training,
                 input_ids1, input_mask1, segment_ids1,
                 input_ids2, input_mask2, segment_ids2,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = siamese_bert_model.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids1=input_ids1,
        input_mask1=input_mask1,
        token_type_ids1=segment_ids1,
        input_ids2=input_ids2,
        input_mask2=input_mask2,
        token_type_ids2=segment_ids2,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output1 = model.get_sequence_output1()  # [batch_size, seq_length, embedding_size]
    output2 = model.get_sequence_output2()
    print(output1.shape, output2.shape)
    # 接入lstm层
    model_layer = SiameseLSTM(output1, output2, FLAGS.hidden_units, FLAGS.dropout_keep_prob)
    output = model_layer.output

    # with tf.name_scope("out"):
    #     intermediate_output = tf.layers.dense(output, 256, activation=tf.nn.relu)
    #     if is_training:
    #         intermediate_output = tf.nn.dropout(intermediate_output, keep_prob=0.9)
    #     logits = tf.layers.dense(intermediate_output, 1)
    #     logits = tf.reshape(logits, [-1])
    #     predict = tf.nn.sigmoid(logits)
    labels = tf.cast(labels, tf.float32)

    def contrastive_loss(y, d):
        tmp = (1-y) * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = y * tf.square(tf.maximum((1 - d), 0))
        return tmp+tmp2

    with tf.name_scope("loss"):
        print('label ,logits shape',labels.shape,output.shape)
        per_example_loss = contrastive_loss(labels, output)#, FLAGS.batch_size)
        loss=tf.reduce_mean(per_example_loss)
    # with tf.name_scope("loss"):
    #     per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    #     loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, output)
    # else:
    #     #将两个output拼接
    #     output = tf.concat([tf.multiply(output1, output2), tf.abs(tf.subtract(output1, output2))], axis=-1)
    #     # output = tf.concat([output1, output2], axis=-1)
    #
    #     with tf.variable_scope("bert_output_binary_cls"):
    #         intermediate_output = tf.layers.dense(output, 256, activation=tf.nn.relu)
    #         if is_training:
    #             intermediate_output = tf.nn.dropout(intermediate_output, keep_prob=0.9)
    #         logits = tf.layers.dense(intermediate_output, 1)
    #         logits = tf.reshape(logits, [-1])
    #         predict = tf.nn.sigmoid(logits)
    #
    #     labels = tf.cast(labels, tf.float32)
    #     with tf.variable_scope("loss"):
    #         per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    #         loss = tf.reduce_mean(per_example_loss)
    #
    #
    #         return (loss, per_example_loss, logits, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids1 = features["input_ids1"]
        input_mask1 = features["input_mask1"]
        segment_ids1 = features["segment_ids1"]

        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss,  probabilities) = create_model(
            bert_config, is_training,
            input_ids1, input_mask1, segment_ids1,
            input_ids2, input_mask2, segment_ids2,
            label_ids,
            num_labels, use_one_hot_embeddings)
        print("total_loss::", total_loss)
        print("per_example_loss::", per_example_loss)
        print("probabilities::", probabilities)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = siamese_bert_model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        if mode == tf.estimator.ModeKeys.TRAIN:
            predictions = tf.cast(probabilities > 0.5, tf.int32)
            accuracy = tf.metrics.accuracy(label_ids, predictions, name="accuracy")
            tf.summary.scalar('accuracy', accuracy)
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            show_dict = {
                "total_loss": total_loss,
                # 'accuracy':accuracy
            }
            logging_hook = tf.train.LoggingTensorHook(show_dict, every_n_iter=10)
            # logging_hook = tf.train.LoggingTensorHook({"total_loss:": total_loss}, every_n_iter=10)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities):
                predictions = tf.cast(probabilities > 0.5, tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, probabilities])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=probabilities,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids1 = []
    all_input_mask1 = []
    all_segment_ids1 = []
    all_input_ids2 = []
    all_input_mask2 = []
    all_segment_ids2 = []
    all_label_ids = []

    for feature in features:
        all_input_ids1.append(feature.input_ids1)
        all_input_mask1.append(feature.input_mask1)
        all_segment_ids1.append(feature.segment_ids1)

        all_input_ids2.append(feature.input_ids2)
        all_input_mask2.append(feature.input_mask2)
        all_segment_ids2.append(feature.segment_ids2)

        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids1":
                tf.constant(
                    all_input_ids1,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask1":
                tf.constant(
                    all_input_mask1,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids1":
                tf.constant(
                    all_segment_ids1,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_ids2":
                tf.constant(
                    all_input_ids2,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask2":
                tf.constant(
                    all_input_mask2,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids2":
                tf.constant(
                    all_segment_ids2,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(
                    all_label_ids,
                    shape=[num_examples],
                    dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "sim": MyProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = siamese_bert_model.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        shutil.rmtree(FLAGS.output_dir)  # 将整个文件夹删除
        os.makedirs(FLAGS.output_dir)  # 重新创建文件夹
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        #  train part
        best_val_loss = 10000.0
        no_best_num = 0
        for i in range(1, int(FLAGS.num_train_epochs + 1)):
            train_steps = int(len(train_examples) / FLAGS.train_batch_size) * i

            tf.logging.info("Run train %d epoch..." % i)
            tf.logging.info("train step %d ..." % train_steps)

            estimator.train(input_fn=train_input_fn, max_steps=train_steps)
            result = estimator.evaluate(input_fn=eval_input_fn)

            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "a") as writer:
                tf.logging.info("***** Eval results %d round...*****" % i)
                writer.write("***** Eval results %d round...*****\n" % i)
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

                eval_loss = result["eval_loss"]

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                no_best_num = 0
            else:
                no_best_num += 1

            if no_best_num >= 2:
                tf.logging.info("has no better loss after 2 round...")
                tf.logging.info("end...")
                break
    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.evaluate(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "test_eval_results.txt")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()
