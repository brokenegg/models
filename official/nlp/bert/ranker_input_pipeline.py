# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""BERT model input pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.sparse.to_dense(t)
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


def single_file_dataset(input_file, name_to_features):
  """Creates a single-file dataset to be passed for BERT custom training."""
  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  d = tf.data.TFRecordDataset(input_file)
  d = d.map(
      lambda record: decode_record(record, name_to_features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # When `input_file` is a path to a single file or a list
  # containing a single path, disable auto sharding so that
  # same input file is sent to all workers.
  if isinstance(input_file, str) or len(input_file) == 1:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    d = d.with_options(options)
  return d


def _load_records(filename):
  """Read file and return a dataset of tf.Examples."""
  return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)


def _filter_max_length(example, max_length=256):
  """Indicates whether the example's length is lower than the maximum length."""
  return tf.logical_and(tf.size(example['inputs']) <= max_length,
                        tf.size(example['targets']) <= max_length)


def create_ranker_dataset(input_patterns,
                            seq_length,
                            batch_size,
                            is_training=True,
                            input_pipeline_context=None,
                            use_next_sentence_label=True,
                            use_position_id=False,
                            output_fake_labels=True):

  for input_pattern in input_patterns:
    if not tf.io.gfile.glob(input_pattern):
      raise ValueError('%s does not match any files.' % input_pattern)

  input_files = []
  for input_pattern in input_patterns:
    input_files.extend(tf.io.gfile.glob(input_pattern))
  shuffle_buffer_size = len(input_files)

  def _create_dataset_internal(input_pattern,
                              input1_type_id,
                              input2_type_id,
                              append_input1_to_input2):
    """Creates input dataset from (tf)records files for pretraining."""
    name_to_features = {
        "inputs": tf.io.VarLenFeature(tf.int64),
        "targets": tf.io.VarLenFeature(tf.int64)
    }

    dataset = tf.data.Dataset.list_files(input_pattern, shuffle=is_training)

    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                              input_pipeline_context.input_pipeline_id)
    if is_training:
      dataset = dataset.repeat()

      # We set shuffle buffer to exactly match total number of
      # training files to ensure that training data is well shuffled.
      dataset = dataset.shuffle(shuffle_buffer_size)

    # In parallel, create tf record dataset for each train files.
    # cycle_length = 8 means that up to 8 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      dataset = dataset.shuffle(100)

    decode_fn = lambda record: decode_record(record, name_to_features)
    dataset = dataset.map(
        decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Remove examples where the input or target length exceeds the maximum length,
    dataset = dataset.filter(lambda example: _filter_max_length(example, seq_length))

    def _select_data_from_record(record):
      """Filter out features to use for pretraining."""
      inputs = tf.concat([tf.constant([1], dtype=tf.int64), record['inputs']], axis=0)
      targets = tf.concat([tf.constant([1], dtype=tf.int64), record['targets']], axis=0)
      x = {
          'input1_ids': inputs,
          'input1_mask': tf.ones_like(inputs),
          'input1_type_ids': tf.ones_like(inputs),
          'input2_ids': targets,
          'input2_mask': tf.ones_like(targets),
          'input2_type_ids': tf.ones_like(targets),
      }
      return x

    dataset = dataset.map(
        _select_data_from_record,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size,
        {
          'input1_ids': [seq_length],
          'input1_mask': [seq_length],
          'input1_type_ids': [seq_length],
          'input2_ids': [seq_length],
          'input2_mask': [seq_length],
          'input2_type_ids': [seq_length],
        },
        drop_remainder=is_training)

    def _may_append_input1_to_input2(example):
      input1_ids = example['input1_ids']
      input1_mask = example['input1_mask']
      input1_type_ids = example['input1_type_ids']
      input2_ids = example['input2_ids']
      input2_mask = example['input2_mask']
      input2_type_ids = example['input2_type_ids']
      if append_input1_to_input2:
        input2_ids = tf.concat([input2_ids, input1_ids], axis=0)
        input2_mask = tf.concat([input2_mask, input1_mask], axis=0)
        input2_type_ids = tf.concat([input2_type_ids, input1_type_ids], axis=0)

      return {
          'input1_ids': input1_ids,
          'input1_mask': input1_mask,
          'input1_type_ids': input1_type_ids * input1_type_id,
          'input2_ids': input2_ids,
          'input2_mask': input2_mask,
          'input2_type_ids': input2_type_ids * input2_type_id,
      }

    dataset = dataset.map(_may_append_input1_to_input2)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda example: (example, tf.range(tf.shape(example['input1_ids'])[0])),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

  assert len(input_patterns) == 4
  input_patterns = input_patterns
  input1_type_ids = [0, 0, 0, 0]
  input2_type_ids = [0, 0, 1, 1]
  append_input1_to_input2 = [False, False, True, True]

  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(input_patterns, dtype=tf.string),
     tf.constant(input1_type_ids, dtype=tf.int32),
     tf.constant(input2_type_ids, dtype=tf.int32),
     tf.constant(append_input1_to_input2, dtype=tf.bool),
     ))

  dataset = dataset.interleave(_create_dataset_internal, cycle_length=4, block_length=1)

  return dataset


def create_classifier_dataset(file_path,
                              seq_length,
                              batch_size,
                              is_training=True,
                              input_pipeline_context=None,
                              label_type=tf.int64,
                              include_sample_weights=False):
  """Creates input dataset from (tf)records files for train/eval."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], label_type),
  }
  if include_sample_weights:
    name_to_features['weight'] = tf.io.FixedLenFeature([], tf.float32)
  dataset = single_file_dataset(file_path, name_to_features)

  # The dataset is always sharded by number of hosts.
  # num_input_pipelines is the number of hosts rather than number of cores.
  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  def _select_data_from_record(record):
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    y = record['label_ids']
    if include_sample_weights:
      w = record['weight']
      return (x, y, w)
    return (x, y)

  if is_training:
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset