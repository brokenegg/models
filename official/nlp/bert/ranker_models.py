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
"""BERT ranker models that are compatible with TF 2.0."""

import tensorflow as tf
from official.nlp.bert.bert_models import get_transformer_encoder

def ranker_model(bert_config,
                   seq_length,
                   initializer=None,
                   return_core_pretrainer_model=False):
  """Returns model to be used for pre-training.

  Args:
      bert_config: Configuration that defines the core BERT model.
      seq_length: Maximum sequence length of the training data.
      max_predictions_per_seq: Maximum number of tokens in sequence to mask out
        and use for pretraining.
      initializer: Initializer for weights in BertPretrainer.
      use_next_sentence_label: Whether to use the next sentence label.
      return_core_pretrainer_model: Whether to also return the `BertPretrainer`
        object.

  Returns:
      A Tuple of (1) Pretraining model, (2) core BERT submodel from which to
      save weights after pretraining, and (3) optional core `BertPretrainer`
      object if argument `return_core_pretrainer_model` is True.
  """
  input1_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input1_word_ids', dtype=tf.int32)
  input1_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input1_mask', dtype=tf.int32)
  input1_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input1_type_ids', dtype=tf.int32)

  input2_word_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input2_word_ids', dtype=tf.int32)
  input2_mask = tf.keras.layers.Input(
      shape=(seq_length,), name='input2_mask', dtype=tf.int32)
  input2_type_ids = tf.keras.layers.Input(
      shape=(seq_length,), name='input2_type_ids', dtype=tf.int32)

  transformer_encoder = get_transformer_encoder(bert_config, seq_length)
  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  # [<tf.Tensor 'input_word_ids:0' shape=(None, 64) dtype=int32>,
  #  <tf.Tensor 'input_mask:0' shape=(None, 64) dtype=int32>,
  #  <tf.Tensor 'input_type_ids:0' shape=(None, 64) dtype=int32>]
  _, cls_output1 = transformer_encoder([input1_word_ids, input1_mask, input1_type_ids])
  _, cls_output2 = transformer_encoder([input2_word_ids, input2_mask, input2_type_ids])

  output_scores = tf.matmul(cls_output1, cls_output2, transpose_b=True)

  inputs = {
      'input1_ids': input1_word_ids,
      'input1_mask': input1_mask,
      'input1_type_ids': input1_type_ids,
      'input2_ids': input2_word_ids,
      'input2_mask': input2_mask,
      'input2_type_ids': input2_type_ids,
  }

  keras_model = tf.keras.Model(inputs=inputs, outputs=output_scores)
  if return_core_pretrainer_model:
    return keras_model, transformer_encoder, pretrainer_model
  else:
    return keras_model, transformer_encoder