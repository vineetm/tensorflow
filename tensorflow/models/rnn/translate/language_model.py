# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from commons import LM_VOCAB_FILE
import tensorflow as tf

logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

def data_type():
  return tf.float32

class LanguageModel(object):
  """The PTB model."""

  def __init__(self, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.is_decoder = True

    size = config.hidden_size
    vocab_size = config.vocab_size

    gpu_device = '/cpu:0'
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.

    gpu_device_name = '%s' % gpu_device
    logging.info('GPU Device: %s' % gpu_device_name)
    with tf.device(gpu_device_name):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

      self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
        "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.device(gpu_device_name), tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    with tf.device(gpu_device_name):
      output = tf.reshape(tf.concat(1, outputs), [-1, size])
      softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
      logits = tf.matmul(output, softmax_w) + softmax_b

    self.probs = tf.nn.softmax(logits)
    self._final_state = state

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    if self.is_decoder:
      return None
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    if self.is_decoder:
      return None
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    if self.is_decoder:
      return None
    return self._train_op


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 30002
