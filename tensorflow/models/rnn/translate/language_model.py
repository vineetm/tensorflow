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

from reader import EOS_WORD
from progress.bar import Bar
import tensorflow as tf, os, cPickle as pkl, numpy as np
from nltk.tokenize import word_tokenize as tokenizer
import argparse, codecs
logging = tf.logging
logging.set_verbosity(tf.logging.INFO)
from commons import execute_bleu_command, get_num_lines
from progress.bar import Bar

def data_type():
  return tf.float32

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 1
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 1
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 1
  vocab_size = 10000
  max_len = 50


class LanguageModel(object):
  """The PTB model."""

  def generate_id_to_word(self):
    id_to_word = {}
    for word in self.word_to_id:
      id_to_word[self.word_to_id[word]] = word
    self.id_to_word = id_to_word


  def __init__(self, data_path, model_path, config=LargeConfig, max_refs=10):
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    self.data_path = data_path
    self.model_path = model_path
    vocab_file = os.path.join(data_path, 'vocab.pkl')
    self.word_to_id = pkl.load(open(vocab_file, 'rb'))
    self.generate_id_to_word()
    logging.info('Vocab Size: %d' % len(self.word_to_id))

    with tf.variable_scope("model", reuse=None, initializer=initializer):
      self.batch_size = batch_size = config.batch_size
      self.num_steps = num_steps = config.num_steps
      self.is_decoder = True

      size = config.hidden_size
      vocab_size = config.vocab_size

      self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

      # Slightly better results can be obtained with forget gate biases
      # initialized to 1 but the hyperparameters of the model would need to be
      # different than reported in the paper.

      lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
      cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

      self._initial_state = cell.zero_state(batch_size, data_type())
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
      with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          (cell_output, state) = cell(inputs[:, time_step, :], state)
          outputs.append(cell_output)


      output = tf.reshape(tf.concat(outputs, 1), [-1, size])
      softmax_w = tf.get_variable(
          "softmax_w", [size, vocab_size], dtype=data_type())
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
      self.logits = tf.matmul(output, softmax_w) + softmax_b

      # self.probs = tf.nn.softmax(logits)
      self._final_state = state

    self.session = tf.Session()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(self.model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      logging.error('No Model found at %s'%self.model_path)
      return
    self.max_refs = max_refs

  @property
  def input_data(self):
    return self._input_data

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  def compute_fraction(self, logit, index):
    return np.exp(logit[index]) / np.sum(np.exp(logit))

  def compute_prob(self, sentence):
    # Lower case, tokenize, generate token #s
    sentence = sentence.lower()
    tokens = tokenizer(sentence)
    if len(tokens) <= 1:
      return 0.0

    token_ids = [self.word_to_id[token] for token in tokens]
    token_ids.insert(0, self.word_to_id[EOS_WORD])

    state = self.session.run(self.initial_state)
    total_prob = 0.0

    feed_dict = {}
    for index in range(len(token_ids) - 1):
      fetches = [self.logits, self.final_state]

      #Setup feed_dict
      current_token = token_ids[index]
      next_token = token_ids[index +1]
      feed_dict[self.input_data] = np.array([[current_token]])
      feed_dict[self.initial_state] = state

      logits, state = self.session.run(fetches, feed_dict)
      prob = self.compute_fraction(logits[0], next_token)
      total_prob += prob

    return total_prob / (len(token_ids) - 1)

  def reorder_sentences(self, input_filename, sep, min_prob=0.1):
    fw = codecs.open('%s.lm'%input_filename, 'w', 'utf-8')

    lines = codecs.open(input_filename, 'r', 'utf-8').readlines()

    bar = Bar('LM Reordering', max=len(lines))
    for line in lines:
      parts = line.split(sep)
      parts = [part.strip() for part in parts]

      if len(parts) == 1:
        fw.write(line)
        bar.next()
        continue

      lm_scores = [(part, self.compute_prob(part)) for part in parts[1:]]
      lm_scores = sorted(lm_scores, key=lambda x:x[1], reverse=True)

      lm_scores = [lm_score for lm_score in lm_scores if lm_score[1] > min_prob]
      write_data = []
      write_data.append(parts[0])
      for sorted_variation in lm_scores:
        write_data.append(sorted_variation[0])

      fw.write(sep.join(write_data) + '\n')
      bar.next()

  '''
    Get BLEU score for each hypothesis
    '''

  def get_bleu_scores(self, list_hypothesis, references, suffix=None):

    bleu_scores = []
    # Write all the references
    ref_string = ''
    for index, reference in enumerate(references):
      ref_file = os.path.join(self.model_path, 'ref%d.txt' % index)
      if suffix is not None:
        ref_file = '%s.%s' % (ref_file, suffix)

      with codecs.open(ref_file, 'w', 'utf-8') as f:
        f.write(reference.strip() + '\n')
      ref_string += ' %s' % ref_file

    hyp_file = os.path.join(self.model_path, 'hyp.txt')
    if suffix is not None:
      hyp_file = '%s.%s' % (hyp_file, suffix)

    for index, hypothesis in enumerate(list_hypothesis):
      with codecs.open(hyp_file, 'w', 'utf-8') as f:
        f.write(hypothesis.strip() + '\n')

      bleu = execute_bleu_command(ref_string, hyp_file)
      bleu_scores.append(bleu)
    return bleu_scores

  def merge_models(self, eval_file, report_file, beam_size, model1_tx_fname):
    eval_lines = codecs.open(eval_file, 'r', 'utf-8').readlines()
    model1_tx = pkl.load(open(model1_tx_fname))

    #References file fw
    ref_file_names = [os.path.join('all_ref%d.txt' % index) for index in range(self.max_refs)]
    ref_fw = [codecs.open(file_name, 'w', 'utf-8') for file_name in ref_file_names]

    # Hyp file fw
    hyp_file = os.path.join('all_hyp.txt')
    hyp_fw = codecs.open(hyp_file, 'w', 'utf-8')

    # #Report file fw
    # report_fname = os.path.join(report_file)

    bar = Bar('Merge Models', max=len(eval_lines))
    for index, line in enumerate(eval_lines):
      parts = line.split('\t')

      #Fill in the gold references
      references = [part for part in parts[1:]][:self.max_refs]
      rem_ref = self.max_refs - len(references)
      references.extend(['' for _ in range(rem_ref)])
      for ref_index, reference in enumerate(references):
        ref_fw[ref_index].write(reference.strip() + '\n')

      input_sentence = parts[0]
      all_hypothesis = model1_tx[index][:beam_size]
      list_hypothesis = [hyp[0] for hyp in all_hypothesis]

      bleu_scores = self.get_bleu_scores(list_hypothesis, references)
      if len(bleu_scores) > 0:
        best_index = np.argmax(bleu_scores)
        hyp_fw.write(list_hypothesis[best_index].strip() + '\n')
      else:
        hyp_fw.write('\n')

      bar.next()

    [fw.close() for fw in ref_fw]
    ref_str = ' '.join(ref_file_names)
    final_bleu = execute_bleu_command(ref_str, hyp_file)
    logging.info('Final BLEU: %f'%final_bleu)


DEF_MODEL_DIR='trained-models/lm'


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-input', help='Sentences file')
  parser.add_argument('-model_dir', help='Trained Model Directory', default=DEF_MODEL_DIR)
  parser.add_argument('-sep', default=';', help='Sentence separator')
  parser.add_argument('-min_prob', default='0.1', type=float)

  parser.add_argument('-merge', default=False, action='store_true')
  parser.add_argument('-eval_file', default='eval.data')
  parser.add_argument('-model1_tx', default=None)
  parser.add_argument('-beam_size', type=int, default=16)
  parser.add_argument('-report_file', default='report.txt')
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  lm = LanguageModel(args.model_dir, args.model_dir)

  #Test LM
  test_qs = 'How are you doing?'
  p = lm.compute_prob(test_qs)
  logging.info('Pr(%s): %.2f'%(test_qs, p))

  if not args.merge:
    lm.reorder_sentences(args.input, args.sep, args.min_prob)
  else:
    lm.merge_models(args.eval_file, args.report_file, args.beam_size, args.model1_tx)

if __name__ == '__main__':
  main()
