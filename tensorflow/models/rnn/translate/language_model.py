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

class Candidate(object):
  def __init__(self, text, seq2seq_score, lm_score, bleu_score, model):
    self.text = text
    self.seq2seq_score = seq2seq_score
    self.lm_score = lm_score
    self.bleu_score = bleu_score
    self.model = model

  def set_lm_score(self, lm_score):
    self.lm_score = lm_score

  def set_bleu_score(self, bleu_score):
    self.bleu_score = bleu_score

  def str_scores(self):
    return 'S: %.2f LM: %.2f B: %.2f'%(self.seq2seq_score, self.lm_score, self.bleu_score)

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


  def __init__(self, data_path, model_path, config=LargeConfig):
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

  def get_bleu_scores(self, exp_dir, eval_candidates, references, suffix=None):

    bleu_scores = []
    # Write all the references
    ref_string = ''
    for index, reference in enumerate(references):
      ref_file = os.path.join(exp_dir, 'ref%d.txt' % index)
      if suffix is not None:
        ref_file = '%s.%s' % (ref_file, suffix)

      with codecs.open(ref_file, 'w', 'utf-8') as f:
        f.write(reference.strip() + '\n')
      ref_string += ' %s' % ref_file

    hyp_file = os.path.join(exp_dir, 'hyp.txt')
    if suffix is not None:
      hyp_file = '%s.%s' % (hyp_file, suffix)

    for eval_candidate in eval_candidates:
      with codecs.open(hyp_file, 'w', 'utf-8') as f:
        f.write(eval_candidate.text.strip() + '\n')

      bleu = execute_bleu_command(ref_string, hyp_file)
      bleu_scores.append(bleu)
    return bleu_scores


  def read_eval_data(self, eval_file):
    fr = codecs.open(eval_file, 'r', 'utf-8')
    eval_lines = fr.readlines()
    fr.close()

    all_parts = [eval_line.split('\t') for eval_line in eval_lines]
    len_parts = [len(parts) for parts in all_parts]
    max_len = max(len_parts)

    return eval_lines, max_len-1

  '''
  First obtain the translation file mapping for each model
  '''
  def fetch_candidate_hypothesis(self, models_file, beam_size):
    candidates = {}
    models_map = {}
    for line in open(models_file):
      if line[0] == '#':
        continue
      parts = line.split(';')
      logging.info(parts)
      models_map[parts[0]] = parts[1].strip()
    logging.info('Models Map: %s'%models_map)


    for model_name in models_map:
      fr = open(models_map[model_name])
      tx = pkl.load(fr)

      for eval_index in tx:
        if eval_index not in candidates:
          candidates[eval_index] = []

        m_candidates = [Candidate(model=model_name, text=c[0], seq2seq_score=c[1],
                                  lm_score=-99.0, bleu_score=-99.0) for c in tx[eval_index][:beam_size]]
        candidates[eval_index].extend(m_candidates)
    return candidates


  def compute_bleu_multiple_references(self, exp_dir, hyp, references):
    if hyp == '' or len(references) == 0:
      return 0.0

    hyp_file = os.path.join(exp_dir, 'temp-hyp.txt')
    ref_files = [os.path.join(exp_dir, 'temp-ref%d.txt' % ref) for ref in range(len(references))]

    ref_fws = [codecs.open(fname, 'w', 'utf-8') for fname in ref_files]
    for index, reference in enumerate(references):
      ref_fws[index].write(reference.strip() + '\n')

    [ref_fw.close() for ref_fw in ref_fws]

    hyp_fw = codecs.open(hyp_file, 'w', 'utf-8')
    hyp_fw.write(hyp.strip() + '\n')
    hyp_fw.close()

    bleu = execute_bleu_command(' '.join(ref_files), hyp_file)
    return bleu


  def merge_models(self, args):
    all_candidates = self.fetch_candidate_hypothesis(args.models_file, args.beam_size)
    logging.info('Num Candidates: %d index[0]:%d'%(len(all_candidates), len(all_candidates[0])))

    eval_index = 0
    eval_precision = []
    eval_recall = []

    report_fw = codecs.open(os.path.join(args.exp_dir, 'report.txt'), 'w', 'utf-8')
    header_data = []
    header_data.append('Input Sentence')
    header_data.append('Avg Precision')
    header_data.append('Avg Recall')

    header_data.append('Best Precision')
    header_data.append('Best Recall')

    for ref_num in range(args.max_refs):
      header_data.append('Ref%d'% ref_num)
      header_data.append('Bleu')

    for cand_num in range(args.beam_size):
      header_data.append('C%d'% cand_num)
      header_data.append('Model')
      header_data.append('Scores')


    report_fw.write('\t'.join(header_data) + '\n')

    for eval_line in codecs.open(args.eval_file, 'r', 'utf-8'):
      write_data = []
      parts = eval_line.split('\t')
      parts = [part.strip() for part in parts]

      input_qs = parts[0]
      write_data.append(input_qs)

      references = parts[1:][:args.max_refs]
      candidate_sentences = [candidate.text for candidate in all_candidates[eval_index]]

      bleu_scores_recall = [self.compute_bleu_multiple_references(args.exp_dir, reference, candidate_sentences) for reference in references]
      avg_recall = np.average(bleu_scores_recall)

      if len(candidate_sentences) == 0:
        avg_precision = 0.0
        arg_max_pr = 0
        max_precision = 0.0
      else:
        bleu_scores_precision = np.array([self.compute_bleu_multiple_references(args.exp_dir, candidate, references) for candidate in
                                 candidate_sentences])
        avg_precision = np.average(bleu_scores_precision)
        arg_max_pr = np.argmax(bleu_scores_precision)
        max_precision = np.max(bleu_scores_precision)

      write_data.append('%.2f' % avg_precision)
      write_data.append('%.2f' % avg_recall)

      write_data.append('%.2f[%d]' %(max_precision, arg_max_pr))
      write_data.append('%.2f' %np.max(bleu_scores_recall))

      # Write Recall Score
      for reference, bleu_score in zip(references, bleu_scores_recall):
        write_data.append('%s' %reference)
        write_data.append('%.2f'%bleu_score)

      eval_precision.append(avg_precision)
      eval_recall.append(avg_recall)

      # Write Precision Score
      if len(bleu_scores_precision) > 0:
        for candidate, bleu_score in zip(all_candidates[eval_index], bleu_scores_precision):
          candidate.set_bleu_score(bleu_score)
          write_data.append('%s'%candidate.text)
          write_data.append('%s'%candidate.model)
          write_data.append('%s'%candidate.str_scores())

      report_fw.write('\t'.join(write_data) + '\n')

      eval_index += 1
      if eval_index == 10:
        return

      if eval_index % 100 == 0:
        logging.info('Completed: %d Average Precision: %.2f Recall: %.2f'
                     % (eval_index, np.average(eval_precision), np.average(eval_recall)))

    logging.info('Average Precision: %.2f Recall: %.2f' % (np.average(eval_precision), np.average(eval_recall)))

DEF_MODEL_DIR='trained-models/lm'


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-input', help='Sentences file')
  parser.add_argument('-model_dir', help='Trained Model Directory', default=DEF_MODEL_DIR)
  parser.add_argument('-sep', default=';', help='Sentence separator')
  parser.add_argument('-min_prob', default='0.1', type=float)
  parser.add_argument('-max_refs', default=16, type=int)

  parser.add_argument('-merge', default=False, action='store_true')
  parser.add_argument('-eval_file', default='eval.data')
  parser.add_argument('-beam_size', type=int, default=16)
  parser.add_argument('-exp_dir', default=None)
  parser.add_argument('-models_file', default=None)
  parser.add_argument('-skip_lm', default=False, action='store_true')
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
    lm.merge_models(args)

if __name__ == '__main__':
  main()
