from __future__ import division
from __future__ import print_function

import tensorflow as tf, sys
import numpy as np
import os
from language_model import LanguageModel, LargeConfig
from nltk.tokenize import word_tokenize as tokenizer

import cPickle as pkl

logging = tf.logging
logging.set_verbosity(tf.logging.INFO)


class SentenceGenerator(object):
  #Id -> word dictionary
  def generate_id_to_word(self):
    id_to_word = {}
    for word in self.word_to_id:
      id_to_word[self.word_to_id[word]] = word
    self.id_to_word = id_to_word

  def __init__(self, data_path, model_path):
    vocab_file = os.path.join(data_path, 'vocab.pkl')
    self.word_to_id = pkl.load(open(vocab_file, 'rb'))
    self.generate_id_to_word()
    logging.info('Vocab Size: %d' % len(self.word_to_id))

    self.model_path = model_path
    self.config = LargeConfig
    self.config.batch_size = 1
    self.config.num_steps = 1
    self.config.max_len = 50
    initializer = tf.random_uniform_initializer(-self.config.init_scale,
                                                self.config.init_scale)

    # Define Decoder Model
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      self.model = LanguageModel(config=self.config)

    self.session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(self.session, self.model_path)


  def compute_prob(self, sentence):
    # Lower case, tokenize, generate token #s
    sentence = sentence.lower()
    sentence  = '<eos> ' + sentence
    tokens = tokenizer(sentence)
    token_ids = [self.word_to_id[token] if token in self.word_to_id
                 else self.word_to_id['_UNK']
                 for token in tokens]

    state = self.session.run(self.model.initial_state)
    total_prob = 1.0
    for index, token_id in enumerate(token_ids):
      if index == len(token_ids) - 1:
        return total_prob / len(token_ids)

      fetches = [self.model.probs, self.model.final_state]

      feed_dict = {}
      x = np.array([[token_id]])
      feed_dict[self.model.input_data] = x
      feed_dict[self.model.initial_state] = state

      probs, state = self.session.run(fetches, feed_dict)
      total_prob += probs[0][token_ids[index + 1]]