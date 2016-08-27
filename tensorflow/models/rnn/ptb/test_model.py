from __future__ import division
from __future__ import print_function

import tensorflow as tf, sys
import numpy as np
import time, os
from ptb_word_lm import PTBModel, LargeConfig

import cPickle as pkl

flags = tf.flags
flags.DEFINE_string("probe_word", None, "probe_word")
FLAGS = flags.FLAGS

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
      self.model = PTBModel(is_training=False, config=self.config, is_decoder=True)

  def sample(self, a, temperature=1.0):
    probs = a[0]

    # helper function to sample an index from a probability array
    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))

    r = np.random.random()
    total = 0.0
    for i in range(len(probs)):
      total += probs[i]
      if total > r:
        return i
    return len(probs) - 1


  def get_sentence(self, token_ids):
    return ' '.join([self.id_to_word[token_id] for token_id in token_ids])


  def generate_sentence(self, start_token_id, session):
    sentence_tokens = []
    sentence_tokens.append(start_token_id)
    state = session.run(self.model.initial_state)

    while True:
      x = np.array([[start_token_id]])
      fetches = [self.model.probs, self.model.final_state]
      feed_dict = {}
      feed_dict[self.model.input_data] = x
      feed_dict[self.model.initial_state] = state

      probs, state = session.run(fetches, feed_dict)
      new_token_id = self.sample(probs)

      if self.id_to_word[new_token_id] == '<eos>':
        return self.get_sentence(sentence_tokens)

      sentence_tokens.append(new_token_id)
      start_token_id = new_token_id

      if len(sentence_tokens) > self.config.max_len:
        logging.warning('Already found %d tokens, exiting!'%self.config.max_len)
        return self.get_sentence(sentence_tokens)

    return self.get_sentence(sentence_tokens)

  def generate_sentences(self, probe_word, num_sentences=10):
    if probe_word not in self.word_to_id:
      word_id = self.word_to_id['<unk>']
    else:
      word_id = self.word_to_id[probe_word]

    start_token_id = word_id
    sentences = []
    set_sentences = set()

    cf = tf.ConfigProto()
    cf.gpu_options.allocator_type = 'BFC'
    cf.log_device_placement=True
    cf.gpu_options.allow_growth = True


    with tf.Session(config = cf) as session:
      saver = tf.train.Saver()
      saver.restore(session, self.model_path)

      while len(sentences) < num_sentences:
        sentence = self.generate_sentence(start_token_id, session)
        if sentence not in set_sentences:
          sentences.append(sentence)
          set_sentences.add(sentence)

    return sentences


def main(_):
  data_path = FLAGS.data_path
  model_path = FLAGS.model_path
  probe_word = FLAGS.probe_word

  sg = SentenceGenerator(data_path, model_path)
  sentences = sg.generate_sentences(probe_word, 100)
  for sentence in sentences:
    logging.info(sentence)

if __name__ == "__main__":
  tf.app.run()
