from __future__ import division
from __future__ import print_function

import tensorflow as tf, sys
import numpy as np
import time, os
from ptb_word_lm import PTBModel, LargeConfig
from nltk.tokenize import word_tokenize as tokenizer

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

    self.session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(self.session, self.model_path)


  def sample(self, probs, keywords):
    if keywords is None:
      return self.generate_sample(probs)

    #Candidate keywords + EOS
    probs_keywords = np.zeros(len(keywords) + 1)
    keywords_list = list(keywords)
    eos_index = len(keywords_list)

    for index, keyword in enumerate(keywords_list):
      probs_keywords[index] = probs[keyword]
    probs_keywords[eos_index] = probs[self.word_to_id['<eos>']]

    sample_id = self.generate_sample(probs_keywords)
    if sample_id == eos_index:
      return self.word_to_id['<eos>']
    else:
      return keywords_list[sample_id]


  def generate_samples_multiple(self, partial_probs):
    sum_probs = np.sum(partial_probs)
    scale = 1.0 / sum_probs
    probs = partial_probs * scale

  def generate_sample(self, a, temperature=1.0):

    probs = a

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


  def compute_prob(self, sentence):
    # Lower case, tokenize, generate token #s
    sentence = sentence.lower()
    tokens = tokenizer(sentence)
    token_ids = [self.word_to_id[token] if token in self.word_to_id
                 else self.word_to_id['<unk>']
                 for token in tokens]

    state = self.session.run(self.model.initial_state)

    sentence_prob = 1.0
    for index, token_id in enumerate(token_ids):
      if index == len(token_ids) - 1:
        return sentence_prob

      fetches = [self.model.probs, self.model.final_state]

      feed_dict = {}
      x = np.array([[token_id]])
      feed_dict[self.model.input_data] = x
      feed_dict[self.model.initial_state] = state

      probs, state = self.session.run(fetches, feed_dict)
      sentence_prob *= probs[0][token_ids[index + 1]]

    return sentence_prob


  def generate_sentence(self, start_token_id, session, keywords=None):
    curr_keywords = None
    if keywords is not None:
      curr_keywords = set()
      for keyword in keywords:
        curr_keywords.add(keyword)

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
      new_token_id = self.sample(probs[0], curr_keywords)

      #Keep on sampling till you get a word in desired set
      if curr_keywords is not None:
        if self.id_to_word[new_token_id] == '<eos>':
          return self.get_sentence(sentence_tokens)

        curr_keywords.remove(new_token_id)
        if len(curr_keywords) == 0:
          return self.get_sentence(sentence_tokens)


      if self.id_to_word[new_token_id] == '<eos>':
        return self.get_sentence(sentence_tokens)

      sentence_tokens.append(new_token_id)
      start_token_id = new_token_id

      if len(sentence_tokens) > self.config.max_len:
        logging.warning('Already found %d tokens, exiting!'%self.config.max_len)
        return self.get_sentence(sentence_tokens)

    return self.get_sentence(sentence_tokens)

  def generate_sentences(self, probe_word, keywords_text=None, num_sentences=10):
    generated_sentences = 0

    if keywords_text is not None:
      keywords = keywords_text.lower()
      keywords = tokenizer(keywords)

      keywords = [self.word_to_id[word] if word in self.word_to_id else self.word_to_id['<unk>'] for word in keywords]
      keywords = set(keywords)

      keywords_vocab = [self.id_to_word[keyword] for keyword in keywords]
      logging.info('Keywords in vocab: %s'% ' '.join(keywords_vocab))

      if self.word_to_id[probe_word] in keywords:
        keywords.remove(self.word_to_id[probe_word])
      #stopw = [self.word_to_id[word] for word in STOPW if word in self.word_to_id]
      #keywords = set(keywords) | set(stopw)

    if probe_word not in self.word_to_id:
      word_id = self.word_to_id['<unk>']
    else:
      word_id = self.word_to_id[probe_word]

    logging.info('')
    start_token_id = word_id
    sentences = []
    set_sentences = set()

    cf = tf.ConfigProto()
    cf.gpu_options.allocator_type = 'BFC'
    #cf.log_device_placement=True
    cf.gpu_options.allow_growth = True

    with tf.Session(config = cf) as session:
      saver = tf.train.Saver()
      saver.restore(session, self.model_path)

      num_repetitions = 0
      while len(sentences) < num_sentences:
        #Restore Original Keywords
        sentence = self.generate_sentence(start_token_id, session, keywords=keywords)
        if sentence not in set_sentences:
          sentences.append(sentence)
          set_sentences.add(sentence)
          logging.info('S%d(R%d): %s'%(generated_sentences, num_repetitions, sentence))
          generated_sentences += 1
          num_repetitions = 0
        else:
          num_repetitions += 1
          if num_repetitions > 1000:
            logging.info('1000 repetitions, exiting!')
            return sentences

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


