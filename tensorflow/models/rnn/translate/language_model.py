from lm_data_utils import CharsVocabulary
import tensorflow as tf
from lm_1b_eval import _LoadModel
import numpy as np
from nltk.tokenize import word_tokenize as tokenizer

logging = tf.logging
logging.set_verbosity(logging.INFO)

PBTXT = '/Users/vineet/repos/github/mine/models/data/graph-2016-09-10.pbtxt'
CKPT = '/Users/vineet/repos/github/mine/models/data/ckpt-*'
VOCAB_FILE = '/Users/vineet/repos/github/mine/models/data/vocab-2016-09-10.txt'

BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50

class LanguageModel(object):
  def __init__(self, vocab_file=VOCAB_FILE, pbtxt=PBTXT, ckpt=CKPT):

    #Vocab has size 793,471
    self.vocab = CharsVocabulary(vocab_file, MAX_WORD_LEN)
    self.session, self.model = _LoadModel(pbtxt, ckpt)


  def compute_prob(self, sentence):
    tokens = []
    tokens.append('<S>')
    tokens.extend(tokenizer(sentence))
    tokens.append('</S>')

    #Set all variables
    targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
      [BATCH_SIZE, NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)

    probs = []
    for curr_token_index in range(len(tokens) - 1):
      inputs[0, 0] = self.vocab.word_to_id(tokens[curr_token_index])
      char_ids_inputs[0, 0, :] = self.vocab.word_to_char_ids(tokens[curr_token_index])

      softmax = self.session.run(self.model['softmax_out'],
                       feed_dict={self.model['char_inputs_in']: char_ids_inputs,
                                  self.model['inputs_in']: inputs,
                                  self.model['targets_in']: targets,
                                  self.model['target_weights_in']: weights})


      next_token_id = self.vocab.word_to_id(tokens[curr_token_index + 1])
      probs.append(softmax[0][next_token_id])

    #Reset model states
    self.session.run(self.model['states_init'])
    return np.average(probs)

