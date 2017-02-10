from commons import CONFIG_FILE, STOPW_FILE
import os, logging, codecs
import tensorflow as tf
import cPickle as pkl
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary, sentence_to_token_ids, EOS_ID
from collections import OrderedDict
from nltk.tokenize import word_tokenize as tokenizer
import numpy as np

logging = tf.logging

class SequenceGenerator(object):
  def __init__(self, models_dir):
    config_file_path = os.path.join(models_dir, CONFIG_FILE)
    logging.set_verbosity(logging.INFO)

    logging.info('Loading Pre-trained seq2model:%s' % config_file_path)
    config = pkl.load(open(config_file_path))
    logging.info(config)

    #Create session
    self.session = tf.Session()

    #Setup parameters using saved config
    self.model_path = config['train_dir']
    self.data_path = config['data_dir']
    self.src_vocab_size = config['src_vocab_size']
    self.target_vocab_size = config['target_vocab_size']
    self._buckets = config['_buckets']

    #Create model
    self.model = Seq2SeqModel(
      source_vocab_size=config['src_vocab_size'],
      target_vocab_size=config['target_vocab_size'],
      buckets=config['_buckets'],
      size=config['size'],
      num_layers=config['num_layers'],
      max_gradient_norm=config['max_gradient_norm'],
      batch_size=1,
      learning_rate=config['learning_rate'],
      learning_rate_decay_factor=config['learning_rate_decay_factor'],
      forward_only=True,
      compute_prob=False)

    #Restore Model from checkpoint file
    ckpt = tf.train.get_checkpoint_state(self.model_path)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      logging.error('Model not found!')
      return None

    # Load vocabularies
    en_vocab_path = os.path.join(self.data_path,
                                   "vocab%d.en" % self.src_vocab_size)
    fr_vocab_path = os.path.join(self.data_path,
                                   "vocab%d.fr" % self.target_vocab_size)

    self.en_vocab, self.rev_en_vocab = initialize_vocabulary(en_vocab_path)
    self.fr_vocab, self.rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

    self.stopw = set()
    with codecs.open(os.path.join(self.data_path,  STOPW_FILE)) as f:
      for line in f:
        self.stopw.add(line.strip())

    logging.info('#Stopwords: %d'%len(self.stopw))


  def replace_tokens(self, tokens, unk_map):
    replaced_tokens = [unk_map[token] if token in unk_map else token for token in tokens]
    return ' '.join(replaced_tokens)


  def get_unk_map(self, tokens, unk_map=None):
    if unk_map is None:
      unk_map = OrderedDict()
    for token in tokens:
      if token in self.stopw:
        continue
      if token in unk_map:
        continue
      unk_map[token] = '%s%d' % ('UNK', len(unk_map) + 1)

    rev_unk_map = {}
    for token in unk_map:
      rev_unk_map[unk_map[token]] = token

    return unk_map, rev_unk_map


  def generate_output_sequence(self, input_sentence):
    input_tokens = tokenizer(input_sentence)
    unk_map, rev_unk_map = self.get_unk_map(input_tokens)

    unk_sentence = self.replace_tokens(input_tokens, unk_map)
    logging.info('Src: %s'%' '.join(input_tokens))
    logging.info('UNK: %s' %unk_sentence)

    token_ids = sentence_to_token_ids(tf.compat.as_bytes(unk_sentence), self.en_vocab, normalize_digits=False)
    logging.info('Tkn: %s'%str(token_ids))

    bucket_id = min([b for b in xrange(len(self._buckets))
                     if self._buckets[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)


    # Get output logits for the sentence.
    _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)


    # This is a greedy decoder - outputs are just argmaxes of output_logits.

    output_tokens = []
    for output_token_index in xrange(len(output_logits)):
      output_tokens.append(np.argmax(output_logits[output_token_index][0]))
      if EOS_ID in output_tokens:
        output_tokens = output_tokens[:output_tokens.index(EOS_ID)]

    unk_output_sentence = " ".join([tf.compat.as_str(self.rev_fr_vocab[output]) for output in output_tokens])
    logging.info(unk_output_sentence)

    output_sentence = self.replace_tokens(unk_output_sentence.split(), rev_unk_map)
    return output_sentence