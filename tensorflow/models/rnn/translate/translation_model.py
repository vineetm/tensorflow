import argparse, os
import numpy as np
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
from translate import _buckets
from data_utils import initialize_vocabulary, sentence_to_token_ids
import timeit

from datetime import datetime
logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

class TranslationModel(object):

  def __init__(self, model_path, data_path, src_vocab_size, target_vocab_size,
               model_size, num_layers=1):
    self.session = tf.Session()
    self.model_path = model_path
    self.data_path = data_path

    self.model = Seq2SeqModel(
      source_vocab_size= src_vocab_size,
      target_vocab_size = target_vocab_size,
      buckets=_buckets,
      size = model_size,
      num_layers = num_layers,
      max_gradient_norm = 5.0,
      batch_size=1,
      learning_rate=0.5,
      learning_rate_decay_factor=0.99,
      compute_prob=True,
      forward_only=True)

    logging.info('Loading model from %s' % self.model_path)
    self.model.saver.restore(self.session, self.model_path)

    # Load vocabularies.
    en_vocab_path = os.path.join(self.data_path,
                                 "vocab%d.en" % src_vocab_size)
    fr_vocab_path = os.path.join(self.data_path,
                                 "vocab%d.fr" % target_vocab_size)

    self.en_vocab, _ = initialize_vocabulary(en_vocab_path)
    self.fr_vocab, self.rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

  def set_output_tokens(self, output_token_ids, decoder_inputs):
    for index in range(len(output_token_ids)):
      if index + 1 >= len(decoder_inputs):
        logging.info('Skip assignment Decoder_Size:%d Outputs_Size:%d'%(len(decoder_inputs), len(output_token_ids)))
        return
      decoder_inputs[index + 1] = np.array([output_token_ids[index]], dtype=np.float32)


  def compute_fraction(self, logit, token_index):
    sum_all = np.sum(np.exp(logit))
    return np.exp(logit[token_index]) / sum_all


  def compute_prob(self, sentence, output_sentence):
    token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), self.en_vocab, normalize_digits=False)
    output_token_ids = sentence_to_token_ids(tf.compat.as_bytes(output_sentence), self.fr_vocab, normalize_digits=False)

    bucket_id = min([b for b in xrange(len(_buckets))
                     if _buckets[b][0] > len(token_ids)])

    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)

    self.set_output_tokens(output_token_ids, decoder_inputs)
    _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)

    if len(decoder_inputs) > len(output_token_ids):
      max_len = len(output_token_ids)
    else:
      max_len = len(decoder_inputs)

    prob = np.sum([self.compute_fraction(output_logits[index][0], output_token_ids[index])
                   for index in range(max_len)]) / max_len

    # prob = np.sum([self.session.run(tf.nn.softmax(output_logits[index]))[0][output_token_ids[index]]
    #                    for index in range(max_len)]) / max_len

    return prob