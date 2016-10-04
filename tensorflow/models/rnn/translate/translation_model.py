import os
import numpy as np
import tensorflow as tf
import cPickle as pkl
from seq2seq_model import Seq2SeqModel
from translate import _buckets
from data_utils import initialize_vocabulary, sentence_to_token_ids
from commons import CONFIG_FILE
logging = tf.logging
logging.set_verbosity(tf.logging.INFO)


class TranslationModel(object):

  def __init__(self, models_dir):
    config_file_path = os.path.join(models_dir, CONFIG_FILE)
    logging.info('Loading Pre-trained seq2model:%s'%config_file_path)

    config = pkl.load(open(config_file_path))
    logging.info(config)

    self.session = tf.Session()
    self.model_path = config['train_dir']
    self.data_path = config['data_dir']
    self.src_vocab_size = config['src_vocab_size']
    self.target_vocab_size = config['target_vocab_size']

    self.model = Seq2SeqModel(
      source_vocab_size = config['src_vocab_size'],
      target_vocab_size = config['target_vocab_size'],
      buckets=config['_buckets'],
      size = config['size'],
      num_layers = config['num_layers'],
      max_gradient_norm = config['max_gradient_norm'],
      batch_size=1,
      learning_rate=config['learning_rate'],
      learning_rate_decay_factor=config['learning_rate_decay_factor'],
      compute_prob=True,
      forward_only=True)

    ckpt = tf.train.get_checkpoint_state(self.model_path)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      logging.info('Model not found!')
      return None


    # Load vocabularies.
    en_vocab_path = os.path.join(self.data_path,
                                 "vocab%d.en" % self.src_vocab_size)
    fr_vocab_path = os.path.join(self.data_path,
                                 "vocab%d.fr" % self.target_vocab_size)

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
    return prob