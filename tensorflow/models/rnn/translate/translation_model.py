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

  def __init__(self, model_path, data_path, vocab_size, model_size):
    self.session = tf.Session()
    self.model_path = model_path
    self.data_path = data_path

    self.model = Seq2SeqModel(
      source_vocab_size= vocab_size,
      target_vocab_size = vocab_size,
      buckets=_buckets,
      size = model_size,
      num_layers = 1,
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
                                 "vocab%d.en" % vocab_size)
    fr_vocab_path = os.path.join(self.data_path,
                                 "vocab%d.fr" % vocab_size)

    self.en_vocab, _ = initialize_vocabulary(en_vocab_path)
    self.fr_vocab, self.rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

  def compute_prob(self, sentence, output_sentence):
    #st_0 = datetime.now().microsecond
    token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), self.en_vocab, normalize_digits=False)
    #end_0 = datetime.now().microsecond
    #logging.info('Input to tokens: %dus'%(end_0 - st_0))

    #logging.info('Sentence: %s'%sentence)
    #logging.info('Sentence_IDs: %s'%str(token_ids))

    #st_1 = datetime.now().microsecond
    output_token_ids = sentence_to_token_ids(tf.compat.as_bytes(output_sentence), self.fr_vocab, normalize_digits=False)
    #end_1 = datetime.now().microsecond
    #logging.info('Total:%dus Output to tokens: %dus' %((end_1 - st_0), (end_1 - st_1)))

    #logging.info('Output Sentence: %s'%output_sentence)
    #logging.info('Output Token IDs: %s'%str(output_token_ids))

    bucket_id = min([b for b in xrange(len(_buckets))
                     if _buckets[b][0] > len(token_ids)])

    #st_2 = datetime.now().microsecond
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)
    #end_2 = datetime.now().microsecond
    #logging.info('Total:%dus Decoder format: %dus' % ((end_2 - st_0), (end_2 - st_2)))

    #st_3 = datetime.now().microsecond
    for index in range(len(output_token_ids)):
      decoder_inputs[index+1] = np.array([output_token_ids[index]], dtype=np.float32)
    #end_3 = datetime.now().microsecond
    #logging.info('Total:%dus Decoder inputs: %dus' % ((end_3 - st_0), (end_3 - st_3)))

    #logging.info('encoder_inputs Len:%d %s'%(len(encoder_inputs), str(encoder_inputs)))
    #logging.info('decoder_inputs Len:%d %s' % (len(decoder_inputs), str(decoder_inputs)))
    #logging.info('target weights Shape:%d %s' % (len(target_weights), str(target_weights)))

    # Get output logits for the sentence.
    #st_4 = timeit.default_timer()
    _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)

    #end_4 = timeit.default_timer()
    #logging.info('Logits: %ds' % (end_4 - st_4))

    log_prob = 0.0
    st_5 = timeit.default_timer()

    # for index, token_id in enumerate(output_token_ids):
    #   token_probs = self.session.run(tf.nn.softmax(output_logits[index]))
    #   log_prob += np.math.log(token_probs[0][token_id])

    # log_prob = np.sum([np.math.log(self.session.run(tf.nn.softmax(output_logits[index]))[0][output_token_ids[index]])
    #             for index in range(len(output_token_ids))])

    prob = np.sum([self.session.run(tf.nn.softmax(output_logits[index]))[0][output_token_ids[index]]
                       for index in range(len(output_token_ids))]) / len(output_token_ids)

    end_5 = timeit.default_timer()
    # prob = np.math.exp(log_prob)
    #logging.info('Prob computation: %ds' % (end_4 - st_4))

    return prob