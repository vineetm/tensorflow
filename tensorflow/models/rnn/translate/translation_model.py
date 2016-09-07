import argparse, os
import numpy as np
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
from translate import _buckets
from data_utils import initialize_vocabulary, sentence_to_token_ids

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
    #logging.info('Sentence: %s'%sentence)
    token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), self.en_vocab, normalize_digits=False)
    #logging.info('Sentence_IDs: %s'%str(token_ids))


    output_token_ids = sentence_to_token_ids(tf.compat.as_bytes(output_sentence), self.fr_vocab, normalize_digits=False)
    #logging.info('Output Sentence: %s'%output_sentence)
    #logging.info('Output Token IDs: %s'%str(output_token_ids))

    bucket_id = min([b for b in xrange(len(_buckets))
                     if _buckets[b][0] > len(token_ids)])

    #logging.info('Bucket id: %d'%bucket_id)

    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)

    for index in range(len(output_token_ids)):
      decoder_inputs[index+1] = np.array([output_token_ids[index]], dtype=np.float32)

    #logging.info('encoder_inputs Len:%d %s'%(len(encoder_inputs), str(encoder_inputs)))
    #logging.info('decoder_inputs Len:%d %s' % (len(decoder_inputs), str(decoder_inputs)))
    #logging.info('target weights Shape:%d %s' % (len(target_weights), str(target_weights)))

    # Get output logits for the sentence.
    _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)


    log_prob = 0.0
    for index, token_id in enumerate(output_token_ids):
      probs = tf.nn.softmax(output_logits[index])
      tokens_probs = self.session.run(probs)
      token_prob = tokens_probs[0][token_id]
      #logging.info('Token[:%d]: %s Prob: %f'%(token_id, self.rev_fr_vocab[token_id], token_prob))
      log_prob += np.math.log(token_prob)

    prob = np.math.exp(log_prob)
    #logging.info('Log Prob: %f Prob: %f'%(log_prob, prob))
    return prob