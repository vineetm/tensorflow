from language_model import LanguageModel
import cPickle as pkl, os, codecs
import tensorflow as tf, timeit
from commons import DEV_OUTPUT, CANDIDATES_SUFFIX, ALL_HYP, ALL_REF

from commons import get_bleu_score, execute_bleu_command

DEFAULT_LM_MODEL = 'trained/lm/models/qs_lm_large.ckpt'
DEFAULT_LM_DATA  = 'trained/lm/data'

logging = tf.logging


class LMRescorer(object):
  def __init__(self, data_dir):
    self.lm = LanguageModel(DEFAULT_LM_DATA, DEFAULT_LM_MODEL)
    candidates_path = os.path.join(data_dir, '%s.%s'%(DEV_OUTPUT, CANDIDATES_SUFFIX))
    self.candidates = pkl.load(open(candidates_path))
    logging.info('#Candidates: %d'%len(self.candidates))

  def rescore(self, k=100, lm_weight=0.5):
    # fw_all_hyp = codecs.open(ALL_HYP, 'w', 'utf-8')
    # fw_all_ref = codecs.open(ALL_REF, 'w', 'utf-8')

    start_time = timeit.default_timer()
    all_lm_scores = []
    for index, candidate in enumerate(self.candidates):
      start_time_curr = timeit.default_timer()
      lm_scores = [(self.lm.compute_prob(score[1]) ,score[0],score[1]) for score in candidate]
      all_lm_scores.append(lm_scores)
      end_time = timeit.default_timer()
      logging.info('Index: %d Total Time: %ds' % (index, (end_time - start_time_curr)))

    end_time = timeit.default_timer()
    logging.info('Total Time: %ds'%(end_time-start_time))
    return all_lm_scores
