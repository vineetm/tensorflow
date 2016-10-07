from language_model import LanguageModel
import cPickle as pkl, os, codecs, numpy as np
import tensorflow as tf, timeit
from commons import DEV_OUTPUT, CANDIDATES_SUFFIX, ALL_HYP, ALL_REF, LM_SCORES_SUFFIX, ORIG_PREFIX

from commons import get_bleu_score, execute_bleu_command, convert_phrase

DEFAULT_LM_MODEL = 'trained/lm/models/qs_lm_large.ckpt'
DEFAULT_LM_DATA  = 'trained/lm/data'

logging = tf.logging


class LMRescorer(object):
  def __init__(self, data_dir):
    self.lm = LanguageModel(DEFAULT_LM_DATA, DEFAULT_LM_MODEL)
    self.data_dir = data_dir
    candidates_path = os.path.join(data_dir, '%s.%s'%(DEV_OUTPUT, CANDIDATES_SUFFIX))
    self.gold_lines = codecs.open(os.path.join(self.data_dir, '%s.%s'%(ORIG_PREFIX, DEV_OUTPUT))).readlines()
    self.candidates = pkl.load(open(candidates_path))
    logging.info('#Gold:%d #Candidates: %d'%(len(self.gold_lines), len(self.candidates)))


  def rescore(self, k=100, lm_weight=0.5):
    fw_all_hyp = codecs.open(ALL_HYP, 'w', 'utf-8')
    fw_all_ref = codecs.open(ALL_REF, 'w', 'utf-8')
    final_scores_path = os.path.join(self.data_dir, '%s.%s' % (DEV_OUTPUT, LM_SCORES_SUFFIX))

    start_time = timeit.default_timer()
    all_final_scores = []
    for index, candidate in enumerate(self.candidates):
      ref_line = convert_phrase(self.gold_lines[index])

      start_time_curr = timeit.default_timer()

      candidate = candidate[:k]
      lm_scores = [(self.lm.compute_prob(score[1]), score[0], score[1]) for score in candidate]
      combined_scores = [(((lm_weight * score[0]) + ((1 - lm_weight) * score[1])), score[0], score[1], score[2])
                         for score in lm_scores]

      combined_scores = sorted(combined_scores, key=lambda t:t[0], reverse=True)
      bleu_scores = [get_bleu_score(ref_line, score[-1]) for score in combined_scores]
      final_scores = zip(bleu_scores, combined_scores)
      best_bleu_index = np.argmax(bleu_scores)
      best_bleu_score = bleu_scores[best_bleu_index]
      hyp_line = final_scores[best_bleu_index][1][-1]
      logging.info('Line:%d Best_BLEU:%f(%d)' % (index, best_bleu_score, best_bleu_index))

      fw_all_hyp.write(hyp_line.strip() + '\n')
      fw_all_ref.write(ref_line.strip() + '\n')

      all_final_scores.append(final_scores)
      end_time = timeit.default_timer()
      logging.info('Index: %d Total Time: %ds' % (index, (end_time - start_time_curr)))

    fw_all_hyp.close()
    fw_all_ref.close()


    end_time = timeit.default_timer()
    logging.info('Total Time: %ds'%(end_time-start_time))

    # return all_final_scores
    pkl.dump(all_final_scores, open(final_scores_path, 'w'))
    bleu_score = execute_bleu_command(ALL_REF, ALL_HYP)
    return bleu_score