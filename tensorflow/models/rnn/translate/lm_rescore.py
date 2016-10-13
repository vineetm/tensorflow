from language_model import LanguageModel
import cPickle as pkl, os, codecs, numpy as np
import tensorflow as tf, timeit
from commons import DEV_OUTPUT, DEV_INPUT, CANDIDATES_SUFFIX, ALL_HYP, ALL_REF, LM_SCORES_SUFFIX, ORIG_PREFIX, FINAL_SCORES_SUFFIX

from commons import get_bleu_score, execute_bleu_command, convert_phrase

DEFAULT_LM_MODEL = 'trained/lm/models/qs_lm_large.ckpt'
DEFAULT_LM_DATA  = 'trained/lm/data'

LM_SCORES_FILE = ''

logging = tf.logging

class Score(object):

  def __init__(self, candidate=None, lm_score=0.0, seq2seq_score=0.0, final_score=-1.0, bleu_score=-1.0, score=None):
    if score is None:
      self.lm_score = lm_score
      self.seq2seq_score = seq2seq_score
      self.final_score = final_score
      self.bleu_score = bleu_score
      self.candidate = candidate
    else:
      self.lm_score = score.lm_score
      self.seq2seq_score = score.seq2seq_score
      if score.final_score != -1:
        self.final_score = score.final_score
      else:
        self.final_score = final_score
      self.bleu_score = bleu_score
      self.candidate = score.candidate


  def __str__(self):
    return 'C:%s BLEU:%f Final:%f LM:%f seq:%f'%(self.candidate, self.bleu_score, self.final_score,
                                                 self.lm_score, self.seq2seq_score)


class LMRescorer(object):
  def __init__(self, data_dir):
    candidates_path = os.path.join(data_dir, '%s.%s' % (DEV_OUTPUT, CANDIDATES_SUFFIX))
    self.candidates = pkl.load(open(candidates_path))

    self.lm = LanguageModel(DEFAULT_LM_DATA, DEFAULT_LM_MODEL)
    self.data_dir = data_dir
    self.gold_lines = codecs.open(os.path.join(self.data_dir, '%s.%s'%(ORIG_PREFIX, DEV_OUTPUT))).readlines()
    self.gold_unk_lines = codecs.open(os.path.join(self.data_dir, '%s' % DEV_OUTPUT)).readlines()

    self.input_lines = codecs.open(os.path.join(self.data_dir, '%s.%s' % (ORIG_PREFIX, DEV_INPUT))).readlines()
    self.input_unk_lines = codecs.open(os.path.join(self.data_dir, '%s'% DEV_INPUT)).readlines()

    logging.info('#Gold:%d #Candidates: %d'%(len(self.gold_lines), len(self.candidates)))

    self.lm_scores_path = os.path.join(data_dir, '%s.%s'%(DEV_OUTPUT, LM_SCORES_SUFFIX))
    self.final_scores_path = os.path.join(self.data_dir, '%s.%s' % (DEV_OUTPUT, FINAL_SCORES_SUFFIX))


  def save_lm_scores(self):
    all_lm_scores = []

    for index, candidate in enumerate(self.candidates):
      lm_scores = [Score(lm_score=self.lm.compute_prob(score[1]), candidate=score[-1], seq2seq_score=score[0]) for score in candidate]
      # lm_scores = [(self.lm.compute_prob(score[1]), score[0], score[1]) for score in candidate]
      all_lm_scores.append(lm_scores)

      if index % 2 == 0:
        logging.info('Done: %d'%index)

    pkl.dump(all_lm_scores, open(self.lm_scores_path, 'w'))


  def rescore(self, k=100, lm_weight=0.5):
    all_lm_scores = pkl.load(open(self.lm_scores_path))
    logging.info('Loaded %d LM Scores'%len(all_lm_scores))

    fw_all_hyp = codecs.open(ALL_HYP, 'w', 'utf-8')
    fw_all_ref = codecs.open(ALL_REF, 'w', 'utf-8')

    all_final_scores = []
    for index, candidate in enumerate(self.candidates):
      ref_line = convert_phrase(self.gold_lines[index])
      lm_scores = all_lm_scores[index]

      combined_scores = [Score(score=score, final_score=((lm_weight * score.lm_score) + ((1.0 - lm_weight) * score.seq2seq_score)))
                         for score in lm_scores]

      combined_scores = sorted(combined_scores, key=lambda t:t.final_score, reverse=True)
      combined_scores = combined_scores[:k]

      bleu_scores = [get_bleu_score(ref_line, score.candidate) for score in combined_scores]
      final_scores = [Score(score=combined_scores[b_index], bleu_score=bleu_scores[b_index])
        for b_index in range(len(bleu_scores))]

      best_bleu_index = np.argmax(bleu_scores)
      best_bleu_score = bleu_scores[best_bleu_index]
      hyp_line = final_scores[best_bleu_index].candidate
      logging.info('Line:%d Best_BLEU:%f(%d)' % (index, best_bleu_score, best_bleu_index))

      fw_all_hyp.write(hyp_line.strip() + '\n')
      fw_all_ref.write(ref_line.strip() + '\n')

      all_final_scores.append(final_scores)

    fw_all_hyp.close()
    fw_all_ref.close()

    # return all_final_scores
    with open(self.final_scores_path, 'w') as f:
      pkl.dump(all_final_scores, f)

    bleu_score = execute_bleu_command(ALL_REF, ALL_HYP)
    return bleu_score