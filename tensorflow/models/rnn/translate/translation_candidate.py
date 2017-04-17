NOT_SET = -99.0

class Candidate(object):
  def __init__(self, text, seq2seq_score, model):
    self.text = text
    self.seq2seq_score = seq2seq_score
    self.model = model
    self.lm_score = NOT_SET
    self.bleu_score = NOT_SET
    self.final_score = NOT_SET

  '''
  Assign combined score using a linear combination of seq2seq and lm_score
  '''
  def set_final_score(self, lm_weight):
    assert self.lm_score != NOT_SET
    assert self.seq2seq_score != NOT_SET
    assert lm_weight >= 0.0
    assert lm_weight <= 1.0

    assert self.final_score == NOT_SET
    self.final_score = (lm_weight * self.lm_score) + ((1 - lm_weight) * self.seq2seq_score)


  def set_lm_score(self, lm_score):
    self.lm_score = lm_score


  def set_bleu_score(self, bleu_score):
    self.bleu_score = bleu_score


  def str_scores(self):
    return 'Final: %.4f[S: %.4f LM: %.4f] B: %.2f'%(self.final_score, self.seq2seq_score,
                                                    self.lm_score, self.bleu_score)