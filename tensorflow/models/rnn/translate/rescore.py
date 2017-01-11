from language_model import LanguageModel
import cPickle as pkl, codecs
from candidate_generator import Score, NSUResult
import tensorflow as tf, os
from commons import merge_and_sort_scores, execute_bleu_command, convert_phrase
from commons import CANDIDATES_SUFFIX, TEST

logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

import argparse

DEFAULT_LM_MODEL = 'trained/lm/data-2M/models/qs_lm_large.ckpt'
DEFAULT_LM_DATA  = 'trained/lm/data-2M/data'
DEFAULT_LM_SCORES = 'lm_scores.pkl'

ALL_HYP = 'lm_hyp.txt'
ALL_REF = 'lm_ref.txt'

class FinalScore(object):
    def __init__(self, score, lm_score):
        self.score = score
        self.lm_score = lm_score

    def reweight_score(self, weight):
        self.weighted_score = ((1.0 - weight) * self.lm_score) + (weight * self.score.seq2seq_score)

    def __str__(self):
        return 'C    : %s\nC_UNK: %s\nS:%f B:%f LM:%f' % (self.score.candidate,
                                                        self.score.candidate_unk, self.score.seq2seq_score,
                                                              self.score.bleu_score, self.lm_score)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Trained seq2 model Directory')
    parser.add_argument('-lm_model', help='Trained Language model', default=DEFAULT_LM_MODEL)
    parser.add_argument('-lm_data', help='Language model data', default=DEFAULT_LM_DATA)
    parser.add_argument('-lm_scores', help='LM scores file', default=DEFAULT_LM_SCORES)
    parser.add_argument('-index', default=-1, type=int)
    parser.add_argument('-weight', default=0.5, type=float)
    parser.add_argument('-k', default=100, type=int)
    parser.add_argument('-test', dest='test', default=False, action='store_true')
    args = parser.parse_args()
    return args


class Rescorer(object):
    def __init__(self, seq2seq_dir, lm_data=DEFAULT_LM_DATA, lm_model=DEFAULT_LM_MODEL, test=False):
        self.lm = LanguageModel(lm_data, lm_model)
        self.base_dir = seq2seq_dir
        seq2seq_results_file = os.path.join(self.base_dir, CANDIDATES_SUFFIX)
        if test:
            seq2seq_results_file = '%s.%s'%(seq2seq_results_file, TEST)

        self.seq2seq_results = pkl.load(open(seq2seq_results_file))
        logging.info('Loaded %d saved seq2scores:%s'%(len(self.seq2seq_results), seq2seq_results_file))


    def get_lm_scores(self, score_index):
        final_scores = []
        seq2seq_scores = merge_and_sort_scores(self.seq2seq_results[score_index])

        for index, seq2seq_score in enumerate(seq2seq_scores):
            lm_score = self.lm.compute_prob(convert_phrase(seq2seq_score.candidate))
            final_score = FinalScore(seq2seq_score, lm_score)
            final_scores.append(final_score)
            if index % 100 == 0:
                logging.info('LM_scores :%d' % index)
        return final_scores

    def _get_best_lm_index(self, lm_scores):
        max_index = -1
        max_score = -99.0

        for index, score in enumerate(lm_scores):
            if score.lm_score > max_score:
                max_score = score.lm_score
                max_index = index

        return max_score, max_index

    def _get_best_seq2seq_index(self, lm_scores):
        max_index = -1
        max_score = -99.0

        for index, score in enumerate(lm_scores):
            if score.score.seq2seq_score > max_score:
                max_score = score.score.seq2seq_score
                max_index = index

        return max_score, max_index

    def _get_best_bleu_index(self, lm_scores):
        max_index = -1
        max_score = -99.0

        for index, score in enumerate(lm_scores):
            if score.score.bleu_score > max_score:
                max_score = score.score.bleu_score
                max_index = index

        return max_score, max_index


    def reorder_scores(self, final_lm_scores, weight):
        final_sorted_scores = []
        for lm_scores in final_lm_scores:
            for lm_score in lm_scores:
                lm_score.reweight_score(weight)

            sorted_scores = sorted(lm_scores, key = lambda t:t.weighted_score, reverse=True)
            final_sorted_scores.append(sorted_scores)
        return final_sorted_scores


def main():
    args = setup_args()
    logging.info(args)
    rescorer = Rescorer(args.model_dir, test=args.test, lm_data=args.lm_data, lm_model=args.lm_model)

    final_lm_scores = []
    lm_scores_file = os.path.join(rescorer.base_dir, args.lm_scores)
    if args.test:
        lm_scores_file = '%s.%s'%(lm_scores_file, TEST)

    if os.path.exists(lm_scores_file):
        final_lm_scores = pkl.load(open(lm_scores_file))
        logging.info('Loaded %d LM_Scores from %s'%(len(final_lm_scores), lm_scores_file))
    else:
        if args.index == -1:
            num_lines = len(rescorer.seq2seq_results)
        else:
            num_lines = args.index

        logging.info('Getting LM Scores for %d lines'%num_lines)
        for line_index in range(num_lines):
            lm_scores = rescorer.get_lm_scores(line_index)
            final_lm_scores.append(lm_scores)
            logging.info('Line: %d lm_score done'%line_index)
        pkl.dump(final_lm_scores, open(lm_scores_file, 'w'))

    reordered_lm_scores = rescorer.reorder_scores(final_lm_scores, weight=args.weight)
    all_hyp_file = os.path.join(rescorer.base_dir, ALL_HYP)
    all_ref_file = os.path.join(rescorer.base_dir, ALL_REF)
    if args.test:
        all_hyp_file = '%s.%s'%(all_hyp_file, TEST)
        all_ref_file = '%s.%s' % (all_ref_file, TEST)

    logging.info('BLEU: Ref_file: %s Hyp_file: %s'%(all_ref_file, all_hyp_file))
    fw_all_hyp = codecs.open(all_hyp_file, 'w', 'utf-8')
    fw_all_ref = codecs.open(all_ref_file, 'w', 'utf-8')


    for index, final_scores in enumerate(reordered_lm_scores):
        final_scores = final_scores[:args.k]
        best_bleu_score, best_bleu_index = rescorer._get_best_bleu_index(final_scores)
        logging.info('Index:%d Best_BLEU:%f(%d)'%(index, best_bleu_score, best_bleu_index))
        fw_all_hyp.write(convert_phrase(final_scores[best_bleu_index].score.candidate.strip()) + '\n')
        fw_all_ref.write(rescorer.seq2seq_results[index].gold_line.strip() + '\n')

    bleu_score = execute_bleu_command(all_ref_file, all_hyp_file)
    logging.info('Final BLEU: %f'%bleu_score)

if __name__ == '__main__':
    main()