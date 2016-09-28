import argparse, logging, cPickle as pkl, os, codecs, re

from test_model import SentenceGenerator
from commons import ORIG_PREFIX, RESULTS_SUFFIX, SOURCE, TARGET, get_unk_map, \
  FINAL_RESULTS_SUFFIX, replace_line, LM_SCORES_SUFFIX

'''
Commandline arguments
'''
def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('seq2seq_dir', help='Seq2seq data directory')
  parser.add_argument('-candidates', default='data.dev')
  parser.add_argument('-missing', dest='missing', action='store_true', default=False,
                      help='replace unresolved UNK symbols')

  parser.add_argument('-lm_scores', dest='lm_scores', default=False, action='store_true', help='Language Model Scores')
  parser.add_argument('-lm', default=None, help='Language Model')
  parser.add_argument('-lm_data', default='qs_data', help='LM Data path')
  parser.add_argument('-weight', default=0.5, type=float, help='LM Weight')
  parser.add_argument('-debug', dest='debug', help='debug mode', default=False, action='store_true')
  args = parser.parse_args()
  if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)
  return args


'''
Load Seq2seq UNK Candidates
'''
def get_seq2seq_candidates(args):
  candidates_path = os.path.join(args.seq2seq_dir, '%s.%s.%s'%(args.candidates, SOURCE, RESULTS_SUFFIX))
  candidates = pkl.load(open(candidates_path))
  logging.info('Seq2Seq Candidates file: %s #Candidates: %d'%(candidates_path, len(candidates)))
  return candidates

'''
Load inputs, inputs_unk
Needed to generate UNK Symbols
'''
def get_inputs(args):
  inputs_path = os.path.join(args.seq2seq_dir, '%s%s.%s'%(ORIG_PREFIX, args.candidates, SOURCE))
  inputs = codecs.open(inputs_path, 'r', 'utf-8').readlines()

  inputs_unk_path = os.path.join(args.seq2seq_dir, '%s.%s'%(args.candidates, SOURCE))
  inputs_unk = codecs.open(inputs_unk_path, 'r', 'utf-8').readlines()

  logging.info('#Inputs: %d'%len(inputs))
  return inputs, inputs_unk


def fill_missing_symbols(orig_candidates, unk_map):
  unk_candidates = []
  symbol_start = set(['Q', 'A', '_'])

  for orig_candidate in orig_candidates:
    prob, candidate = orig_candidate

    tokens = candidate.split()
    symbols = set([token for token in tokens if token[0] in symbol_start])
    unresolved_symbols = symbols - set(unk_map.keys())
    logging.debug('C:%s Unresolved:%s'%(candidate, str(unresolved_symbols)))

    if len(unresolved_symbols) == 1:
      unresolved_symbol = list(unresolved_symbols)[0]
      unused_symbols = set(unk_map.keys()) - symbols
      for unused_symbol in unused_symbols:
        new_candidate = re.sub(unresolved_symbol, unused_symbol, candidate)
        logging.debug('New Candidate: %s'%new_candidate)
        unk_candidates.append((prob, new_candidate))
    else:
      unk_candidates.append(orig_candidate)

  return unk_candidates


def write_candidates(final_candidates, args):
  candidates_path = os.path.join(args.seq2seq_dir, '%s.%s.%s'%(args.candidates, SOURCE, FINAL_RESULTS_SUFFIX))
  logging.info('Writing Final Candidates to %s'%candidates_path)
  pkl.dump(final_candidates, open(candidates_path, 'w'))


def load_lm(args):
  if args.lm is None:
    return None

  logging.info('Using LM: %s Data: %s'%(args.lm_data, args.lm))
  lm = SentenceGenerator(args.lm_data, args.lm)
  return lm


def combine_and_sort_candidates(seq2seq_candidates, lm_scores, weight):
  assert len(seq2seq_candidates) == len(lm_scores)
  final_candidates = []
  for index in range(len(seq2seq_candidates)):
    candidate = seq2seq_candidates[index][1]
    score_seq2seq = seq2seq_candidates[index][0]
    score_lm  = lm_scores[index]
    score = (weight * score_lm) + ((1 - weight) * score_seq2seq)
    final_candidates.append((score, candidate))

  final_candidates = sorted(final_candidates, key=lambda t:t[0], reverse=True)
  return final_candidates


def get_lm_scores_path(args):
  return os.path.join(args.seq2seq_dir, '%s.%s.%s'%(args.candidates, SOURCE, LM_SCORES_SUFFIX))


def get_lm_scores(args):
  lm_scores_path = get_lm_scores_path(args)
  lm_scores = pkl.load(open(lm_scores_path))

  logging.info('Loaded %d LM scores from %s' % (len(lm_scores), lm_scores_path))
  return lm_scores

def convert_phrases(orig_candidates):
  candidates = []

  for orig_candidate in orig_candidates:
    tokens = orig_candidate[1].split()
    final_tokens = []
    for token in tokens:
      if token[0] in set(['Q', 'A']):
        final_tokens.append(token)
        continue

      sub_tokens = token.split('_')
      if len(sub_tokens) > 1:
        final_tokens.extend(sub_tokens)
      else:
        final_tokens.append(token)

    candidates.append((orig_candidate[0], ' '.join(final_tokens)))

  return candidates


def main():
  args = setup_args()
  seq2seq_unk_candidates = get_seq2seq_candidates(args)
  inputs, inputs_unk = get_inputs(args)
  assert len(inputs) == len(inputs_unk)

  if args.lm_scores:
    all_lm_scores = get_lm_scores(args)
    assert len(all_lm_scores) == len(seq2seq_unk_candidates)
    lm = None
  else:
    all_lm_scores = []
    lm = load_lm(args)

  final_candidates = []
  for index, seq2seq_unk_candidate in enumerate(seq2seq_unk_candidates):
    unk_map = get_unk_map(inputs[index], inputs_unk[index])
    logging.debug('UNK_Map: %s'%str(unk_map))

    if args.missing:
      unk_candidates = fill_missing_symbols(seq2seq_unk_candidate, unk_map)
    else:
      unk_candidates = seq2seq_unk_candidate

    logging.info('Index: %d Orig candidates:%d New:%d' % (index, len(seq2seq_unk_candidate), len(unk_candidates)))
    seq2seq_candidates = [replace_line(line, unk_map) for line in unk_candidates]
    seq2seq_candidates = convert_phrases(seq2seq_candidates)

    if args.lm_scores:
      lm_scores = all_lm_scores[index]
      candidates = combine_and_sort_candidates(seq2seq_candidates, lm_scores, args.weight)
    elif lm:
      lm_scores = [lm.compute_prob(line[1]) for line in seq2seq_candidates]
      all_lm_scores.append(lm_scores)
      candidates = combine_and_sort_candidates(seq2seq_candidates, lm_scores, args.weight)
    else:
      candidates = seq2seq_candidates

    final_candidates.append(candidates)
  write_candidates(final_candidates, args)

  if lm:
    lm_scores_path = get_lm_scores_path(args)
    logging.info('Writing LM Scores to:%s'%lm_scores_path)
    pkl.dump(all_lm_scores, open(lm_scores_path, 'w'))


if __name__ == '__main__':
    main()