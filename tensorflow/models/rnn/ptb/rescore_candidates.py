import argparse, codecs, logging, re
import cPickle as pkl
import numpy as np

np.random.seed(1543)

from test_model import SentenceGenerator
from commons import replace_line, get_unk_map

RESULTS_SUFFIX = '.results.pkl'
ORIG_PREFIX = 'orig.'
LM_RESULTS_SUFFIX = '.results.lm.pkl'

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', help='Inputs for which BLEU is to be computed')
  parser.add_argument('k', type=int, help='Top K results to consider')
  parser.add_argument('lm')
  parser.add_argument('-data', default='qs_data', help='LM Data path')
  args = parser.parse_args()
  return args

def contains_symbols(line):
  for token in line.split():
    if re.search(r'UNK(\d+)', token):
      return True
  return False

def fill_unknowns(line, cand_words):
  candidates  = []

  unk_tokens = [token for token in line.split() if re.search(r'UNK(\d+)', token)]
  if len(cand_words) < len(unk_tokens):
    return candidates





def resolve_candidates(unk_candidates, unk_map):
  final_candidates = []
  cand_words = set()

  for key in unk_map:
    cand_words.add(unk_map[key])

  for unk_candidate in unk_candidates:
    resolved_line = replace_line(unk_candidate, unk_map)
    if contains_symbols(resolved_line):
      final_candidates.extend(fill_unknowns(resolved_line, cand_words))
    else:
      final_candidates.append(resolved_line)

  return final_candidates


def main():
  # Command line arguments setup
  args = setup_args()
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  # Load Input data, and Input results, Gold_lines
  input_lines = codecs.open(args.inputs, 'r', 'utf-8').readlines()
  orig_input_lines = codecs.open(ORIG_PREFIX + args.inputs, 'r', 'utf-8').readlines()
  input_results = pkl.load(open(args.inputs + RESULTS_SUFFIX, 'r'))

  assert len(input_lines) == len(input_results)
  assert len(orig_input_lines) == len(input_lines)

  #Load LM
  logging.info('Loading LM from data_path:%s model_path:%s'%(args.data, args.lm))
  lm = SentenceGenerator(args.data, args.lm)

  lm_results = []
  for index, result in enumerate(input_results):
    logging.info('Index:%d #Results:%d'%(index, len(result)))

    #Get Original UNK Map
    unk_map = get_unk_map(orig_input_lines[index], input_lines[index])

    #Get Top-K Candidates
    unk_candidates = [candidate[1] for candidate in result[:args.k]]

    resolved_candidates = resolve_candidates(unk_candidates, unk_map)

    lm_candidates = ['<eos> ' + candidate for candidate in resolved_candidates]
    scores = [(lm.compute_prob(candidate), ' '.join(candidate.split()[1:])) for candidate in lm_candidates]
    scores = sorted(scores, key=lambda s: s[0], reverse=True)

    lm_results.append(scores)
    logging.info(index)

  pkl.dump(lm_results, open(args.inputs + LM_RESULTS_SUFFIX, 'wb'))


if __name__ == '__main__':
    main()