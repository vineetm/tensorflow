import argparse, codecs, logging
import cPickle as pkl

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
    unk_map = get_unk_map(orig_input_lines[index], input_lines[index])
    replaced_lines = ['<eos> '+ replace_line(line[1], unk_map) for line in result[:args.k]]
    scores = [(lm.compute_prob(candidate), ' '.join(candidate.split()[1:])) for candidate in replaced_lines]
    scores = sorted(scores, key=lambda s: s[0], reverse=True)
    lm_results.append(scores)
    logging.info(index)

  pkl.dump(lm_results, open(args.inputs + LM_RESULTS_SUFFIX, 'wb'))


if __name__ == '__main__':
    main()