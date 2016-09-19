import argparse, logging, codecs, os

import cPickle as pkl
import timeit

from translation_model import TranslationModel

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('train')
  parser.add_argument('model')
  parser.add_argument('src_vocab_size', type=int)
  parser.add_argument('target_vocab_size', type=int)
  parser.add_argument('model_size', type=int)
  parser.add_argument('input', help='Input Data')
  parser.add_argument('candidates')
  parser.add_argument('-k', type=int, default=100)
  args = parser.parse_args()
  return args


def read_candidates(args):
  candidates_path = os.path.join(args.train, args.candidates)
  candidates = set()
  for line in codecs.open(candidates_path, 'r', 'utf-8'):
    candidates.add(line.strip())
  return list(candidates)


def main():
  # Logging setup
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  args = setup_args()
  logging.info(args)

  global tm
  model_path = os.path.join(args.train, args.model)
  data_path = os.path.join(args.train, 'data')
  tm = TranslationModel(model_path, data_path, args.src_vocab_size, args.target_vocab_size, args.model_size)

  candidates = read_candidates(args)
  logging.info('Candidates: %d'%len(candidates))

  input_lines = codecs.open(args.input, 'r', 'utf-8').readlines()

  final_results = []

  st = timeit.default_timer()
  for line_num, input_line in enumerate(input_lines):
    st_curr = timeit.default_timer()
    probs = [(tm.compute_prob(input_line, candidate), candidate) for candidate in candidates]
    sorted_probs = sorted(probs, key = lambda t:t[0], reverse=True)[:args.k]
    end_curr = timeit.default_timer()
    logging.info('Line:%d Time:%d'%(line_num, end_curr - st_curr))
    final_results.append(sorted_probs)

  logging.info('Total Time:%d'% (timeit.default_timer() - st))
  pkl.dump(final_results, open(args.input + '.results.pkl', 'w'))


if __name__ == '__main__':
    main()

