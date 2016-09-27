import argparse, codecs, logging, os
import cPickle as pkl


from test_model import SentenceGenerator
from commons import get_unk_map, replace_line

ORIG_PREFIX = 'orig.'

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('lm', help='Trained Language Model')
  parser.add_argument('dir', help='Data Dir')
  parser.add_argument('candidates', help='Candidates')
  parser.add_argument('input', help='input file')
  parser.add_argument('-data', default='qs_data', help='LM Data path')
  args = parser.parse_args()
  return args


def main():
  # Command line arguments setup
  args = setup_args()
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  #Load LM
  logging.info('Loading LM from data_path:%s model_path:%s'%(args.data, args.lm))
  lm = SentenceGenerator(args.data, args.lm)

  #Read Candidates, and rescore them as per LM Scores
  lm_results = []

  candidates_path = os.path.join(args.dir, args.candidates)
  results = pkl.load(open(candidates_path, 'r'))
  logging.info('#Orig Results: %d'%len(results))

  input_lines_path = os.path.join(args.dir, args.input)
  input_lines = codecs.open(input_lines_path, 'r', 'utf-8').readlines()

  orig_input_lines_path = os.path.join(args.dir, ORIG_PREFIX + args.input)
  orig_input_lines = codecs.open(orig_input_lines_path, 'r', 'utf-8').readlines()

  assert len(input_lines) == len(orig_input_lines)
  assert len(input_lines) == len(results)

  for index, result in enumerate(results[:5]):
    unk_map = get_unk_map(orig_input_lines[index], input_lines[index])
    logging.info('UNK_Map:%s'%str(unk_map))
    replaced_sentences = [replace_line(sentence[1], unk_map) for sentence in result]

    logging.info('Line:%d Candidates:%d'%(index, len(result)))
    lm_scores = [(lm.compute_prob(sentence), sentence) for sentence in replaced_sentences]

    sorted_lm_scores = sorted(lm_scores, key=lambda score:score[0], reverse=True)
    lm_results.append(sorted_lm_scores)

  logging.info('Num LM Results:%d'%len(lm_results))

  lm_candidates_path = os.path.join(args.dir, 'lm.' + args.candidates)
  pkl.dump(lm_results, open(lm_candidates_path, 'w'))

if __name__ == '__main__':
    main()