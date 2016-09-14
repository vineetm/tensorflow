import argparse, logging, codecs

import cPickle as pkl
import timeit

from translation_model import TranslationModel

class Work(object):
  def __init__(self, tm, candidate, input_line, index):
    self.tm = tm
    self.candidate = candidate
    self.input_line = input_line
    self.index = index

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path')
  parser.add_argument('data_path')
  parser.add_argument('vocab_size', type=int)
  parser.add_argument('model_size', type=int)
  parser.add_argument('candidates')
  #parser.add_argument('k', type=int, help='# of Candidates to select')
  parser.add_argument('input', help='Input Data')
  args = parser.parse_args()
  return args



def read_candidates(args):
  candidates = set()
  for line in codecs.open(args.candidates, 'r', 'utf-8'):
    candidates.add(line.strip())
  return list(candidates)


def compute_prob((candidate, input_line)):
  return tm.compute_prob(input_line, candidate)


def compute_scores(args, candidates, input_line):
  results = []
  st = timeit.default_timer()
  logging.info('Debug: Input Line: %s'%input_line)
  for i, candidate in enumerate(candidates):
    curr_prob = tm.compute_prob(input_line, candidate)
    logging.info('C: %s Pr:%f'%(candidate, curr_prob))
    results.append(curr_prob)
  end = timeit.default_timer()
  logging.info('Total time: %ds'% (end - st))

  # results = [tm.compute_prob(input_line, candidate) for candidate in candidates]
  return results


def main():
  # Logging setup
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  args = setup_args()
  logging.info(args)

  global tm
  tm = TranslationModel(args.model_path, args.data_path, args.vocab_size, args.model_size)

  input_lines = codecs.open(args.input, 'r', 'utf-8').readlines()
  fw = codecs.open(args.input + '.results', 'w', 'utf-8')

  prefix_tree = pkl.load(open(args.candidates))

  for index, input_line in enumerate(input_lines):
    logging.info('Input Line: %s'%input_line)
    candidates = prefix_tree['']['what'].keys()
    logging.info('Num Candidates: %d'%len(candidates))
    probs = [tm.compute_prob(input_line, output_sentence) for output_sentence in candidates]
    results = zip(candidates, probs)
    results = sorted(results, key = lambda t : t[1], reverse=True)

    for (candidate, prob) in results:
      logging.info('Candidate: %s Prob: %f'%(candidate, prob))


def old_main():
  #Logging setup
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  args = setup_args()
  logging.info(args)

  #Get Candidates
  candidates = read_candidates(args)
  global tm
  tm = TranslationModel(args.model_path, args.data_path, args.vocab_size, args.model_size)

#  gold_lines = codecs.open(args.gold, 'r', 'utf-8').readlines()
  input_lines = codecs.open(args.input, 'r', 'utf-8').readlines()
 # assert (len(input_lines) == len(gold_lines))

  logging.info('Inputs: %d Candidates:%d'%(len(input_lines), len(candidates)))

  fw = codecs.open(args.input + '.results', 'w', 'utf-8')

  # ranks = OrderedDict()
  # for index in range(len(candidates)):
  #   ranks[index+1] = 0

  for index, input_line in enumerate(input_lines):
    probs = compute_scores(args, candidates, input_line)
    results = zip(probs, candidates)
    results = sorted(results, key=lambda tup: tup[0], reverse=True)

    fw.write('Input: %s\n'%input_line.strip())
    num = 1
    for (prob, candidate) in results:
  #    if candidate == gold_lines[index].strip():
  #      ranks[num] += 1
  #    num += 1
      fw.write('Candidate: %s prob %f\n'%(candidate, prob))
    fw.write('\n')
   # logging.info('Recall Ranks :%s'%str(ranks))

  #logging.info(ranks)

if __name__ == '__main__':
    old_main()

