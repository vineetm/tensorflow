import argparse, logging, codecs

from multiprocessing import Pool, Manager
from itertools import repeat
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
  parser.add_argument('k', type=int, help='# of Candidates to select')
  parser.add_argument('input', help='Input Data')
  parser.add_argument('-t', help='Num threads', default=8, type=int)
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
  # jobs_data = zip(candidates, repeat(input_line))
  # logging.info('Jobs Data Size:%d'%len(jobs_data))
  #
  # p = Pool(args.t)
  # results = p.map(compute_prob, jobs_data)
  # p.close()
  # p.join()

  results = [tm.compute_prob(input_line, candidate) for candidate in candidates]
  return results


def main():
  #Logging setup
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  args = setup_args()
  logging.info(args)

  #Get Candidates
  candidates = read_candidates(args)
  logging.info('File: %s Num Candidates:%d'%(args.candidates, len(candidates)))
  global tm
  tm = TranslationModel(args.model_path, args.data_path, args.vocab_size, args.model_size)


  for input_line in codecs.open(args.input, 'r', 'utf-8'):
    probs = compute_scores(args, candidates, input_line)
    logging.info('Input_Line: %s'%input_line)

    results = zip(probs, candidates)
    results = sorted(results, key=lambda tup: tup[0], reverse=True)

    for (prob, candidate) in results:
      logging.info('Candidate: %s prob %f'%(candidate, prob))


if __name__ == '__main__':
    main()

