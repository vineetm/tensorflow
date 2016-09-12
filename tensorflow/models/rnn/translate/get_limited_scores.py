import argparse, codecs, logging, cPickle as pkl
from translation_model import TranslationModel
from commons import LEAVES, SUBTREE


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path')
  parser.add_argument('data_path')
  parser.add_argument('vocab_size', type=int)
  parser.add_argument('model_size', type=int)
  parser.add_argument('input', help='Input Data')
  parser.add_argument('-k', type=int, default=5, help='# of Candidates to select')
  parser.add_argument('-candidates', default='candidates.pkl')
  parser.add_argument('-tree', default='cand_tree.pkl')
  args = parser.parse_args()
  return args


class PendingWork:
  def __init__(self, prob, tree, prefix):
    self.prob = prob
    self.tree = tree
    self.prefix = prefix

  def __str__(self):
    return 'Str=%s(%f)'%(self.prefix, self.prob)


def compute_prob(input_line, prefix, tree):
  if len(tree[LEAVES]) == 1:
    return tm.compute_prob(input_line, candidates[tree[LEAVES][0]])
  return tm.compute_prob(input_line, prefix)

def prune_work(work, k=5):
  pending_work = sorted(work, key=lambda t:t.prob)[-k:]
  return pending_work

def get_prefix(tree, prefix):
  if len(tree[SUBTREE][prefix][LEAVES]) == 1:
    return candidates[tree[SUBTREE][prefix][LEAVES][0]]
  return prefix

def compute_scores(input_line, prefix_tree, k):
  pending_work = [PendingWork(-1.0, prefix_tree, '')]
  final_scores = []
  num_comparisons = 0

  while True:
    work = pending_work.pop()
    logging.info('Work: %s Comparisons:%d Pending:%d'%(str(work), num_comparisons, len(pending_work)))

    prefixes = [get_prefix(work.tree, child) for child in work.tree[SUBTREE]]
    num_comparisons += len(prefixes)

    for prefix in prefixes:
      if prefix not in work.tree[SUBTREE]:
        final_scores.append((tm.compute_prob(input_line, prefix), prefix))
      else:
        pending_work.append(PendingWork(tm.compute_prob(input_line, prefix), work.tree[SUBTREE][prefix], prefix))

    pending_work = prune_work(pending_work, k)
    if len(pending_work) == 0:
      return final_scores, num_comparisons


def get_bestk_candidates(input_line, prefix_tree, k):
  scores, num_comparisons = compute_scores(input_line, prefix_tree, k)
  sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
  return sorted_scores[:k]


def main():
  # Logging setup
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  args = setup_args()
  logging.info(args)

  global tm
  tm = TranslationModel(args.model_path, args.data_path, args.vocab_size, args.model_size)

  input_lines = codecs.open(args.input, 'r', 'utf-8').readlines()
  prefix_tree = pkl.load(open(args.tree))

  global candidates
  candidates = pkl.load(open(args.candidates))

  logging.info('Candidates:%d Tree Leaves:%d'%(len(candidates), len(prefix_tree[LEAVES])))

  fw = codecs.open(args.input + '.k%d.results'%args.k, 'w', 'utf-8')

  for line_num, input_line in enumerate(input_lines):
    sorted_scores = get_bestk_candidates(input_line, prefix_tree, args.k)
    for score in sorted_scores:
      p, c = score
      logging.info('Str: %s Pr:%f' % (c, p))
      fw.write('C:%s Pr:%f\n' % (c, p))


if __name__ == '__main__':
    main()
