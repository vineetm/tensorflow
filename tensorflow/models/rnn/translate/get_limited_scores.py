import argparse, codecs, logging, cPickle as pkl, os
from translation_model import TranslationModel
from commons import LEAVES, SUBTREE, DEFAULT_CANDIDATES, DEFAULT_TREE


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('train')
  parser.add_argument('model')
  parser.add_argument('src_vocab_size', type=int)
  parser.add_argument('target_vocab_size', type=int)
  parser.add_argument('model_size', type=int)

  parser.add_argument('input', help='Input Data')
  parser.add_argument('-k', type=int, default=5, help='# of Candidates to select')
  parser.add_argument('-candidates', default=DEFAULT_CANDIDATES)
  parser.add_argument('-tree', default=DEFAULT_TREE)
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

def prune_work(work, k):
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
  return sorted_scores, num_comparisons


def main():
  # Logging setup
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  args = setup_args()
  logging.info(args)

  global tm
  model_path = os.path.join(args.train, 'models/%s'%args.model)
  data_path = os.path.join(args.train, 'data')
  tm = TranslationModel(model_path, data_path, args.src_vocab_size, args.target_vocab_size, args.model_size)

  input_lines = codecs.open(args.input, 'r', 'utf-8').readlines()

  prefix_tree_path = os.path.join(args.train, args.tree)
  prefix_tree = pkl.load(open(prefix_tree_path))

  global candidates
  candidates_path = os.path.join(args.train, args.candidates)
  candidates = pkl.load(open(candidates_path))
  logging.info('Candidates:%d Tree Leaves:%d'%(len(candidates), len(prefix_tree[LEAVES])))

  final_results = []
  for line_num, input_line in enumerate(input_lines):
    result = []
    sorted_scores, num_comparisons = get_bestk_candidates(input_line, prefix_tree, args.k)
    logging.info('Input:(%d) %s #Comparisons:%d'%(line_num, input_line.strip(), num_comparisons))

    for score in sorted_scores:
      p, c = score
      logging.info('Str: %s Pr:%f' % (c, p))
      result.append((p, c))
    final_results.append(result)

    if line_num != (len(input_line) - 1):
      pkl.dump(final_results, open(args.input + '.%d.results.pkl'%line_num, 'w'))


  pkl.dump(final_results, open(args.input + '.results.pkl', 'w'))

if __name__ == '__main__':
    main()
