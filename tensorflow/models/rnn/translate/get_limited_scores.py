import argparse, codecs, logging, cPickle as pkl, os, timeit, re
from translation_model import TranslationModel
from commons import LEAVES, SUBTREE, DEFAULT_CANDIDATES, DEFAULT_TREE

unk_set = set(['Q', 'A'])

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('train')
  parser.add_argument('model')
  parser.add_argument('src_vocab_size', type=int)
  parser.add_argument('target_vocab_size', type=int)
  parser.add_argument('model_size', type=int)
  parser.add_argument('input', help='Input Data')
  parser.add_argument('-num_layers', type=int, default=1)
  parser.add_argument('-k', type=int, default=5, help='# of Candidates to select')
  parser.add_argument('-candidates', default=DEFAULT_CANDIDATES)
  parser.add_argument('-tree', default=DEFAULT_TREE)
  parser.add_argument('-savefreq', default=10, type=int)
  parser.add_argument('-useq1', dest='useq1', default=False, action='store_true')
  parser.add_argument('-debug', dest='debug', default=False, action='store_true')
  args = parser.parse_args()
  if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  return args


class PendingWork:
  def __init__(self, prob, tree, prefix):
    self.prob = prob
    self.tree = tree
    self.prefix = prefix

  def __str__(self):
    return 'Str=%s(%f)'%(self.prefix, self.prob)


def compute_prob(candidates, input_line, prefix, tree):
  if len(tree[LEAVES]) == 1:
    return tm.compute_prob(input_line, candidates[tree[LEAVES][0]])
  return tm.compute_prob(input_line, prefix)


def prune_work(work, k):
  pending_work = sorted(work, key=lambda t:t.prob)[-k:]
  return pending_work


def get_prefix(candidates, tree, prefix):
  if len(tree[SUBTREE][prefix][LEAVES]) == 1:
    return candidates[tree[SUBTREE][prefix][LEAVES][0]]
  return prefix


def compute_scores(candidates, input_line, prefix_tree, k):
  # pending_work = [PendingWork(-1.0, prefix_tree, '')]
  final_scores = []
  num_comparisons = 0

  leaves = prefix_tree[SUBTREE].keys()
  pending_work = [PendingWork(tm.compute_prob(input_line, leaf) ,prefix_tree[SUBTREE][leaf], leaf)
                  for leaf in leaves]
  num_comparisons += len(pending_work)

  pending_work = prune_work(pending_work, k)

  while True:
    work = pending_work.pop()
    #logging.info('Work: %s Comparisons:%d Pending:%d'%(str(work), num_comparisons, len(pending_work)))

    prefixes = [get_prefix(candidates, work.tree, child) for child in work.tree[SUBTREE]]
    num_comparisons += len(prefixes)

    for prefix in prefixes:
      if prefix not in work.tree[SUBTREE]:
        final_scores.append((tm.compute_prob(input_line, prefix), prefix))
      else:
        pending_work.append(PendingWork(tm.compute_prob(input_line, prefix), work.tree[SUBTREE][prefix], prefix))

    pending_work = prune_work(pending_work, k)
    if len(pending_work) == 0:
      return final_scores, num_comparisons


def get_bestk_candidates(candidates, new_candidates, input_line, prefix_tree, k):
  scores, num_comparisons = compute_scores(candidates, input_line, prefix_tree, k)
  logging.debug('Old Scores: %d'%len(scores))

  scores = []
  num_comparisons = 0
  if new_candidates is not None:
    for new_candidate in new_candidates:
      scores.append((tm.compute_prob(input_line, new_candidate), new_candidate))

  logging.debug('Total Scores: %d'%len(scores))
  sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
  return sorted_scores, num_comparisons


def get_unk_symbols(part):
  unk_symbols = [token for token in part.split() if token[0] in unk_set]
  return unk_symbols


def generate_new_candidates(input_line):
  new_candidates = set()
  parts = input_line.split('EOS')
  q1 = parts[0]
  unk_q1 = get_unk_symbols(parts[0])
  set_unk_q1 = set(unk_q1)

  unk_a1 = set(get_unk_symbols(parts[1])) - set_unk_q1
  unk_q2 = set(get_unk_symbols(parts[2])) - set_unk_q1

  candidate_unk_symbols = unk_a1 | unk_q2
  # logging.debug('Candidate UNK Symbols: %s'%str(candidate_unk_symbols))

  for unk1 in set_unk_q1:
    for unk2 in candidate_unk_symbols:
      candidate = re.sub(unk1, unk2, q1)
      logging.debug(candidate)
      new_candidates.add(candidate)
  logging.debug('New Candidates: %d'%len(new_candidates))
  return new_candidates


def main():
  # Logging setup

  args = setup_args()
  logging.info(args)

  global tm
  model_path = os.path.join(args.train, args.model)
  data_path = os.path.join(args.train, 'data')
  tm = TranslationModel(model_path, data_path, args.src_vocab_size, args.target_vocab_size,
                        args.model_size, args.num_layers)

  input_lines = codecs.open(args.input, 'r', 'utf-8').readlines()

  prefix_tree_path = os.path.join(args.train, args.tree)
  prefix_tree = pkl.load(open(prefix_tree_path))

  candidates_path = os.path.join(args.train, args.candidates)
  candidates = pkl.load(open(candidates_path))
  logging.info('Candidates:%d Tree Leaves:%d'%(len(candidates), len(prefix_tree[LEAVES])))

  final_results = []

  st = timeit.default_timer()
  for line_num, input_line in enumerate(input_lines):
    result = []
    st_curr = timeit.default_timer()
    if args.useq1:
      new_candidates = generate_new_candidates(input_line)
    else:
      new_candidates = None

    sorted_scores, num_comparisons = get_bestk_candidates(candidates, new_candidates, input_line, prefix_tree, args.k)
    logging.info('Input:(%d) %s #Comparisons:%d'%(line_num, input_line.strip(), num_comparisons))

    for score in sorted_scores:
      p, c = score
      #logging.info('Str: %s Pr:%f' % (c, p))
      result.append((p, c))
    final_results.append(result)

    logging.info('Line:%d Time :%d sec' %(line_num, (timeit.default_timer() - st_curr)))

    # if line_num % args.savefreq == 0:
    #   pkl.dump(final_results, open(args.input + '.%d.results.pkl'%line_num, 'w'))

  end = timeit.default_timer()
  logging.info('Total Time :%d sec'%(end - st))

  pkl.dump(final_results, open(args.input + '.results.pkl', 'w'))

if __name__ == '__main__':
    main()
