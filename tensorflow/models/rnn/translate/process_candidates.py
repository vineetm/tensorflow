import logging, argparse, codecs
import cPickle as pkl, os, timeit

from data_utils import initialize_vocabulary
from commons import LEAVES, SUBTREE, DEFAULT_CANDIDATES, DEFAULT_TREE


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('train')
  parser.add_argument('vocab', type=int, help='Target Vocab Size')
  parser.add_argument('-candidates', default=DEFAULT_CANDIDATES)
  parser.add_argument('-tree', default=DEFAULT_TREE)
  args = parser.parse_args()
  return args


def get_raw_candidates(args, vocab):
  candidates_path = os.path.join(args.train, 'data/data.train.fr')
  candidates = codecs.open(candidates_path, 'r', 'utf-8').readlines()
  logging.info('Read %d raw candidates:%s'%(len(candidates), candidates_path))

  final_candidates = []
  for candidate in candidates:
    tokens = [token if token in vocab else '_UNK' for token in candidate.split()]
    final_candidates.append(' '.join(tokens))

  return final_candidates


def get_vocab(args):
  vocab_path = os.path.join(args.train, 'data/vocab%d.fr'%args.vocab)
  logging.info('Reading from Vocab: %s'%vocab_path)
  vocab, rev_vocab = initialize_vocabulary(vocab_path)
  logging.info('Vocab: %d'% len(vocab))
  return vocab, rev_vocab


def get_node():
  node = {}
  node[LEAVES] = []
  node[SUBTREE] = {}
  return node


def build_prefix_tree(candidates):
  root = get_node()

  for candidate_index, candidate in enumerate(candidates):
    root[LEAVES].append(candidate_index)
    tree = root
    tokens = candidate.split()

    for index in range(len(tokens)):
      tokens = candidate.split()
      prefix = ' '.join(tokens[:index+1])
      if prefix not in tree[SUBTREE]:
        tree_node = get_node()
        tree[SUBTREE][prefix] = tree_node
      tree[SUBTREE][prefix][LEAVES].append(candidate_index)
      tree = tree[SUBTREE][prefix]

  return root


def prune_tree(tree):
  if len(tree[SUBTREE]) == 0:
    return

  for child in tree[SUBTREE]:
    if len(tree[SUBTREE][child][LEAVES]) == 1:
      tree[SUBTREE][child][SUBTREE] = {}
    prune_tree(tree[SUBTREE][child])


def main():
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  args = setup_args()
  logging.info(args)

  vocab, rev_vocab = get_vocab(args)

  candidates = get_raw_candidates(args, vocab)
  logging.info('Raw Candidates: %d'%len(candidates))

  #Get Unique Candidates
  candidates = list(set(candidates))

  candidates_path = os.path.join(args.train, args.candidates)
  with open(candidates_path, 'wb') as fw:
    pkl.dump(candidates, fw)
  logging.info('Wrote %d candidates to %s'%(len(candidates), candidates_path))

  prefix_tree = build_prefix_tree(candidates)
  prefix_tree_path = os.path.join(args.train, args.tree)
  prune_tree((prefix_tree))

  with open(prefix_tree_path, 'wb') as fw:
    pkl.dump(prefix_tree, fw)
  logging.info('Prefix Tree:%s #Leaves:%d'%(prefix_tree_path, len(prefix_tree[LEAVES])))


if __name__ == '__main__':
    main()