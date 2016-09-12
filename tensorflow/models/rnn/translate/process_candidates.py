import logging, argparse, codecs
import cPickle as pkl, re

from collections import OrderedDict
from data_utils import initialize_vocabulary
import numpy as np

LEAVES='_LEAVES_'
SUBTREE='_SUBTREE_'

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('candidates', help='Raw Candidates file')
  parser.add_argument('vocab', help='Target Vocab file')
  parser.add_argument('-len', type=int, default=4)
  args = parser.parse_args()
  return args


def get_raw_candidates(args, vocab):
  candidates = codecs.open(args.candidates, 'r', 'utf-8').readlines()

  final_candidates = []
  for candidate in candidates:
    tokens = [token if token in vocab else '_UNK' for token in candidate.split()]
    final_candidates.append(' '.join(tokens))

  return final_candidates


def get_vocab(args):
  vocab, rev_vocab = initialize_vocabulary(args.vocab)
  logging.info('Vocab: %d'% len(vocab))
  return vocab, rev_vocab


def old_build_prefix_tree(candidates):
  prefix_tree = {}

  for cand_index, candidate in enumerate(candidates):
    tree = prefix_tree
    tokens = candidate.split()
    for index, token in enumerate(tokens):
      key = ' '.join(tokens[:index])
      if key not in tree:
        tree[key] = {}
      tree = tree[key]
  return prefix_tree


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
  pkl.dump(candidates, open('candidates.pkl', 'wb'))
  logging.info('Unique Candidates: %d' % len(candidates))

  prefix_tree = build_prefix_tree(candidates)
  prune_tree((prefix_tree))
  pkl.dump(prefix_tree, open('cand_tree.pkl', 'wb'))


if __name__ == '__main__':
    main()