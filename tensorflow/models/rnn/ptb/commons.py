from collections import OrderedDict

LEAVES='_LEAVES_'
SUBTREE='_SUBTREE_'
ORIG_PREFIX = 'orig.'
RESULTS_SUFFIX = 'results.pkl'
FINAL_RESULTS_SUFFIX = 'final.results.pkl'
SOURCE = 'en'
TARGET =' fr'

def compute_unk_map(src_tokens, stopw):
  unk_map = OrderedDict()

  for token in src_tokens:
    if token in stopw:
      continue
    elif token not in unk_map:
      unk_map[token] = 'UNK%d'%(len(unk_map)+1)

  return unk_map


def get_unk_map(orig_line, replaced_line):
  unk_map = OrderedDict()

  orig_tokens = orig_line.split()
  replaced_tokens = replaced_line.split()
  assert len(orig_tokens) == len(replaced_tokens)

  for orig_token, replaced_token in zip(orig_tokens, replaced_tokens):
    if orig_token == replaced_token:
      continue
    unk_map[replaced_token] = orig_token
  return unk_map


def replace_line(line, unk_map):
  prob, unk_line = line
  tokens = unk_line.split()
  return (prob, ' '.join(replace_tokens(tokens, unk_map)))


def replace_tokens(orig_tokens, unk_map):
  return [unk_map[token] if token in unk_map else token for token in orig_tokens]