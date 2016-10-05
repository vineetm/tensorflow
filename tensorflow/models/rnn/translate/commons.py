from collections import OrderedDict
import codecs, re, commands
from itertools import permutations

LEAVES='_LEAVES_'
SUBTREE='_SUBTREE_'
CONFIG_FILE = 'config.ckpt'

RAW_CANDIDATES = 'data.train.fr'
IGNORE_SW = ['s', 't', 'd', 'm', 'o', 'y']
SYMBOL_START = set(['Q', 'A', '_'])
UNK_SET = set(['Q', 'A'])

STOPW_FILE = 'stopw.txt'
DEV = 'data.dev'

ORIG_PREFIX = 'orig'
INPUT_SUFFIX = 'en'
OUTPUT_SUFFIX = 'fr'

DEV_INPUT = '%s.%s'%(DEV, INPUT_SUFFIX)
DEV_OUTPUT = '%s.%s'%(DEV, OUTPUT_SUFFIX)

ALL_REF = 'all_ref.txt'
ALL_HYP = 'all_hyp.txt'

from nltk.corpus import stopwords


def execute_bleu_command(ref_file, hyp_file):
  command = './multi-bleu.perl %s < %s' % (ref_file, hyp_file)
  status, output = commands.getstatusoutput(command)
  if status:
    print(output)

  match = re.search(r'BLEU\ \=\ (\d+\.\d+)', output)
  if match:
    return float(match.group(1))
  else:
    print 'BLEU not found! %s' % output
    return 0.0


def get_bleu_score(reference, hypothesis):
  with codecs.open('ref.txt', 'w', 'utf-8') as fw_ref:
    fw_ref.write(reference.strip() + '\n')

  with codecs.open('hyp.txt', 'w', 'utf-8') as fw_hyp:
    fw_hyp.write(hypothesis.strip() + '\n')
  return execute_bleu_command('ref.txt', 'hyp.txt')


def read_stopw(stopw_file):
  stopw_set = set()
  with codecs.open(stopw_file, 'r', 'utf-8') as fr:
    for line in fr:
      word = line.strip()
      stopw_set.add(word)

  stopw_set -= set(IGNORE_SW)
  return stopw_set


def get_stopw(stopw_file=STOPW_FILE):
  if stopw_file is None:
    return None

  nltk_stopw = stopwords.words('english')
  stopw = read_stopw(stopw_file)
  for word in nltk_stopw:
    stopw.add(word)
  return stopw


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
  tokens = line.split()
  return ' '.join(replace_tokens(tokens, unk_map))


def replace_tokens(orig_tokens, unk_map):
  return [unk_map[token] if token in unk_map else token for token in orig_tokens]


def merge_parts(parts):
  final_tokens = []
  for part in parts:
    final_tokens.extend(part.split())
    final_tokens.append('EOS')
  return ' '.join(final_tokens)


def get_rev_unk_map(unk_map):
  rev_unk_map = OrderedDict()
  for token in unk_map:
    rev_unk_map[unk_map[token]] = token
  return rev_unk_map


def get_phrase(tokens, start_index, max_len, phrases):
  for index in range(max_len):
    phrase_len = max_len - index
    if phrase_len == 1:
      return tokens[start_index], 1

    phrase = ' '.join(tokens[start_index: start_index + phrase_len])
    if phrase in phrases:
      return '_'.join(tokens[start_index: start_index + phrase_len]), phrase_len


def replace_part(part, phrases, max_len):
  final_tokens = []
  tokens = part.split()

  index = 0
  while index < len(tokens):
    phrase, phrase_len = get_phrase(tokens, index, max_len, phrases)
    index += phrase_len
    final_tokens.append(phrase)

  replaced_part = ' '.join(final_tokens)
  return replaced_part


def replace_phrases(parts, phrases):
  if len(phrases) == 0:
    return parts
  max_len = sorted([len(phrase.split()) for phrase in phrases], reverse=True)[0]
  return [replace_part(part, phrases, max_len) for part in parts]


def get_diff_unk_map(stopw, tokens, unk_map, symbol, numbers):
  replacement_num = 0
  num_numbers = 0
  for token in tokens:
    # Continue, if this is a stop word
    if token in stopw:
      continue

    if token in unk_map:
      continue

    if numbers:
      match = re.search(r'^(\d+)(\.\d+)?$', token)
      if match:
        unk_map[token] = '%s_NUM%d' % (symbol, num_numbers)
        num_numbers += 1
        continue

    unk_map[token] = '%s_%d' % (symbol, replacement_num)
    replacement_num += 1

  return unk_map


def get_diff_map(parts, stopw, numbers=True):
  unk_map = OrderedDict()
  unk_symbols = ['Q1', 'A1', 'Q2']

  for part_index, part in enumerate(parts):
    tokens = part.split()
    unk_map = get_diff_unk_map(stopw, tokens, unk_map, unk_symbols[part_index], numbers)
  return unk_map


def generate_replacements(tokens, indexes, unused_symbols):
  replacement_candidates = []
  target_indexes = range(len(unused_symbols))

  for mapping in permutations(target_indexes, len(indexes)):
    found = 0
    target_tokens = []
    for index, token in enumerate(tokens):
      if index in indexes:
        target_tokens.append(unused_symbols[mapping[found]])
        found += 1
      else:
        target_tokens.append(token)

    assert found == len(indexes)
    new_candidate = ' '.join(target_tokens)
    replacement_candidates.append(new_candidate)
  return replacement_candidates


def get_unresolved_symbols_and_indexes(tokens, available_symbols):
  indexes = set()
  unresolved_symbols = []
  used_symbols = set()

  for index, token in enumerate(tokens):
    if token in available_symbols:
      used_symbols.add(token)
      continue

    if token[0] not in SYMBOL_START:
      continue
    indexes.add(index)
    unresolved_symbols.append(token)
  return indexes, unresolved_symbols, used_symbols


def fill_missing_symbols(orig_candidates, unk_map):
  unchanged = 0
  skipped = 0
  available_symbols = set(unk_map.keys())
  unk_candidates = []
  for orig_candidate in orig_candidates:
    prob, candidate = orig_candidate
    tokens = candidate.split()
    unresolved_symbols_indexes, unresolved_symbols, used_symbols = get_unresolved_symbols_and_indexes(tokens, available_symbols)
    unused_symbols = available_symbols - used_symbols
    if len(unresolved_symbols) >= 1:
      if len(unresolved_symbols) > len(unused_symbols):
        skipped += 1
        continue
      replacement_candidates = generate_replacements(tokens, unresolved_symbols_indexes, list(unused_symbols))
      unk_candidates.extend([(prob, replacement_candidate) for replacement_candidate in replacement_candidates])
    else:
      unchanged += 1

    unk_candidates.append((prob, candidate))
  return unk_candidates

def get_unk_symbols(part):
  unk_symbols = [token for token in part.split() if token[0] in UNK_SET]
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

  for unk1 in set_unk_q1:
    for unk2 in candidate_unk_symbols:
      candidate = re.sub(unk1, unk2, q1)
      new_candidates.add(candidate)
  return new_candidates


def convert_phrase(line):
  tokens = line.split()
  final_tokens = []
  for token in tokens:
    if token[0] in SYMBOL_START:
      final_tokens.append(token)
      continue

    sub_tokens = token.split('_')
    if len(sub_tokens) > 1:
      final_tokens.extend(sub_tokens)
    else:
      final_tokens.append(token)

  return ' '.join(final_tokens)