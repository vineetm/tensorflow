import cPickle as pkl, logging, os, codecs, re
from translation_model import TranslationModel
from get_limited_scores import PendingWork, prune_work
from commons import SUBTREE, LEAVES, replace_line
from textblob.en.np_extractors import FastNPExtractor
from nltk.tokenize import word_tokenize as tokenizer
from nltk.corpus import stopwords
from collections import OrderedDict


IGNORE_SW = ['s', 't', 'd', 'm', 'o', 'y']

def get_config_file(config_file):
  config = pkl.load(open(config_file))
  return config

def read_stopw(stopw_file):
  stopw_set = set()
  with codecs.open(stopw_file, 'r', 'utf-8') as fr:
    for line in fr:
      word = line.strip()
      stopw_set.add(word)

  stopw_set -= set(IGNORE_SW)
  logging.info('SW: %s'%str(stopw_set))
  logging.info('%s file: %d stopwords'%(stopw_file, len(stopw_set)))
  return stopw_set


def get_stopw(stopw_file):
  if stopw_file is None:
    logging.info('No stopw found')
    return None

  nltk_stopw = stopwords.words('english')
  logging.info('NLTK Stopwords: %d' % len(nltk_stopw))
  stopw = read_stopw(stopw_file)
  for word in nltk_stopw:
    stopw.add(word)

  logging.info('Num Stopwords:%d' % len(stopw))
  return stopw


class CandidateScorer:
  def __init__(self, config_file, k=5):
    self.k = k
    config = get_config_file(config_file)

    self.config = config
    self.tm = TranslationModel(config.model_path, config.data_path, config.src_vocab_size, config.target_vocab_size,
                          config.model_size, config.num_layers)

    self.prefix_tree = pkl.load(open(config.prefix_tree))
    self.candidates = pkl.load(open(config.candidates))
    self.np_ex = FastNPExtractor()
    self.stopw = get_stopw(config.stopw_file)


  def get_prefix(self, tree, prefix):
    if len(tree[SUBTREE][prefix][LEAVES]) == 1:
      return self.candidates[tree[SUBTREE][prefix][LEAVES][0]]
    return prefix


  def compute_scores(self, input_line, prefix_tree, k):
    # pending_work = [PendingWork(-1.0, prefix_tree, '')]
    final_scores = []
    num_comparisons = 0

    leaves = prefix_tree[SUBTREE].keys()
    pending_work = [PendingWork(self.tm.compute_prob(input_line, leaf), prefix_tree[SUBTREE][leaf], leaf)
                    for leaf in leaves]
    num_comparisons += len(pending_work)

    pending_work = prune_work(pending_work, k)

    while True:
      work = pending_work.pop()
      # logging.info('Work: %s Comparisons:%d Pending:%d'%(str(work), num_comparisons, len(pending_work)))

      prefixes = [self.get_prefix(work.tree, child) for child in work.tree[SUBTREE]]
      num_comparisons += len(prefixes)

      for prefix in prefixes:
        if prefix not in work.tree[SUBTREE]:
          final_scores.append((self.tm.compute_prob(input_line, prefix), prefix))
        else:
          pending_work.append(PendingWork(self.tm.compute_prob(input_line, prefix), work.tree[SUBTREE][prefix], prefix))

      pending_work = prune_work(pending_work, k)
      if len(pending_work) == 0:
        return final_scores, num_comparisons


  def get_bestk_candidates(self, input_line, prefix_tree, k):
    scores, num_comparisons = self.compute_scores(input_line, prefix_tree, k)
    sorted_scores = sorted(scores, key=lambda t: t[0], reverse=True)
    return sorted_scores, num_comparisons


  def get_phrase(self, tokens, start_index, max_len, phrases):
    for index in range(max_len):
      phrase_len = max_len - index
      if phrase_len == 1:
        return tokens[start_index], 1

      phrase = ' '.join(tokens[start_index: start_index + phrase_len])
      if phrase in phrases:
        return '_'.join(tokens[start_index: start_index + phrase_len]), phrase_len


  def replace_part(self, part, phrases, max_len):
    final_tokens = []
    tokens = part.split()
    logging.debug('Orig Part: %s' % part)

    index = 0
    while index < len(tokens):
      phrase, phrase_len = self.get_phrase(tokens, index, max_len, phrases)
      if phrase_len > 1:
        logging.debug('Phrase: %s Len:%d' % (phrase, phrase_len))
      index += phrase_len
      final_tokens.append(phrase)

    replaced_part = ' '.join(final_tokens)
    logging.debug('Replaced Part: %s' % replaced_part)
    return replaced_part


  def replace_phrases(self, parts, phrases):
    if len(phrases) == 0:
      return parts
    max_len = sorted([len(phrase.split()) for phrase in phrases], reverse=True)[0]
    return [self.replace_part(part, phrases, max_len) for part in parts]

  def get_phrases(self, parts):
    phrases = set()
    for part in parts:
      part_phrases = self.np_ex.extract(part)
      for phrase in part_phrases:
        if len(phrase.split()) > 1:
          phrases.add(phrase)
    return phrases


  def get_diff_unk_map(self, tokens, unk_map, symbol, numbers):
    replacement_num = 0
    num_numbers = 0
    for token in tokens:
      # Continue, if this is a stop word
      if token in self.stopw:
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


  def get_diff_map(self, parts, numbers=True):
    unk_map = OrderedDict()
    unk_symbols = ['Q1', 'A1', 'Q2']

    for part_index, part in enumerate(parts):
      tokens = part.split()
      unk_map = self.get_diff_unk_map(tokens, unk_map, unk_symbols[part_index], numbers)
      logging.debug('Part:%s UNK_map:%s' % (unk_symbols[part_index], str(unk_map)))
    return unk_map


  def merge_parts(self, parts):
    final_tokens = []
    for part in parts:
      final_tokens.extend(part.split())
      final_tokens.append('EOS')
    return ' '.join(final_tokens)


  def rev_unk_map(self, unk_map):
    rev_unk_map = OrderedDict()
    for token in unk_map:
      rev_unk_map[unk_map[token]] = token

    return rev_unk_map


  def transform_input(self, input_sentence):
    input_sentence = input_sentence.lower()
    parts = input_sentence.split(';')
    parts = [' '.join(tokenizer(part)) for part in parts]
    phrases = self.get_phrases(parts)
    replaced_parts = self.replace_phrases(parts, phrases)
    unk_map = self.get_diff_map(replaced_parts)
    input_sequence_orig = self.merge_parts(replaced_parts)
    input_sequence = replace_line(input_sequence_orig, unk_map)
    rev_unk_map = self.rev_unk_map(unk_map)
    logging.info('UNK Map: %s'%str(unk_map))
    return rev_unk_map, unk_map, input_sequence_orig, input_sequence


  def fill_missing_symbols(self, orig_candidates, unk_map):
    unk_candidates = []
    symbol_start = set(['Q', 'A', '_'])

    for (prob, candidate) in orig_candidates:
      tokens = candidate.split()
      symbols = set([token for token in tokens if token[0] in symbol_start])
      unresolved_symbols = symbols - set(unk_map.keys())
      logging.debug('C:%s Unresolved:%s' % (candidate, str(unresolved_symbols)))

      if len(unresolved_symbols) == 0:
        unk_candidates.append((prob, candidate))
        continue

      if len(unresolved_symbols) == 1:
        unresolved_symbol = list(unresolved_symbols)[0]
        unused_symbols = set(unk_map.keys()) - symbols
        for ununsed_symbol in unused_symbols:
          new_candidate = re.sub(unresolved_symbol, ununsed_symbol, candidate)
          logging.debug('New Candidate: %s' % new_candidate)
          unk_candidates.append((prob, new_candidate))
    return unk_candidates


  def get_best_candidates(self, input_sentence, missing=False, k=20):
    rev_unk_map, unk_map, input_sequence_orig, input_sequence = self.transform_input(input_sentence)

    logging.info('Original: %s'%input_sequence_orig)
    logging.info('UNK: %s' % input_sequence)

    all_unk_candidates, num_comparisons = self.get_bestk_candidates(input_sentence, self.prefix_tree, self.k)
    unk_candidates = all_unk_candidates[:k]

    if missing:
      unk_candidates = self.fill_missing_symbols(unk_candidates, rev_unk_map)

    seq2seq_candidates = [(line[0], replace_line(line[1], rev_unk_map)) for line in unk_candidates]

    for index, (prob, candidate) in enumerate(seq2seq_candidates):
      logging.info('C: %s Pr:%f '%(candidate, prob))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  cs = CandidateScorer('config.pkl')

  cs.get_best_candidates('what is the capital of India ?; new delhi; and japan?')