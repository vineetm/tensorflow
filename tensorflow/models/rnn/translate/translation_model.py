import os, codecs, numpy as np
import tensorflow as tf
import cPickle as pkl
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary, sentence_to_token_ids
from commons import CONFIG_FILE, SUBTREE, LEAVES, RAW_CANDIDATES, \
  get_stopw, replace_line, replace_phrases, get_diff_map, merge_parts, get_rev_unk_map
from nltk.tokenize import word_tokenize as tokenizer
from textblob.en.np_extractors import FastNPExtractor

logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

class PendingWork:
  def __init__(self, prob, tree, prefix):
    self.prob = prob
    self.tree = tree
    self.prefix = prefix

  def __str__(self):
    return 'Str=%s(%f)'%(self.prefix, self.prob)


class TranslationModel(object):
  def __init__(self, models_dir):
    config_file_path = os.path.join(models_dir, CONFIG_FILE)
    logging.info('Loading Pre-trained seq2model:%s'%config_file_path)

    config = pkl.load(open(config_file_path))
    logging.info(config)

    self.session = tf.Session()
    self.model_path = config['train_dir']
    self.data_path = config['data_dir']
    self.src_vocab_size = config['src_vocab_size']
    self.target_vocab_size = config['target_vocab_size']
    self._buckets = config['_buckets']
    self.np_ex = FastNPExtractor()
    self.stopw = get_stopw()
    logging.info('Stopw: %d'%len(self.stopw))

    self.model = Seq2SeqModel(
      source_vocab_size = config['src_vocab_size'],
      target_vocab_size = config['target_vocab_size'],
      buckets=config['_buckets'],
      size = config['size'],
      num_layers = config['num_layers'],
      max_gradient_norm = config['max_gradient_norm'],
      batch_size=1,
      learning_rate=config['learning_rate'],
      learning_rate_decay_factor=config['learning_rate_decay_factor'],
      compute_prob=True,
      forward_only=True)

    ckpt = tf.train.get_checkpoint_state(self.model_path)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      logging.info('Model not found!')
      return None


    # Load vocabularies.
    en_vocab_path = os.path.join(self.data_path,
                                 "vocab%d.en" % self.src_vocab_size)
    fr_vocab_path = os.path.join(self.data_path,
                                 "vocab%d.fr" % self.target_vocab_size)

    self.en_vocab, _ = initialize_vocabulary(en_vocab_path)
    self.fr_vocab, self.rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

    #Read Candidates and build a prefix tree
    self.build_prefix_tree()
    logging.info('Prefix Tree Leaves:%d' % len(self.prefix_tree[LEAVES]))

  def get_node(self):
    node = {}
    node[LEAVES] = []
    node[SUBTREE] = {}
    return node

  def prune_tree(self, tree):
    if len(tree[SUBTREE]) == 0:
      return

    for child in tree[SUBTREE]:
      if len(tree[SUBTREE][child][LEAVES]) == 1:
        tree[SUBTREE][child][SUBTREE] = {}
      self.prune_tree(tree[SUBTREE][child])


  def build_prefix_tree(self):
    self.read_raw_candidates()
    root = self.get_node()

    for candidate_index, candidate in enumerate(self.candidates):
      root[LEAVES].append(candidate_index)
      tree = root
      tokens = candidate.split()

      for index in range(len(tokens)):
        tokens = candidate.split()
        prefix = ' '.join(tokens[:index + 1])
        if prefix not in tree[SUBTREE]:
          tree_node = self.get_node()
          tree[SUBTREE][prefix] = tree_node
        tree[SUBTREE][prefix][LEAVES].append(candidate_index)
        tree = tree[SUBTREE][prefix]

    self.prefix_tree = root
    self.prune_tree(self.prefix_tree)


  def read_raw_candidates(self):
    candidates_path = os.path.join(self.data_path, RAW_CANDIDATES)
    raw_candidates = codecs.open(candidates_path, 'r', 'utf-8').readlines()

    candidates = []
    #Repalce OOV words with _UNK
    for candidate in raw_candidates:
      tokens = [token if token in self.fr_vocab else '_UNK' for token in candidate.split()]
      candidates.append(' '.join(tokens))

    # Get Unique Candidates
    self.candidates = list(set(candidates))
    logging.info('Candidates: %d/%d' % (len(self.candidates), len(raw_candidates)))


  def set_output_tokens(self, output_token_ids, decoder_inputs):
    for index in range(len(output_token_ids)):
      if index + 1 >= len(decoder_inputs):
        logging.info('Skip assignment Decoder_Size:%d Outputs_Size:%d'%(len(decoder_inputs), len(output_token_ids)))
        return
      decoder_inputs[index + 1] = np.array([output_token_ids[index]], dtype=np.float32)


  def compute_fraction(self, logit, token_index):
    sum_all = np.sum(np.exp(logit))
    return np.exp(logit[token_index]) / sum_all


  #Compute probability of output_sentence given an input sentence
  def compute_prob(self, sentence, output_sentence):
    token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), self.en_vocab, normalize_digits=False)
    output_token_ids = sentence_to_token_ids(tf.compat.as_bytes(output_sentence), self.fr_vocab, normalize_digits=False)

    bucket_id = min([b for b in xrange(len(self._buckets))
                     if self._buckets[b][0] > len(token_ids)])

    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)

    self.set_output_tokens(output_token_ids, decoder_inputs)
    _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)

    if len(decoder_inputs) > len(output_token_ids):
      max_len = len(output_token_ids)
    else:
      max_len = len(decoder_inputs)

    prob = np.sum([self.compute_fraction(output_logits[index][0], output_token_ids[index])
                   for index in range(max_len)]) / max_len
    return prob

  def prune_work(self, work, k):
    pending_work = sorted(work, key=lambda t: t.prob)[-k:]
    return pending_work

  def get_prefix(self, tree, prefix):
    if len(tree[SUBTREE][prefix][LEAVES]) == 1:
      return self.candidates[tree[SUBTREE][prefix][LEAVES][0]]
    return prefix

  def compute_scores(self, input_line, work_buffer):
    final_scores = []
    num_comparisons = 0

    leaves = self.prefix_tree[SUBTREE].keys()
    pending_work = [PendingWork(self.compute_prob(input_line, leaf), self.prefix_tree[SUBTREE][leaf], leaf)
                    for leaf in leaves]
    num_comparisons += len(pending_work)
    pending_work = self.prune_work(pending_work, work_buffer)

    while True:
      work = pending_work.pop()
      # logging.info('Work: %s Comparisons:%d Pending: %d'%(str(work), num_comparisons, len(pending_work)))

      prefixes = [self.get_prefix(work.tree, child) for child in work.tree[SUBTREE]]
      num_comparisons += len(prefixes)

      for prefix in prefixes:
        if prefix not in work.tree[SUBTREE]:
          final_scores.append((self.compute_prob(input_line, prefix), prefix))
        else:
          pending_work.append(PendingWork(self.compute_prob(input_line, prefix), work.tree[SUBTREE][prefix], prefix))

      pending_work = self.prune_work(pending_work, work_buffer)
      if len(pending_work) == 0:
        return final_scores, num_comparisons


  def get_phrases(self, parts):
    phrases = set()
    for part in parts:
      part_phrases = self.np_ex.extract(part)
      for phrase in part_phrases:
        if len(phrase.split()) > 1:
          phrases.add(phrase)
    return phrases


  def transform_input(self, input_sentence):
    input_sentence = input_sentence.lower()
    parts = input_sentence.split(';')
    parts = [' '.join(tokenizer(part)) for part in parts]
    phrases = self.get_phrases(parts)
    replaced_parts = replace_phrases(parts, phrases)
    unk_map = get_diff_map(replaced_parts, self.stopw)
    input_sequence_orig = merge_parts(replaced_parts)
    input_sequence = replace_line(input_sequence_orig, unk_map)
    rev_unk_map = get_rev_unk_map(unk_map)
    return rev_unk_map, unk_map, input_sequence_orig, input_sequence


  def get_seq2seq_candidates(self, input_sentence, phrase=True, generate_codes=True, k=100, work_buffer=5):

    if generate_codes:
      rev_unk_map, unk_map, input_seq_orig, input_seq = self.transform_input(input_sentence)
      logging.info('UNK_Map: %s Reverse UNK_Map: %s'% (str(unk_map), str(rev_unk_map)))
      logging.info('Input_Seq_Orig: %s'%input_seq_orig)
      logging.info('Input_Seq: %s' %input_seq)
    else:
      rev_unk_map = None
      input_seq = input_sentence

    scores, num_comparisons = self.compute_scores(input_seq, work_buffer)
    scores = sorted(scores, key=lambda t:t[0], reverse=True)[:k]
    if generate_codes:
      scores = [(score[0], replace_line(score[1], rev_unk_map)) for score in scores]

    return scores