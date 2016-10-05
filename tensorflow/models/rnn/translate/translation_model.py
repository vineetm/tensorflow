import os, codecs, numpy as np
import tensorflow as tf
import cPickle as pkl
import numpy as np
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary, sentence_to_token_ids
from commons import get_stopw, replace_line, replace_phrases, get_diff_map, merge_parts, \
  get_rev_unk_map, fill_missing_symbols, generate_new_candidates, get_bleu_score, execute_bleu_command, get_unk_map, convert_phrase
from nltk.tokenize import word_tokenize as tokenizer
from textblob.en.np_extractors import FastNPExtractor

#Constants
from commons import CONFIG_FILE, SUBTREE, LEAVES, RAW_CANDIDATES, DEV_INPUT, DEV_OUTPUT, ORIG_PREFIX, ALL_HYP, ALL_REF


logging = tf.logging


class PendingWork:
  def __init__(self, prob, tree, prefix):
    self.prob = prob
    self.tree = tree
    self.prefix = prefix

  def __str__(self):
    return 'Str=%s(%f)'%(self.prefix, self.prob)


class TranslationModel(object):
  def __init__(self, models_dir, debug=False):
    config_file_path = os.path.join(models_dir, CONFIG_FILE)
    if debug:
      logging.set_verbosity(tf.logging.DEBUG)
    else:
      logging.set_verbosity(tf.logging.INFO)
    self.debug = debug

    logging.info('Loading Pre-trained seq2model:%s' % config_file_path)
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
      logging.error('Model not found!')
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
        logging.debug('Skip assignment Decoder_Size:%d Outputs_Size:%d'%(len(decoder_inputs), len(output_token_ids)))
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


  def get_seq2seq_candidates(self, input_sentence, orig_unk_map=None, k=100, generate_codes=True,
                             missing=False, use_q1=True, work_buffer=5):

    if generate_codes:
      rev_unk_map, unk_map, input_seq_orig, input_seq = self.transform_input(input_sentence)
      logging.info('UNK_Map: %s Reverse UNK_Map: %s'% (str(unk_map), str(rev_unk_map)))
      logging.info('Input_Seq_Orig: %s'%input_seq_orig)
      logging.info('Input_Seq: %s' %input_seq)
    else:
      rev_unk_map = orig_unk_map
      input_seq = input_sentence

    scores, num_comparisons = self.compute_scores(input_seq, work_buffer)
    if missing:
      scores = fill_missing_symbols(scores, rev_unk_map)

    if use_q1:
      new_candidates = generate_new_candidates(input_seq)
      logging.debug('New Q1 Candidates: %d'%len(new_candidates))
      scores.extend([(self.compute_prob(input_seq, new_candidate), new_candidate) for new_candidate in new_candidates])

    logging.debug('Num candidates: %d'%len(scores))
    scores = sorted(scores, key=lambda t:t[0], reverse=True)[:k]
    if rev_unk_map is not None:
      replaced_scores = [(score[0], replace_line(score[1], rev_unk_map)) for score in scores]

    if orig_unk_map is None:
      return replaced_scores
    else:
      return replaced_scores, scores


  def read_data(self, input_file, output_file, base_dir):
    if base_dir is None:
      base_dir = self.data_path

    input_file_path = os.path.join(base_dir, input_file)
    input_lines = codecs.open(input_file_path, 'r', 'utf-8').readlines()

    orig_input_file_path = os.path.join(base_dir, '%s.%s'%(ORIG_PREFIX, input_file))
    orig_input_lines = codecs.open(orig_input_file_path, 'r', 'utf-8').readlines()
    assert len(input_lines) == len(orig_input_lines)

    gold_file_path = os.path.join(base_dir, output_file)
    gold_lines = codecs.open(gold_file_path, 'r', 'utf-8').readlines()

    orig_gold_file_path = os.path.join(base_dir, '%s.%s' % (ORIG_PREFIX, output_file))
    orig_gold_lines = codecs.open(orig_gold_file_path, 'r', 'utf-8').readlines()

    assert len(gold_lines) == len(orig_gold_lines)
    assert len(gold_lines) == len(input_lines)

    return orig_input_lines, input_lines, orig_gold_lines, gold_lines


  def compute_bleu(self, k=100, num_lines=-1, input_file=DEV_INPUT, output_file=DEV_OUTPUT, base_dir=None):
    orig_input_lines, input_lines, orig_gold_lines, gold_lines = self.read_data(input_file, output_file, base_dir)

    num_inputs = len(input_lines)
    if num_lines > 0:
      num_inputs = num_lines

    logging.info('Num inputs: %d'%num_inputs)

    fw_all_ref = codecs.open(ALL_REF, 'w', 'utf-8')
    fw_all_hyp = codecs.open(ALL_HYP, 'w', 'utf-8')

    perfect_matches = 0
    for index in range(num_inputs):
      gold_line = convert_phrase(orig_gold_lines[index].strip())

      unk_map = get_unk_map(orig_input_lines[index], input_lines[index])
      scores, unk_scores = self.get_seq2seq_candidates(input_sentence=input_lines[index],
                                           orig_unk_map=unk_map, k=k, generate_codes=False)

      bleu_scores = [get_bleu_score(gold_line, convert_phrase(score[1])) for score in scores]
      best_score_index = np.argmax(bleu_scores)
      best_bleu_score = bleu_scores[best_score_index]
      hyp_line = convert_phrase(scores[best_score_index][1].strip())

      if best_bleu_score == 100.0:
        perfect_matches += 1

      logging.info('Line:%d Best_BLEU:%f(%d)'%(index, best_bleu_score, best_score_index))
      fw_all_ref.write(gold_line + '\n')
      fw_all_hyp.write(hyp_line + '\n')

      if self.debug:
        logging.debug('Input: %s'%orig_input_lines[index].strip())
        logging.debug('Input_UNK: %s' % input_lines[index].strip())
        logging.debug('')
        logging.debug('Gold: %s'%orig_gold_lines[index].strip())
        logging.debug('Gold_UNK: %s' % gold_lines[index].strip())
        logging.debug('')
        logging.debug('UNK_Map: %s'%str(unk_map))

        for score_index in range(len(scores)):
          logging.debug('C: %s B:%f Seq:%f'%(scores[score_index][1], bleu_scores[score_index], scores[score_index][0]))
          logging.debug('C_UNK: %s' %unk_scores[score_index][1])
          logging.debug('')

    fw_all_ref.close()
    fw_all_hyp.close()

    bleu_score = execute_bleu_command(ALL_REF, ALL_HYP)
    logging.info('Perfect Matches: %d/%d'%(perfect_matches, num_inputs))
    return bleu_score