from commons import CONFIG_FILE, STOPW_FILE, EVAL_DATA
import os, logging, codecs, time, nltk
import tensorflow as tf
import cPickle as pkl
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary, sentence_to_token_ids, EOS_ID
from collections import Counter

from nltk.tokenize import word_tokenize as tokenizer
from commons import execute_bleu_command, get_num_lines, load_pkl, save_pkl, compute_bleu_multiple_references, TRANSLATIONS_FILE
import numpy as np, argparse
from progress.bar import Bar
from symbol_assigner import SymbolAssigner
from translation_candidate import Candidate


ENTITIES = ['PER', 'GEO', 'ORG', 'BLD']

RECALL_FILE = 'recall.pkl'
PRECISION_FILE = 'precision.pkl'

logging = tf.logging
logging.set_verbosity(logging.INFO)

DEF_MODEL_DIR = 'trained-models/seq2seq'

class SequenceGenerator(object):
  def __init__(self, model_dir=DEF_MODEL_DIR, max_unk_symbols=8, entity=False, phrase=False):

    config_file_path = os.path.join(model_dir, CONFIG_FILE)
    logging.set_verbosity(logging.INFO)

    logging.info('Loading Pre-trained seq2model:%s' % config_file_path)
    config = pkl.load(open(config_file_path))
    logging.info(config)

    #Create session
    self.session = tf.Session()

    #Setup parameters using saved config
    self.model_path = config['train_dir']
    self.data_path = config['data_dir']
    self.src_vocab_size = config['src_vocab_size']
    self.target_vocab_size = config['target_vocab_size']
    self._buckets = config['_buckets']
    self.max_unk_symbols = max_unk_symbols

    compute_prob = True

    #Create model
    self.model = Seq2SeqModel(
      source_vocab_size=config['src_vocab_size'],
      target_vocab_size=config['target_vocab_size'],
      buckets=config['_buckets'],
      size=config['size'],
      num_layers=config['num_layers'],
      max_gradient_norm=config['max_gradient_norm'],
      batch_size=1,
      learning_rate=config['learning_rate'],
      learning_rate_decay_factor=config['learning_rate_decay_factor'],
      forward_only=True,
      compute_prob=compute_prob,
      num_samples=-1)

    #Restore Model from checkpoint file
    ckpt = tf.train.get_checkpoint_state(self.model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      logging.error('Model not found!')
      return None

    # Load vocabularies
    en_vocab_path = os.path.join(self.data_path,
                                   "vocab%d.en" % self.src_vocab_size)
    fr_vocab_path = os.path.join(self.data_path,
                                   "vocab%d.fr" % self.target_vocab_size)

    self.en_vocab, self.rev_en_vocab = initialize_vocabulary(en_vocab_path)
    self.fr_vocab, self.rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

    stopw_file = os.path.join(self.data_path,  STOPW_FILE)
    if entity or phrase:
      self.sa = SymbolAssigner(stopw_file, entity_mapping_file='entity-map.pkl', valid_entity_list=ENTITIES)
    else:
      self.sa = SymbolAssigner(stopw_file, valid_entity_list=None, entity_mapping_file=None)

  def generate_output_sequence(self, sentence, unk_tx=True):
    sentence = sentence.lower()
    logging.debug('Src: %s' % ' '.join(tokenizer(sentence)))

    if unk_tx:
      unk_sentence, unk_map, rev_unk_map = self.convert_to_unk_sequence(sentence)
      logging.debug('UNK: %s' % unk_sentence)
    else:
      unk_sentence = sentence

    token_ids = sentence_to_token_ids(tf.compat.as_bytes(unk_sentence), self.en_vocab, normalize_digits=False)
    logging.debug('Tkn: %s'%str(token_ids))

    if len(token_ids) >= self._buckets[-1][0]:
      token_ids = token_ids[:self._buckets[-1][0]-1]
      bucket_id = len(self._buckets) - 1
    else :
      bucket_id = min([b for b in xrange(len(self._buckets))
                     if self._buckets[b][0] > len(token_ids)])

    # logging.info('Bucket_ID: %d len_tokens:%d'%(bucket_id, len(token_ids)))
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)


    # Get output logits for the sentence.
    _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)


    # This is a greedy decoder - outputs are just argmaxes of output_logits.

    output_tokens = []
    for output_token_index in xrange(len(output_logits)):
      output_tokens.append(np.argmax(output_logits[output_token_index][0]))
      if EOS_ID in output_tokens:
        output_tokens = output_tokens[:output_tokens.index(EOS_ID)]

    unk_output_sentence = " ".join([tf.compat.as_str(self.rev_fr_vocab[output]) for output in output_tokens])
    logging.debug(unk_output_sentence)

    if unk_tx:
      output_sentence = self.replace_tokens(unk_output_sentence.split(), rev_unk_map)
    else:
      output_sentence = unk_output_sentence
    return output_sentence


  def generate_outputs(self, suffix):
    eval_sentences = pkl.load(open(EVAL_DATA))
    logging.info('Num Eval Sentences: %d'%len(eval_sentences))

    output_sentences = [(input_sentence, self.generate_output_sequence(input_sentence)) for input_sentence in eval_sentences]
    pkl.dump(output_sentences, open('%s.%s.pkl'%('results', suffix), 'w'))


  def convert_to_prob(self, logit):
    exp_logit = np.exp(logit)
    return exp_logit / np.sum(exp_logit)


  def set_output_tokens(self, output_token_ids, decoder_inputs):
    for index in range(len(output_token_ids)):
      decoder_inputs[index + 1] = np.array([output_token_ids[index]], dtype=np.float32)


  def cleanup_final_translations(self, final_translations, source_sentence, phrase):
    sentences_set = set()
    sentences_set.add(source_sentence)

    cleaned_translations = []
    for sentence, prob in final_translations:
      sentence = ' '.join(sentence.split()[:-1])
      if sentence not in sentences_set:

        cleaned_translations.append((sentence, prob))
        sentences_set.add(sentence)

    if phrase:
      cleaned_translations = [(self.sa.replace_phrase(sentence), prob) for sentence, prob in cleaned_translations]
    return cleaned_translations


  def get_new_work(self, encoder_inputs, decoder_inputs, target_weights, bucket_id, fixed_tokens, curr_prob, beam_size):
    self.set_output_tokens(fixed_tokens, decoder_inputs)
    _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

    probs = self.convert_to_prob(output_logits[len(fixed_tokens)][0])
    sorted_probs = sorted([(index, probs[index]) for index in xrange(len(probs))], key=lambda x: x[1],
                            reverse=True)

    new_work = []

    for part in sorted_probs:
      new_tokens = []
      new_tokens.extend(fixed_tokens)
      new_tokens.append(part[0])
      new_tokens = [self.rev_fr_vocab[token] for token in new_tokens]
      new_work.append((' '.join(new_tokens), part[1] * curr_prob))
    return new_work[:beam_size]


  def get_bucket_id(self, token_ids):
    bucket_ids = [b for b in xrange(len(self._buckets))
                  if self._buckets[b][0] > len(token_ids)]
    if len(bucket_ids) == 0:
      bucket_id = len(self._buckets) - 1
    else:
      bucket_id = min(bucket_ids)
    return bucket_id


  def generate_topk_sequences(self, sentence, unk_tx=True, tokenize=True, beam_size=16, phrase=False, entity=False):
    if tokenize:
      sentence = sentence.lower()
      sentence = ' '.join(tokenizer(sentence))

    if phrase:
      sentence = self.sa.convert_to_phrases(sentence)

    unk_map = None
    if entity:
      unk_map, unk_list = self.sa.build_entity_unk_map(sentence)
      unk_sentence = self.sa.assign_unk_symbols(sentence, unk_map)
    elif unk_tx:
      unk_map = self.sa.build_unk_map(sentence)
      unk_sentence = self.sa.assign_unk_symbols(sentence, unk_map)
    else:
      unk_sentence = sentence

    if unk_map is not None and len(unk_map) > self.max_unk_symbols:
      # logging.warn('Skipping Unk_symbols:%d Max:%d'%(len(unk_map), self.max_unk_symbols))
      return []

    token_ids = sentence_to_token_ids(tf.compat.as_bytes(unk_sentence), self.en_vocab, normalize_digits=False)
    bucket_id = self.get_bucket_id(token_ids)

    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)

    #Initialize with empty fixed tokens
    rem_work = self.get_new_work(encoder_inputs, decoder_inputs, target_weights, bucket_id, [], 1.0, beam_size)

    final_translations = []
    while True:
      if len(rem_work) == 0:
        if unk_tx or entity:
          final_translations = [(self.sa.recover_orginal_sentence(final_translation[0], unk_map), final_translation[1]) for final_translation in final_translations]
        final_translations = sorted(final_translations, key=lambda x: x[1], reverse=True)[:beam_size]
        return self.cleanup_final_translations(final_translations, sentence, phrase)

      #Remove top of work
      curr_work = rem_work[0]
      del rem_work[0]

      curr_sentence, curr_prob = curr_work
      curr_tokens = [self.fr_vocab[token] for token in curr_sentence.split()]

      if curr_tokens[-1] == self.rev_fr_vocab[EOS_ID]:
        final_translations.append((' '.join(curr_tokens[:-1]), curr_prob))
        continue

      #Check if we received an EOS or went past length
      if len(curr_tokens) == self._buckets[bucket_id][1] or curr_tokens[-1] == EOS_ID:
        final_translations.append(curr_work)
        continue

      new_work = self.get_new_work(encoder_inputs, decoder_inputs, target_weights, bucket_id, curr_tokens, curr_prob, beam_size)
      rem_work.extend(new_work)
      rem_work = sorted(rem_work, key=lambda x:x[1], reverse=True)
      rem_work = rem_work[:beam_size]


  '''
  Get BLEU score for each hypothesis
  '''
  def get_bleu_scores(self, list_hypothesis, references, suffix=None):

    bleu_scores = []
    #Write all the references
    ref_string = ''
    for index, reference in enumerate(references):
      ref_file = os.path.join(self.model_path, 'ref%d.txt'%index)
      if suffix is not None:
        ref_file = '%s.%s'%(ref_file, suffix)

      with codecs.open(ref_file, 'w', 'utf-8') as f:
        f.write(reference.strip() + '\n')
      ref_string += ' %s'%ref_file

    hyp_file = os.path.join(self.model_path, 'hyp.txt')
    if suffix is not None:
      hyp_file = '%s.%s' % (hyp_file, suffix)

    for index, hypothesis in enumerate(list_hypothesis):
      with codecs.open(hyp_file, 'w', 'utf-8') as f:
        f.write(hypothesis.strip() + '\n')

      bleu = execute_bleu_command(ref_string, hyp_file)
      bleu_scores.append(bleu)
    return bleu_scores


  def write_bleu_data(self, max_refs, input_sentences, selected_hypothesis, all_references, report_data, suffix=None):
    #Write all references
    ref_file_names = [os.path.join(self.model_path, 'all_ref%d.txt' % index) for index in range(max_refs)]
    if suffix is not None:
      ref_file_names = ['%s.%s'%(ref_file_name, suffix) for ref_file_name in ref_file_names]
    ref_fw = [codecs.open(file_name, 'w', 'utf-8') for file_name in ref_file_names]

    for references in all_references:
      for index, reference in enumerate(references):
        ref_fw[index].write(reference.strip() + '\n')
    [fw.close() for fw in ref_fw]

    #Write hypothesis
    hyp_file = os.path.join(self.model_path, 'all_hyp.txt')
    if suffix is not None:
      hyp_file = '%s.%s'%(hyp_file, suffix)
    hyp_fw = codecs.open(hyp_file, 'w', 'utf-8')
    for hyp in selected_hypothesis:
      hyp_fw.write(hyp)
    hyp_fw.close()

    all_inputs_fname = os.path.join(self.model_path, 'all_inputs.txt')
    if suffix is not None:
      all_inputs_fname = '%s.%s'%(all_inputs_fname, suffix)
    inp_fw = codecs.open(all_inputs_fname, 'w', 'utf-8')

    for input_sentence in input_sentences:
      inp_fw.write(input_sentence + '\n')
    inp_fw.close()

    report_fname = os.path.join(self.model_path, 'report.txt')
    if suffix is not None:
      report_fname = '%s.%s'%(report_fname, suffix)

    fw_report = codecs.open(report_fname, 'w', 'utf-8')
    for write_data in report_data:
      fw_report.write('\t'.join(write_data) + '\n')
    fw_report.close()

    ref_str = ' '.join(ref_file_names)
    final_bleu = execute_bleu_command(ref_str, hyp_file)
    return final_bleu


  def read_lines(self, file_name):
    fr = codecs.open(file_name, 'r', 'utf-8')
    lines = fr.readlines()
    fr.close()
    return lines



  def compute_average_bleu_scores(self, args):
    def get_header_data():
      header_data = []
      header_data.append('Input Sentence')
      header_data.append('Avg Precision')
      header_data.append('Avg Recall')

      header_data.append('Best Precision')
      header_data.append('Best Recall')

      for ref_num in range(args.max_refs):
        header_data.append('Ref%d' % ref_num)
        header_data.append('Bleu')

      for cand_num in range(args.beam_size):
        header_data.append('C%d' % cand_num)
        header_data.append('Bleu')
      return header_data

    all_avg_precision = []
    all_avg_recall = []

    report_fw = codecs.open(os.path.join(args.model_dir, 'report.txt'), 'w', 'utf-8')
    report_fw.write('\t'.join(get_header_data()) + '\n')
    for part_num in range(1, args.workers+1):
      # eval_file_part = 'eval.data.p1'
      eval_file_part = '%s.p%d'%(args.eval_file, part_num)
      eval_lines = codecs.open(eval_file_part, 'r', 'utf-8').readlines()

      translations = load_pkl(os.path.join(args.model_dir, '%s.%s' % (eval_file_part, TRANSLATIONS_FILE)))
      recall_scores = load_pkl(os.path.join(args.model_dir, '%s.%s' % (eval_file_part, RECALL_FILE)))
      precision_scores = load_pkl(os.path.join(args.model_dir, '%s.%s' % (eval_file_part, PRECISION_FILE)))

      for part_index in translations:
        write_data = []
        #Write input question
        write_data.append(eval_lines[part_index].split('\t')[0])
        precision_scores_only = [score[1] for score in precision_scores[part_index]]

        if len(precision_scores_only) == 0:
          avg_precision = 0.0
          max_precision = 0.0
          argmax_precision = 0
        else:
          avg_precision = np.average(precision_scores_only)
          max_precision = np.max(precision_scores_only)
          argmax_precision = np.argmax(precision_scores_only)

        #Write Average Precision
        write_data.append('%.2f'%avg_precision)
        all_avg_precision.append(avg_precision)

        # Write Average Recall
        recall_scores_only = [score[1] for score in recall_scores[part_index]]
        avg_recall = np.average(recall_scores_only)
        write_data.append('%.2f'%avg_recall)
        all_avg_recall.append(avg_recall)

        #Write best precision and recall
        write_data.append('%.2f [%d]' % (max_precision, argmax_precision))
        write_data.append('%.2f [%d]'%(np.max(recall_scores_only), np.argmax(recall_scores_only)))

        for reference, bleu in recall_scores[part_index]:
          write_data.append(reference)
          write_data.append('%.2f'%bleu)

        for candidate, bleu in precision_scores[part_index]:
          write_data.append(candidate)
          write_data.append('%.2f' % bleu)

        # if len(all_avg_precision) % 100 == 0:
        logging.info('Index: %d Avg_Pr:%.2f Avg_Re:%.2f'%(len(all_avg_precision), np.average(all_avg_precision),
                                                          np.average(all_avg_recall)))

        report_fw.write('\t'.join(write_data) + '\n')
      logging.info('Final Avg_Pr:%.2f Avg_Re:%.2f' %(np.average(all_avg_precision), np.average(all_avg_recall)))


  def compute_bleu_scores(self, args):
    translations_fname = os.path.join(args.model_dir, '%s.%s' % (args.eval_file, TRANSLATIONS_FILE))
    translations = load_pkl(translations_fname)
    if len(translations) > 0:
      return

    recall_fname = os.path.join(args.model_dir, '%s.%s' % (args.eval_file, RECALL_FILE))
    precision_fname = os.path.join(args.model_dir, '%s.%s' % (args.eval_file, PRECISION_FILE))

    eval_index = 0
    saved_precision_scores = {}
    saved_recall_scores = {}

    for eval_line in codecs.open(args.eval_file, 'r', 'utf-8'):
      parts = eval_line.split('\t')
      input_sentence = parts[0].strip()
      references = [part.strip() for part in parts[1:]][:args.max_refs]

      if eval_index not in translations:
        all_hypothesis = self.generate_topk_sequences(input_sentence, unk_tx=args.unk_tx,
                                                      beam_size=args.beam_size, tokenize=False)

        all_candidates = [Candidate(hyp[0], seq2seq_score=hyp[1], model=args.label) for hyp in all_hypothesis]

        candidate_sentences = [candidate.text for candidate in all_candidates]
        eval_prefix = os.path.join(args.model_dir, args.eval_file)
        bleu_scores_precision = [(candidate, compute_bleu_multiple_references(eval_prefix, candidate, references))
                                 for candidate in candidate_sentences]

        bleu_scores_recall = [(reference, compute_bleu_multiple_references(eval_prefix, reference, candidate_sentences))
                                 for reference in references]
        [candidate.set_bleu_score(bleu_score[1]) for candidate, bleu_score in zip(all_candidates, bleu_scores_precision)]

        translations[eval_index] = all_candidates
        saved_precision_scores[eval_index] = bleu_scores_precision
        saved_recall_scores[eval_index] = bleu_scores_recall
        eval_index += 1
        logging.info('Done: %d'%eval_index)

    save_pkl(recall_fname, saved_recall_scores)
    save_pkl(precision_fname, saved_precision_scores)
    save_pkl(translations_fname, translations)


  def get_imp_words_map(self, args):
    imp_words_map = {}
    rev_imp_words_map = {}

    PREFIX = 'zreptok'
    logging.info('Reading important words from %s'%args.imp_words)
    for line in codecs.open(args.imp_words, 'r', 'utf-8'):
      word = line.strip()
      replaced_word = '%s%s'%(PREFIX, word.lower())
      imp_words_map[word] = replaced_word
      rev_imp_words_map[replaced_word] = word

    return imp_words_map, rev_imp_words_map


  def check_consecutive_repetitions(self, sentence):
    tokens = sentence.split()
    for index in range(len(tokens)):
      if index == len(tokens)-1:
        return False
      if tokens[index] == tokens[index+1]:
        return True
    return False


  def check_repetitions(self, sentence, MAX_COUNT):
    if self.check_consecutive_repetitions(sentence):
      return True
    tokens = sentence.split()
    counter = Counter(tokens)
    for word, freq in counter.most_common():
      if freq > MAX_COUNT:
        return True
    return False


  def replace_and_filter_variations(self, args, variations, rev_imp_words_map, MAX_COUNT=3):
    variations = [variation[0] for variation in variations if variation[1] >= args.min_prob]
    final_variations = []
    for variation in variations:
      tokens = [rev_imp_words_map[token] if token in rev_imp_words_map else token
                            for token in variation.split()]

      replaced_variation = ' '.join(tokens)
      if self.check_repetitions(replaced_variation, MAX_COUNT) is True:
        continue

      final_variations.append(replaced_variation)
    return final_variations

  def generate_variations(self, args):
    var_fw = codecs.open('%s.variations' % args.qs_file, 'w', 'utf-8')
    N = get_num_lines(args.qs_file)
    bar = Bar('Generating variations', max=N)

    imp_words_map, rev_imp_words_map = self.get_imp_words_map(args)
    logging.info('Important words map: %s'%imp_words_map)

    for orig_qs in codecs.open(args.qs_file, 'r', 'utf-8'):
      qs = ' '.join(tokenizer(orig_qs))
      # logging.info('Orig Qs: %s'%qs)
      qs_tokens = [imp_words_map[token] if token in imp_words_map else token for token in qs.split()]
      qs = ' '.join(qs_tokens).lower()
      # logging.info('Repl Qs: %s' %qs)

      write_data = []
      write_data.append(orig_qs.strip())

      variations = self.generate_topk_sequences(sentence=qs, unk_tx=True, tokenize=False)
      variations = self.replace_and_filter_variations(args, variations, rev_imp_words_map)
      write_data.extend(variations)
      var_fw.write(';'.join(write_data) + '\n')
      bar.next()
    var_fw.close()


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model_dir', help='Trained Seq2Seq Model Directory', default=DEF_MODEL_DIR)
  parser.add_argument('-max_unk_symbols', default=8, type=int)
  parser.add_argument('-unk_tx', dest='unk_tx', default=False, action='store_true')

  parser.add_argument('-bleu', dest='bleu', default=False, action='store_true')
  parser.add_argument('-eval_file', dest='eval_file', help='Source and References file', default='wa.eval')
  parser.add_argument('-beam_size', dest='beam_size', default=16, type=int, help='Beam Search size')
  parser.add_argument('-max_refs', dest='max_refs', default=16, type=int, help='Maximum references')
  parser.add_argument('-label', default='base', help='Model label')

  parser.add_argument('-avg', default=False, help='Combine all results to compute avg precision and recall',
                      action='store_true')
  parser.add_argument('-workers', default=100, type=int)

  #Options for question variation generator
  parser.add_argument('-variations', default=False, help='Generate question variations', action='store_true')
  parser.add_argument('-min_prob', default=0.0001, type=float)
  parser.add_argument('-qs_file', dest='qs_file', default=None)
  parser.add_argument('-imp_words', default='imp_words.txt', help='List of important words')
  parser.add_argument('-progress', dest='progress', default=False, action='store_true')

  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)
  sg = SequenceGenerator(model_dir=args.model_dir, max_unk_symbols=args.max_unk_symbols)

  if args.variations:
    sg.generate_variations(args)
  elif args.bleu:
    sg.compute_bleu_scores(args)
  elif args.avg:
    sg.compute_average_bleu_scores(args)

if __name__ == '__main__':
    main()