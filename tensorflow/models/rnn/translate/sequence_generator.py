from commons import CONFIG_FILE, STOPW_FILE, EVAL_DATA
import os, logging, codecs, time, nltk
import tensorflow as tf
import cPickle as pkl
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary, sentence_to_token_ids, EOS_ID

from nltk.tokenize import word_tokenize as tokenizer
from commons import execute_bleu_command, get_num_lines
import numpy as np, argparse
from progress.bar import Bar
from symbol_assigner import SymbolAssigner

ENTITIES = ['PER', 'GEO', 'ORG', 'BLD']

logging = tf.logging
logging.set_verbosity(logging.INFO)

DEF_MODEL_DIR = 'trained-models/model'

class SequenceGenerator(object):
  def __init__(self, model_dir=DEF_MODEL_DIR, eval_file=None):

    config_file_path = os.path.join(model_dir, CONFIG_FILE)
    logging.set_verbosity(logging.INFO)

    logging.info('Loading Pre-trained seq2model:%s' % config_file_path)
    config = pkl.load(open(config_file_path))
    logging.info(config)

    #Create session
    self.session = tf.Session()
    self.eval_file = eval_file

    #Setup parameters using saved config
    self.model_path = config['train_dir']
    self.data_path = config['data_dir']
    self.src_vocab_size = config['src_vocab_size']
    self.target_vocab_size = config['target_vocab_size']
    self._buckets = config['_buckets']

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
    self.sa = SymbolAssigner(stopw_file, entity_mapping_file='entity-map.pkl', valid_entity_list=ENTITIES)

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

  def cleanup_final_translations(self, final_translations, source_sentence):
    sentences_set = set()
    sentences_set.add(source_sentence)

    cleaned_translations = []
    for sentence, prob in final_translations:
      sentence = ' '.join(sentence.split()[:-1])
      if sentence not in sentences_set:

        cleaned_translations.append((sentence, prob))
        sentences_set.add(sentence)

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
        return self.cleanup_final_translations(final_translations, sentence)

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
  def get_bleu_scores(self, list_hypothesis, references):

    bleu_scores = []
    #Write all the references
    ref_string = ''
    for index, reference in enumerate(references):
      ref_file = os.path.join(self.model_path, 'ref%d.txt'%index)
      with codecs.open(ref_file, 'w', 'utf-8') as f:
        f.write(reference.strip() + '\n')
      ref_string += ' %s'%ref_file

    hyp_file = os.path.join(self.model_path, 'hyp.txt')
    for index, hypothesis in enumerate(list_hypothesis):
      with codecs.open(hyp_file, 'w', 'utf-8') as f:
        f.write(hypothesis.strip() + '\n')

      bleu = execute_bleu_command(ref_string, hyp_file)
      bleu_scores.append(bleu)
    return bleu_scores


  def get_nltk_bleu_scores(self, list_hypothesis, references):
    bleu_scores = []

    references_list = [reference.split() for reference in references]
    for hyp in list_hypothesis:
      bleu = nltk.translate.bleu_score.sentence_bleu(references_list, hyp.split())
      bleu_scores.append(bleu)
    return bleu_scores


  def write_bleu_data(self, max_refs, input_sentences, selected_hypothesis, all_references, report_data):
    #Write all references
    ref_file_names = [os.path.join(self.model_path, 'all_ref%d.txt' % index) for index in range(max_refs)]
    ref_fw = [codecs.open(file_name, 'w', 'utf-8') for file_name in ref_file_names]

    for references in all_references:
      for index, reference in enumerate(references):
        ref_fw[index].write(reference.strip() + '\n')
    [fw.close() for fw in ref_fw]

    #Write hypothesis
    hyp_file = os.path.join(self.model_path, 'all_hyp.txt')
    hyp_fw = codecs.open(hyp_file, 'w', 'utf-8')
    for hyp in selected_hypothesis:
      hyp_fw.write(hyp)
    hyp_fw.close()

    inp_fw = codecs.open(os.path.join(self.model_path, 'all_inputs.txt'), 'w', 'utf-8')
    for input_sentence in input_sentences:
      inp_fw.write(input_sentence + '\n')
    inp_fw.close()

    fw_report = codecs.open(os.path.join(self.model_path, 'report.txt'), 'w', 'utf-8')
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


  def get_corpus_bleu_score(self, max_refs, unk_tx, beam_size, progress=False, generate_report=True, entity=False, phrase=False):
    eval_lines = self.read_lines(self.eval_file)

    start_time = time.time()
    if progress:
      N = get_num_lines(self.eval_file)
      bar = Bar('computing bleu', max=N)

    report_data = []
    input_sentences = []
    selected_hypothesis = []
    all_references = []

    for index, line in enumerate(eval_lines):
      write_data=[]
      parts = line.split('\t')

      if progress:
        bar.next()

      input_sentence = parts[0]

      references = [part for part in parts[1:]][:max_refs]
      rem_ref = max_refs - len(references)
      references.extend(['' for _ in range(rem_ref)])

      input_sentence = input_sentence.strip()
      input_sentences.append(input_sentence)

      write_data.append(input_sentence)
      all_hypothesis = self.generate_topk_sequences(input_sentence, unk_tx=unk_tx, beam_size=beam_size, tokenize=False, entity=entity, phrase=phrase)

      list_hypothesis = [hyp[0] for hyp in all_hypothesis]
      list_hypothesis = [self.sa.replace_phrase(hyp) for hyp in list_hypothesis]

      bleu_scores = self.get_bleu_scores(list_hypothesis, references)
      # bleu_scores = self.get_nltk_bleu_scores(list_hypothesis, references)
      if len(bleu_scores) > 0:
        best_index = np.argmax(bleu_scores)
        selected_hypothesis.append(list_hypothesis[best_index].strip() + '\n')
      else:
        selected_hypothesis.append('\n')

      for index in range(len(list_hypothesis)):
        write_data.append(list_hypothesis[index].strip())
        write_data.append(str(bleu_scores[index]))

      all_references.append(references)
      report_data.append(write_data)
    end_time = time.time()
    final_bleu = self.write_bleu_data(max_refs, input_sentences, selected_hypothesis, all_references, report_data)
    logging.info('Final BLEU: %.2f Time: %ds'%(final_bleu, end_time - start_time))


  def generate_variations(self, qs_file, min_prob=0.001):
    var_fw = codecs.open('%s.variations'%qs_file, 'w', 'utf-8')
    N = get_num_lines(qs_file)
    bar = Bar('Generating variations', max=N)
    for qs in codecs.open(qs_file, 'r', 'utf-8'):

      variations = self.generate_topk_sequences(qs.lower(), tokenize=True, unk_tx=True)
      variations = [variation[0] for variation in variations if variation[1] >= min_prob]
      var_fw.write(';'.join(variations) + '\n')

      bar.next()
    var_fw.close()


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model_dir', help='Trained Model Directory', default=DEF_MODEL_DIR)
  parser.add_argument('-eval_file', dest='eval_file', help='Source and References file', default='eval.data')
  parser.add_argument('-beam_size', dest='beam_size', default=16, type=int, help='Beam Search size')
  parser.add_argument('-max_refs', dest='max_refs', default=8, type=int, help='Maximum references')
  parser.add_argument('-progress', dest='progress', default=False, action='store_true')
  parser.add_argument('-bleu', dest='bleu', default=False, action='store_true')
  parser.add_argument('-phrase', dest='phrase', default=False, action='store_true')
  parser.add_argument('-entity', dest='entity', default=False, action='store_true')
  parser.add_argument('-qs_file', dest='qs_file', default=None)
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)
  sg = SequenceGenerator(model_dir=args.model_dir, eval_file=args.eval_file)
  if args.bleu:
    if args.entity:
      sg.get_corpus_bleu_score(max_refs=args.max_refs, beam_size=args.beam_size, unk_tx=False, progress=args.progress, entity=args.entity, phrase=args.phrase)
    else:
      sg.get_corpus_bleu_score(max_refs=args.max_refs, beam_size=args.beam_size, unk_tx=True, progress=args.progress,
                               entity=False, phrase=args.phrase)
  else:
    sg.generate_variations(args.qs_file)



if __name__ == '__main__':
    main()