from commons import CONFIG_FILE, STOPW_FILE, EVAL_DATA
import os, logging, codecs
import tensorflow as tf
import cPickle as pkl
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary, sentence_to_token_ids, EOS_ID
from collections import OrderedDict
from nltk.tokenize import word_tokenize as tokenizer
from commons import execute_bleu_command
import numpy as np, argparse
from progress.bar import Bar

logging = tf.logging

class SequenceGenerator(object):
  def __init__(self, model_dir, eval_file=None, beam_size=16, eval_dir=None):

    config_file_path = os.path.join(model_dir, CONFIG_FILE)
    logging.set_verbosity(logging.INFO)

    logging.info('Loading Pre-trained seq2model:%s' % config_file_path)
    config = pkl.load(open(config_file_path))
    logging.info(config)

    #Create session
    self.session = tf.Session()
    self.beam_size = beam_size
    self.eval_dir = eval_dir
    self.eval_file = eval_file

    #Setup parameters using saved config
    self.model_path = config['train_dir']
    self.data_path = config['data_dir']
    self.src_vocab_size = config['src_vocab_size']
    self.target_vocab_size = config['target_vocab_size']
    self._buckets = config['_buckets']

    compute_prob = True
    if self.beam_size == 1:
      compute_prob = False

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
      compute_prob=compute_prob)

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

    self.stopw = set()
    with codecs.open(os.path.join(self.data_path,  STOPW_FILE)) as f:
      for line in f:
        self.stopw.add(line.strip())

    logging.info('#Stopwords: %d'%len(self.stopw))


  def replace_tokens(self, tokens, unk_map):
    replaced_tokens = [unk_map[token] if token in unk_map else token for token in tokens]
    return ' '.join(replaced_tokens)


  def get_unk_map(self, tokens, unk_map=None):
    if unk_map is None:
      unk_map = OrderedDict()
    for token in tokens:
      if token in self.stopw:
        continue
      if token in unk_map:
        continue
      unk_map[token] = '%s%d' % ('UNK', len(unk_map) + 1)

    rev_unk_map = {}
    for token in unk_map:
      rev_unk_map[unk_map[token]] = token

    return unk_map, rev_unk_map


  def convert_to_unk_sequence(self, sentence):
    sentence = sentence.lower()
    tokens = tokenizer(sentence)
    unk_map, rev_unk_map = self.get_unk_map(tokens)
    unk_sentence = self.replace_tokens(tokens, unk_map)

    return unk_sentence, unk_map, rev_unk_map


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


  def get_new_work(self, encoder_inputs, decoder_inputs, target_weights, bucket_id, fixed_tokens, curr_prob):
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
    return new_work[:self.beam_size]


  def generate_topk_sequences(self, sentence, unk_tx=True):
    if self.beam_size == 1:
      return self.generate_output_sequence(sentence)

    if unk_tx:
      unk_sentence, unk_map, rev_unk_map = self.convert_to_unk_sequence(sentence)
    else:
      unk_sentence = sentence

    token_ids = sentence_to_token_ids(tf.compat.as_bytes(unk_sentence), self.en_vocab, normalize_digits=False)
    sentence_lc = ' '.join(tokenizer(sentence.lower()))
    # logging.info('Input: %s'%sentence_lc)
    #
    # if unk_tx:
    #   logging.info('UNK  :%s'%unk_sentence)

    bucket_ids = [b for b in xrange(len(self._buckets))
                      if self._buckets[b][0] > len(token_ids)]

    if len(bucket_ids) == 0:
      bucket_id = len(self._buckets) - 1
    else:
      bucket_id = min(bucket_ids)

    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)

    #Initialize with empty fixed tokens
    rem_work = self.get_new_work(encoder_inputs, decoder_inputs, target_weights, bucket_id, [], 1.0)
    # logging.info(rem_work)
    final_translations = []
    while True:
      if len(rem_work) == 0:
        if unk_tx:
          final_translations = [(self.replace_tokens(final_translation[0].split(), rev_unk_map), final_translation[1]) for final_translation in final_translations]
        final_translations = sorted(final_translations, key=lambda x: x[1], reverse=True)[:self.beam_size]
        return self.cleanup_final_translations(final_translations, sentence_lc)

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

      new_work = self.get_new_work(encoder_inputs, decoder_inputs, target_weights, bucket_id, curr_tokens, curr_prob)
      rem_work.extend(new_work)
      rem_work = sorted(rem_work, key=lambda x:x[1], reverse=True)
      rem_work = rem_work[:self.beam_size]


  '''
  Get BLEU score for each hypothesis
  '''
  def get_best_bleu_score(self, list_hypothesis, references):
    best_bleu = 0.0
    best_index = 0

    #Write all the references
    ref_string = ''
    for index, reference in enumerate(references):
      ref_file = os.path.join(self.eval_dir, 'ref%d.txt'%index)
      with codecs.open(ref_file, 'w', 'utf-8') as f:
        f.write(reference.strip() + '\n')
      ref_string += ' %s'%ref_file

    hyp_file = os.path.join(self.eval_dir, 'hyp.txt')
    for index, hypothesis in enumerate(list_hypothesis):
      with codecs.open(hyp_file, 'w', 'utf-8') as f:
        f.write(hypothesis.strip() + '\n')

      bleu = execute_bleu_command(ref_string, hyp_file)
      if bleu > best_bleu:
        best_bleu = bleu
        best_index = index

    return best_bleu, best_index

  def get_corpus_bleu_score(self, max_refs):
    ref_file_names = [os.path.join(self.eval_dir, 'all_ref%d.txt'%index) for index in range(max_refs)]
    ref_fw = [codecs.open(file_name, 'w', 'utf-8') for file_name in ref_file_names]

    hyp_file = os.path.join(self.eval_dir, 'all_hyp.txt')
    hyp_fw = codecs.open(hyp_file, 'w', 'utf-8')

    inp_fw = codecs.open(os.path.join(self.eval_dir, 'all_inputs.txt'), 'w', 'utf-8')
    skipped_noq = 0
    skipped_noref = 0
    ref_str = ' '.join(ref_file_names)

    bar = Bar('computing bleu', max=100)
    for line in codecs.open(self.eval_file, 'r', 'utf-8'):
      parts = line.split('\t')

      bar.next()
      #Skip if first part is not a question
      if parts[0][:2] != 'q:':
        skipped_noq += 1
        continue

      input_sentence = ' '.join(tokenizer(parts[0][2:]))

      references = [part[2:] for part in parts[1:] if part[:2] == 'q:'][:max_refs]

      rem_ref = max_refs - len(references)
      references.extend(['' for _ in range(rem_ref)])
      # logging.info('Ref: %s'%references)
      references = [' '.join(tokenizer(reference)) for reference in references]

      #Skip if no references are found!
      if len(references) == 0:
        skipped_noref += 1
        continue

      inp_fw.write(input_sentence.strip() + '\n')
      all_hypothesis = self.generate_topk_sequences(input_sentence)
      list_hypothesis = [hyp[0] for hyp in all_hypothesis]

      best_bleu, best_index = self.get_best_bleu_score(list_hypothesis, references)
      hyp_fw.write(list_hypothesis[best_index].strip() + '\n')
      for index, reference in enumerate(references):
        ref_fw[index].write(reference.strip() + '\n')

    hyp_fw.close()
    [fw.close() for fw in ref_fw]
    final_bleu = execute_bleu_command(ref_str, hyp_file)
    logging.info('Final BLEU: %.2f'%final_bleu)
    logging.info('Skipped: No_q:%d No_ref:%d'%(skipped_noq, skipped_noref))


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir', help='Trained Model Directory')
  parser.add_argument('eval_file', help='Source and References file')
  parser.add_argument('-beam_size', dest='beam_size', default=16, type=int, help='Beam Search size')
  parser.add_argument('-max_refs', dest='max_refs', default=8, type=int, help='Maximum references')
  parser.add_argument('-eval_dir', dest='eval_dir', default='eval', help='Eval results directory')

  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)
  sg = SequenceGenerator(model_dir=args.model_dir, eval_file=args.eval_file, beam_size=args.beam_size, eval_dir=args.eval_dir)
  # logging.info(sg.get_best_bleu_score(['how can my laptop be fixed ?'], ['how can my fix my laptop ?', 'how can my laptop ?']))
  sg.get_corpus_bleu_score(args.max_refs)
  # sg.save_results(args.results)


if __name__ == '__main__':
    main()