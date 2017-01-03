import os, codecs, argparse
import tensorflow as tf
import cPickle as pkl
import numpy as np
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary, sentence_to_token_ids

from nltk.tokenize import word_tokenize as tokenizer
from textblob.en.np_extractors import FastNPExtractor

from commons import read_stopw, replace_line, replace_phrases, get_diff_map, merge_parts, \
    get_rev_unk_map, generate_missing_symbol_candidates, generate_new_q1_candidates, get_bleu_score, \
    execute_bleu_command, get_unk_map, convert_phrase,  generate_q2_anaphora_candidates

#Constants
from commons import CONFIG_FILE, SUBTREE, LEAVES, RAW_CANDIDATES, DEV_INPUT, DEV_OUTPUT, ORIG_PREFIX, \
    ALL_HYP, ALL_REF, CANDIDATES_SUFFIX, STOPW_FILE, RESULTS_SUFFIX, NOT_SET, SCORES_SUFFIX, UNK_SET


logging = tf.logging


class PendingWork:
    def __init__(self, prob, tree, prefix):
        self.prob = prob
        self.tree = tree
        self.prefix = prefix

    def __str__(self):
        return 'Str=%s(%f)'%(self.prefix, self.prob)


class Score(object):
    def __init__(self, candidate, candidate_unk):
        self.candidate = candidate.strip()
        self.candidate_unk = candidate_unk.strip()
        self.seq2seq_score = NOT_SET
        self.bleu_score = NOT_SET

    def set_seq2seq_score(self, prob):
        self.seq2seq_score = prob

    def set_bleu_score(self, gold_line):
        self.bleu_score = get_bleu_score(gold_line, convert_phrase(self.candidate))

    def __str__(self):
        return 'C    : %s\nC_UNK: %s\nS:%f B:%f'%(self.candidate,
                                                 self.candidate_unk, self.seq2seq_score, self.bleu_score)

class NSUResult(object):
    def __init__(self, training_scores,
                 q1_scores, q2_scores, missing_scores, kw_scores):
        self.training_scores = training_scores
        self.q1_scores = q1_scores
        self.q2_scores = q2_scores
        self.missing_scores = missing_scores
        self.kw_scores = kw_scores

    def set_input(self, input_seq, input_seq_unk, unk_map, rev_unk_map, gold_line):
        self.input_seq = input_seq
        self.input_seq_unk = input_seq_unk
        self.unk_map = unk_map
        self.rev_unk_map = rev_unk_map
        self.gold_line = gold_line.strip()


class CandidateGenerator(object):
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
        #self.stopw = get_stopw()


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

        self.stopw = read_stopw(os.path.join(self.data_path,  STOPW_FILE))
        logging.info('Stopw: %d' % len(self.stopw))
        #Read Candidates and build a prefix tree
        self.build_prefix_tree()
        self.build_keyword_hashmap()

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


    def build_keyword_hashmap(self):
        symbols = [token for token in self.en_vocab if token[0] in UNK_SET]
        self.keywords_map = {}

        for index, candidate in enumerate(self.candidates):
            candidate_tokens = candidate.split()
            key1 = candidate_tokens[0]
            candidate_symbols = set([token for token in candidate_tokens if token[0] in UNK_SET])

            for key2 in candidate_symbols:
                key = '%s %s'%(key1, key2)

                if key not in self.keywords_map:
                    self.keywords_map[key] = set()

                self.keywords_map[key].add(index)

        for key in self.keywords_map:
            logging.info('Key: %s #Candidates: %d'%(key, len(self.keywords_map[key])))


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

        bucket_ids = [b for b in xrange(len(self._buckets))
                      if self._buckets[b][0] > len(token_ids)]

        if len(bucket_ids) == 0:
            bucket_id = len(self._buckets) - 1
        else:
            bucket_id = min(bucket_ids)

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


    '''
    Select a subset of training candidates
    '''
    def select_training_candidates(self, input_line, work_buffer):
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
                    final_scores.append(prefix)
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


    def transform_input(self, input_sentence, compute_phrases=True):
        input_sentence = input_sentence.lower()
        parts = input_sentence.split(';')
        parts = [' '.join(tokenizer(part)) for part in parts]
        if compute_phrases:
            phrases = self.get_phrases(parts)
            replaced_parts = replace_phrases(parts, phrases)
        else:
            replaced_parts = parts
        unk_map = get_diff_map(replaced_parts, self.stopw)
        input_sequence_orig = merge_parts(replaced_parts)
        input_sequence = replace_line(input_sequence_orig, unk_map)
        rev_unk_map = get_rev_unk_map(unk_map)
        return rev_unk_map, unk_map, input_sequence_orig, input_sequence

    def fill_scores(self, candidates, rev_unk_map, input_seq):
        filled_candidates = []
        for candidate in candidates:
            filled_candidate = Score(candidate_unk=candidate, candidate=replace_line(candidate, rev_unk_map))
            filled_candidate.set_seq2seq_score(self.compute_prob(input_seq, candidate))

            filled_candidates.append(filled_candidate)
        return filled_candidates


    def generate_kw_candidates(self, input_seq):
        parts = input_seq.split('EOS')
        q1 = parts[0].split()
        a1 = parts[1].split()
        q2 = parts[2].split()

        key1 = set()
        key1.add(q1[0])

        if q2[0] == 'and':
            if len(q2) > 1:
                key1.add(q2[1])
        else:
            key1.add(q2[0])

        candidate_set = set()
        key2 = set()
        key2 |= set([token for token in a1[:1] if token[0] in UNK_SET])
        key2 |= set([token for token in q2[:1] if token[0] in UNK_SET])

        kw_candidates = []
        for k1 in key1:
            for k2 in key2:
                key = '%s %s'%(k1, k2)

            if key in self.keywords_map:
                candidate_set |= self.keywords_map[key]

        for candidate_index in candidate_set:
            kw_candidates.append(self.candidates[candidate_index])

        return kw_candidates


    '''
    Return Candidates for an input sentence in decreasing order of seq2seq scores
    input_sentence: Conversation (Q1, A1, Q2)
    orig_unk_map: UNK symbol map
    generate_codes: If symbols need to be assigned to tokens in input_sentence
    work_buffer: Work buffer to select inputs from training data
    compute_phrases: If input needs to be converted to phrases
    missing: Replace missing symbols with symbols in UNK Map
    '''
    def get_seq2seq_candidates(self, input_seq, rev_unk_map, work_buffer=5, missing=False, kw_candidates=False):

        new_missing_scores = None
        kw_scores = None

        #Get Scores for all candidates generated from training data
        training_candidates, num_comparisons = self.select_training_candidates(input_seq, work_buffer)
        training_scores = self.fill_scores(training_candidates, rev_unk_map, input_seq)
        logging.info('Num Train Candidates: %d Comparisons:%d'%(len(training_scores), num_comparisons))

        if missing:
            new_missing_candidates = generate_missing_symbol_candidates(training_candidates, rev_unk_map)
            new_missing_scores = self.fill_scores(new_missing_candidates, rev_unk_map, input_seq)
            logging.info('New Missing Candidates: %d'%len(new_missing_candidates))

        if kw_candidates:
            new_candidates = self.generate_kw_candidates(input_seq)
            kw_scores = self.fill_scores(new_candidates, rev_unk_map, input_seq)
            logging.info('New KW Candidates: %d'%len(kw_scores))

        new_q1_candidates = generate_new_q1_candidates(input_seq)
        new_q1_scores = self.fill_scores(new_q1_candidates, rev_unk_map, input_seq)
        logging.info('New Q1 Candidates: %d'%len(new_q1_candidates))

        new_q2_candidates = generate_q2_anaphora_candidates(input_seq)
        new_q2_scores = self.fill_scores(new_q2_candidates, rev_unk_map, input_seq)
        logging.info('New Q2 Candidates: %d' % len(new_q2_candidates))

        nsu_result = NSUResult(training_scores, new_q1_scores, new_q2_scores, new_missing_scores, kw_scores)
        return nsu_result


    def add_candidate_scores(self, scores, current_set):
        new_scores = []
        for score in scores:
            if score.candidate_unk in current_set:
                continue
            new_scores.append(score)
            current_set.add(score.candidate_unk)
        return current_set, new_scores


    def merge_and_sort_scores(self, nsu_result, missing=False, use_q1=True, use_q2=True, kw_candidates=False):
        final_scores = nsu_result.training_scores
        final_candidates_set = set()

        if missing and nsu_result.missing_scores is not None:
            final_candidates_set, new_scores = self.add_candidate_scores(nsu_result.missing_scores, final_candidates_set)
            final_scores.extend(new_scores)
        if use_q1:
            final_candidates_set, new_scores = self.add_candidate_scores(nsu_result.q1_scores,
                                                                         final_candidates_set)
            final_scores.extend(new_scores)
        if use_q2:
            final_candidates_set, new_scores = self.add_candidate_scores(nsu_result.q2_scores,
                                                                         final_candidates_set)
            final_scores.extend(new_scores)

        if kw_candidates:
            final_candidates_set, new_scores = self.add_candidate_scores(nsu_result.kw_scores,
                                                                         final_candidates_set)
            final_scores.extend(new_scores)

        final_scores = sorted(final_scores, key=lambda x: x.seq2seq_score, reverse=True)
        logging.info('# Merged scores: %d'%len(final_scores))
        return final_scores

    def get_raw_seq2seq_candidates(self, input_sentence, compute_phrases=True, missing=True, use_q1=True, use_q2=True):
        rev_unk_map, unk_map, input_seq_orig, input_seq = self.transform_input(input_sentence, compute_phrases)
        logging.info('UNK_Map: %s Reverse UNK_Map: %s' % (str(unk_map), str(rev_unk_map)))
        logging.info('Input_Seq_Orig: %s' % input_seq_orig)
        logging.info('Input_Seq(%d): %s' % (len(input_seq), input_seq))

        nsu_result = self.get_seq2seq_candidates(input_seq, rev_unk_map, missing)
        nsu_result.set_input(input_seq=input_seq_orig, input_seq_unk=input_seq, unk_map=unk_map, rev_unk_map=rev_unk_map)
        results_file = '%s.%s'%('temp', RESULTS_SUFFIX)
        pkl.dump(nsu_result, open(results_file, 'w'))

        final_scores = self.merge_and_sort_scores(nsu_result, missing, use_q1, use_q2)
        return final_scores


    def read_data(self, input_file, output_file, base_dir=None):
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


    def get_max_bleu_score(self, final_scores):
        max_bleu = NOT_SET
        max_bleu_index = NOT_SET

        for index, score in enumerate(final_scores):
            if score.bleu_score > max_bleu:
                max_bleu = score.bleu_score
                max_bleu_index = index

        return max_bleu, max_bleu_index


    def add_all_bleu_scores(self, final_scores, gold_line):
        for index, score in enumerate(final_scores):
            score.set_bleu_score(gold_line)


    def compute_bleu(self, k=100, num_lines=-1, input_file=DEV_INPUT, output_file=DEV_OUTPUT,
                     use_q1=True, use_q2=True, missing=False, kw_candidates=False):
        orig_input_lines, input_lines, orig_gold_lines, gold_lines = self.read_data(input_file, output_file)

        num_inputs = len(input_lines)
        if num_lines > 0:
            num_inputs = num_lines

        candidates_file = '%s.%s' % (self.model_path, CANDIDATES_SUFFIX)
        scores_file = '%s.%d.%s' % (self.model_path, k, SCORES_SUFFIX)

        logging.info('Num inputs: %d'%num_inputs)

        all_hyp_file = os.path.join(self.model_path, ALL_HYP)
        all_ref_file = os.path.join(self.model_path, ALL_REF)

        fw_all_hyp = codecs.open(all_hyp_file, 'w', 'utf-8')
        fw_all_ref = codecs.open(all_ref_file, 'w', 'utf-8')

        perfect_matches = 0

        saved_scores = []
        save_results = False
        if os.path.exists(candidates_file):
            saved_candidates = pkl.load(open(candidates_file))
        else:
            logging.warning('Candidates file missing:%s'%candidates_file)
            saved_candidates = []
            save_results = True

        for index in range(num_inputs):
            gold_line = convert_phrase(orig_gold_lines[index].strip())

            unk_map = get_unk_map(orig_input_lines[index], input_lines[index])
            rev_unk_map = get_rev_unk_map(unk_map)

            if save_results:
                nsu_result = self.get_seq2seq_candidates(input_seq=input_lines[index], rev_unk_map=unk_map,
                                                         missing=missing, kw_candidates=kw_candidates)
            else:
                nsu_result = saved_candidates[index]
                logging.info('Loaded Saved results Tr:%d Q1:%d Q2:%d'%(len(nsu_result.training_scores),
                                                                       len(nsu_result.q1_scores),
                                                                       len(nsu_result.q2_scores)))

            nsu_result.set_input(input_seq=orig_input_lines[index], input_seq_unk=input_lines[index], unk_map=unk_map,
                                 rev_unk_map=rev_unk_map, gold_line=gold_line)

            if save_results:
                saved_candidates.append(nsu_result)

            final_scores = self.merge_and_sort_scores(nsu_result, missing, use_q1, use_q2, kw_candidates)
            self.add_all_bleu_scores(final_scores, gold_line)

            saved_scores.append(final_scores)
            best_bleu_score, best_bleu_index = self.get_max_bleu_score(final_scores)

            if best_bleu_index >= k:
                logging.warning('Line:%d Best:%d BLEU:%f'%(index, best_bleu_index, best_bleu_score))
                final_scores = final_scores[:k]
                best_bleu_score, best_bleu_index = self.get_max_bleu_score(final_scores)

            if best_bleu_score == 100.0:
                perfect_matches += 1

            fw_all_ref.write(gold_line + '\n')
            fw_all_hyp.write(convert_phrase(final_scores[best_bleu_index].candidate) + '\n')

            logging.info('Line:%d Best_BLEU:%f(%d)' % (index, best_bleu_score, best_bleu_index))

            logging.debug('I    : %s' % nsu_result.input_seq.strip())
            logging.debug('I_UNK: %s' % nsu_result.input_seq_unk.strip())
            logging.debug(' ')

            logging.debug('G    : %s' % orig_gold_lines[index].strip())
            logging.debug('G_UNK: %s' % gold_lines[index].strip())
            logging.debug(' ')
            for score_index, score in enumerate(final_scores):
                logging.debug('C    :%d B:%2.2f S:%.3f %s'%(score_index, score.bleu_score, score.seq2seq_score, score.candidate))
                logging.debug('C_UNK:%d B:%2.2f S:%.3f %s'%(score_index, score.bleu_score, score.seq2seq_score, score.candidate_unk))
                logging.debug('')

        fw_all_ref.close()
        fw_all_hyp.close()

        bleu_score = execute_bleu_command(all_ref_file, all_hyp_file)
        logging.info('Perfect Matches: %d/%d'%(perfect_matches, num_inputs))

        if save_results:
            pkl.dump(saved_candidates, open(candidates_file, 'w'))
        pkl.dump(saved_scores, open(scores_file, 'w'))


        return bleu_score, perfect_matches


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Trained Model Directory')
    parser.add_argument('-l', default=-1, type=int, help='# of lines to consider')
    parser.add_argument('-k', default=100, type=int, help='# of Training candidates')
    parser.add_argument('-no_q1', dest='use_q1', default=True, action='store_false')
    parser.add_argument('-no_q2', dest='use_q2', default=True, action='store_false')
    parser.add_argument('-missing',dest='missing', default=False, action='store_true')
    parser.add_argument('-debug', dest='debug', default=False, action='store_true')
    parser.add_argument('-kw_candidates', dest='kw_candidates', default=False, action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = setup_args()
    tm = CandidateGenerator(args.model_dir, debug=args.debug)
    logging.info(args)

    bleu, perfect_matches = tm.compute_bleu(num_lines=args.l, k=args.k,
                                            use_q1=args.use_q1, use_q2=args.use_q2, missing=args.missing, kw_candidates=args.kw_candidates)

    logging.info('BLEU: %f Perfect Matches: %d'%(bleu, perfect_matches))