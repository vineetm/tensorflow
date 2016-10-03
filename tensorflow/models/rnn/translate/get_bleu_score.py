import argparse, logging, codecs, commands, re, os
import cPickle as pkl
import numpy as np

RESULTS_SUFFIX = '.results.pkl'
FINAL_RESULTS_SUFFIX = '.final.results.pkl'
ORIG_PREFIX = 'orig.'
BEST_BLEU_SUFFIX = '.best_bleu'

from commons import replace_line, get_unk_map


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('dir', help='Data directory')
  parser.add_argument('inputs', help='Inputs for which BLEU is to be computed')
  parser.add_argument('gold', help='Gold output')
  parser.add_argument('k', type=int, help='Top K results to consider')
  parser.add_argument('-final', dest='final', action='store_true', default=False)
  parser.add_argument('-replace', dest='replace', action='store_true', default=False,
                      help='replace with UNK symbols')
  parser.add_argument('-missing', dest='missing', action='store_true', default=False,
                      help='replace unresolved UNK symbols')
  parser.add_argument('-phrases', dest='phrases', action='store_true', default=False, help='debug mode')
  parser.add_argument('-debug', dest='debug', action='store_true', default=False, help='debug mode')

  args = parser.parse_args()
  return args


def bleu_score(reference, hypothesis):
  with codecs.open('ref.txt', 'w', 'utf-8') as fw_ref:
    fw_ref.write(reference.strip() + '\n')

  with codecs.open('hyp.txt', 'w', 'utf-8') as fw_hyp:
    fw_hyp.write(hypothesis.strip() + '\n')

  command = './multi-bleu.perl %s < %s'%('ref.txt', 'hyp.txt')
  logging.debug('Executing command: %s'%command)

  status, output = commands.getstatusoutput(command)
  if status:
    logging.error(output)

  match = re.search(r'BLEU\ \=\ (\d+\.\d+)', output)
  if match:
    return float(match.group(1))
  else:
    logging.error('BLEU not found! %s'%output)
    return 0.0


def fill_missing_symbols(orig_candidates, unk_map):
  unk_candidates = []
  symbol_start = set(['Q', 'A', '_'])

  for candidate in orig_candidates:
    tokens = candidate.split()
    symbols = [token for token in tokens if token[0] in symbol_start]
    used_symbols = set(unk_map.keys())
    unresolved_symbols = [symbol for symbol in symbols if symbol not in used_symbols]
    logging.debug('C:%s Unresolved:%s'%(candidate, str(unresolved_symbols)))

    if len(unresolved_symbols) == 1:
      unresolved_symbol = unresolved_symbols[0]
      unused_symbols = used_symbols - set(symbols)
      for ununsed_symbol in unused_symbols:
        new_candidate = re.sub(unresolved_symbol, ununsed_symbol, candidate)
        logging.debug('New Candidate: %s'%new_candidate)
        unk_candidates.append(new_candidate)
    else:
      unk_candidates.append(candidate)

  return unk_candidates


def convert_phrases(orig_candidates):
  candidates = []

  for orig_candidate in orig_candidates:
    tokens = orig_candidate.split()
    final_tokens = []
    for token in tokens:
      if token[0] in set(['Q', 'A']):
        final_tokens.append(token)
        continue

      sub_tokens = token.split('_')
      if len(sub_tokens) > 1:
        final_tokens.extend(sub_tokens)
      else:
        final_tokens.append(token)

    candidates.append(' '.join(final_tokens))

  return candidates


def main():
  #Command line arguments setup
  args = setup_args()
  if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  #Load Input data, and Input results, Gold_lines
  input_lines = codecs.open(os.path.join(args.dir, args.inputs), 'r', 'utf-8').readlines()
  orig_input_lines = codecs.open(os.path.join(args.dir, ORIG_PREFIX + args.inputs), 'r', 'utf-8').readlines()

  if args.final:
    input_results = pkl.load(open(os.path.join(args.dir, args.inputs + FINAL_RESULTS_SUFFIX), 'r'))
  else:
    input_results = pkl.load(open(os.path.join(args.dir, args.inputs + RESULTS_SUFFIX), 'r'))

  gold_lines_unk = codecs.open(os.path.join(args.dir, args.gold[5:]), 'r', 'utf-8').readlines()
  gold_lines = codecs.open(os.path.join(args.dir, args.gold), 'r', 'utf-8').readlines()

  if args.phrases:
    gold_lines = convert_phrases(gold_lines)

  assert len(input_lines) == len(input_results)
  assert len(input_lines) == len(gold_lines)
  assert len(orig_input_lines) == len(input_lines)

  logging.info('Num Inputs: %d'%len(input_lines))

  fw = codecs.open(os.path.join(args.dir, args.inputs + BEST_BLEU_SUFFIX), 'w', 'utf-8')

  perfect_matches = 0
  for index, result in enumerate(input_results):
    unk_map = get_unk_map(orig_input_lines[index], input_lines[index])
    orig_candidates = [line[1] for line in result]

    if args.missing:
      unk_candidates = fill_missing_symbols(orig_candidates, unk_map)
      logging.info('Line: %d Orig candidates:%d New:%d' % (index, len(orig_candidates), len(unk_candidates)))
    else:
     unk_candidates = orig_candidates

    unk_candidates = unk_candidates[:args.k]

    if args.replace:
      candidates = [replace_line(candidate, unk_map) for candidate in unk_candidates]
    else:
      candidates  = unk_candidates

    if args.phrases:
      candidates = convert_phrases(candidates)
    bleu_scores = [bleu_score(gold_lines[index], line) for line in candidates]
    best_index = np.argmax(bleu_scores)

    logging.debug('Orig_Input: %s'%orig_input_lines[index].strip())
    logging.debug('Input_UNK: %s'%input_lines[index].strip())

    logging.debug('')
    logging.debug('Orig_Gold: %s' % gold_lines[index].strip())
    logging.debug('Gold_UNK: %s' % gold_lines_unk[index].strip())

    logging.debug('')
    if bleu_scores[best_index] == 100.0:
      perfect_matches += 1
    logging.info('Line: %d Selected Best:%d Score:%f' %(index, best_index, bleu_scores[best_index]))

    for candidate_index in range(len(candidates)):
      logging.debug('C_UNK:%d ::%s'%(candidate_index, unk_candidates[candidate_index].strip()))
      logging.debug('C:%d[%f] ::%s'%(candidate_index, bleu_scores[candidate_index], candidates[candidate_index]))

    fw.write(candidates[best_index].strip() + '\n')
    logging.debug('')

  logging.info('Perfect Matches %d/%d'%(perfect_matches, len(input_lines)))

if __name__ == '__main__':
    main()
