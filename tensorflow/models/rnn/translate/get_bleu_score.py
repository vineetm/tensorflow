import argparse, logging, codecs, commands, re, os
import cPickle as pkl
import numpy as np

RESULTS_SUFFIX = '.results.pkl'
ORIG_PREFIX = 'orig.'
BEST_BLEU_SUFFIX = '.best_bleu'

from commons import replace_line, get_unk_map


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('dir', help='Data directory')
  parser.add_argument('inputs', help='Inputs for which BLEU is to be computed')
  parser.add_argument('gold', help='Gold output')
  parser.add_argument('k', type=int, help='Top K results to consider')
  parser.add_argument('-best', dest='best', action='store_true', default=False)
  parser.add_argument('-replace', dest='replace', action='store_true', default=False,
                      help='replace with UNK symbols')
  parser.add_argument('-missing', dest='missing', action='store_true', default=False,
                      help='replace unresolved UNK symbols')

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

def find_unk_symbols(line):
  unk_syms = [token for token in line.split( ) if re.search(r'UNK', token)]
  return set(unk_syms)

def fill_missing_unk(candidates, unk_map):
  unk_candidates = []
  unk_keys = set(unk_map.keys())

  for candidate in candidates:
    unk_symbols = find_unk_symbols(candidate)
    rem_symbols = unk_symbols - unk_keys

    if len(rem_symbols) == 1:
      symbol = list(rem_symbols)[0]
      replacement_symbols = unk_keys - unk_symbols
      unk_candidates.extend([re.sub(symbol, replacement_symbol, candidate)
                             for replacement_symbol in replacement_symbols])
    else:
      unk_candidates.append(candidate)

  return unk_candidates

def main():
  #Command line arguments setup
  args = setup_args()
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  #Load Input data, and Input results, Gold_lines
  input_lines = codecs.open(os.path.join(args.dir, args.inputs), 'r', 'utf-8').readlines()
  orig_input_lines = codecs.open(os.path.join(args.dir, ORIG_PREFIX + args.inputs), 'r', 'utf-8').readlines()

  input_results = pkl.load(open(os.path.join(args.dir, args.inputs + RESULTS_SUFFIX), 'r'))
  gold_lines_unk = codecs.open(os.path.join(args.dir, args.gold[5:]), 'r', 'utf-8').readlines()
  gold_lines = codecs.open(os.path.join(args.dir, args.gold), 'r', 'utf-8').readlines()

  assert len(input_lines) == len(input_results)
  assert len(input_lines) == len(gold_lines)
  assert len(orig_input_lines) == len(input_lines)

  logging.info('Num Inputs: %d'%len(input_lines))

  fw = codecs.open(os.path.join(args.dir, args.inputs + BEST_BLEU_SUFFIX), 'w', 'utf-8')

  for index, result in enumerate(input_results):
    unk_map = get_unk_map(orig_input_lines[index], input_lines[index])

    orig_candidates = [line[1] for line in result[:args.k]]

    if args.missing:
      unk_candidates = fill_missing_unk(orig_candidates, unk_map)
    else:
      unk_candidates = orig_candidates

    logging.info('Orig candidates:%d New:%d'%(len(orig_candidates), len(unk_candidates)))

    if args.replace:
      candidates = [replace_line(candidate, unk_map) for candidate in unk_candidates]
    else:
      candidates  = unk_candidates

    bleu_scores = [bleu_score(gold_lines[index], line) for line in candidates]
    if args.best:
      best_index = np.argmax(bleu_scores)
    else:
      best_index = 0


    logging.info('Orig_Input: %s'%orig_input_lines[index].strip())
    logging.info('Input_UNK: %s'%input_lines[index].strip())

    logging.info('')
    logging.info('Orig_Gold: %s' % gold_lines[index].strip())
    logging.info('Gold_UNK: %s' % gold_lines_unk[index].strip())

    logging.info('')
    logging.info('Selected Best:%d Score:%f' % (best_index, bleu_scores[best_index]))

    for candidate_index in range(len(candidates)):
      logging.info('C_UNK:%d ::%s'%(candidate_index, unk_candidates[candidate_index].strip()))
      logging.info('C:%d[%f] ::%s'%(candidate_index, bleu_scores[candidate_index], candidates[candidate_index]))

    fw.write(candidates[best_index].strip() + '\n')
    logging.info('')


if __name__ == '__main__':
    main()