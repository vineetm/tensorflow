import argparse, logging, codecs, commands, re
import cPickle as pkl
import numpy as np

RESULTS_SUFFIX = '.results.pkl'
ORIG_PREFIX = 'orig.'
BEST_BLEU_SUFFIX = '.best_bleu'

from commons import replace_line, get_unk_map

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', help='Inputs for which BLEU is to be computed')
  parser.add_argument('gold', help='Gold output')
  parser.add_argument('k', type=int, help='Top K results to consider')
  parser.add_argument('-best', dest='best', action='store_true', default=False)
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


def main():
  #Command line arguments setup
  args = setup_args()
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  #Load Input data, and Input results, Gold_lines
  input_lines = codecs.open(args.inputs, 'r', 'utf-8').readlines()
  orig_input_lines = codecs.open(ORIG_PREFIX + args.inputs, 'r', 'utf-8').readlines()
  input_results = pkl.load(open(args.inputs + RESULTS_SUFFIX, 'r'))
  gold_lines = codecs.open(args.gold, 'r', 'utf-8').readlines()

  assert len(input_lines) == len(input_results)
  assert len(input_lines) == len(gold_lines)
  assert len(orig_input_lines) == len(input_lines)

  logging.info('Num Inputs: %d'%len(input_lines))

  fw = codecs.open(args.inputs + BEST_BLEU_SUFFIX, 'w', 'utf-8')

  for index, result in enumerate(input_results):
    logging.info('Index:%d Len:%d'%(index, len(result)))
    unk_map = get_unk_map(orig_input_lines[index], input_lines[index])
    replaced_lines = [replace_line(line[1], unk_map) for line in result[:args.k]]

    if args.best:
      bleu_scores = [bleu_score(gold_lines[index], line) for line in replaced_lines]
      best_index = np.argmax(bleu_scores)
      logging.info('Line:%d Best:%d Score:%f'%(index, best_index, bleu_scores[best_index]))
    else:
      best_index = 0

    fw.write(replaced_lines[best_index].strip() + '\n')


if __name__ == '__main__':
    main()