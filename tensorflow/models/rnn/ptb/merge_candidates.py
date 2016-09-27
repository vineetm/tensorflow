import argparse, logging, cPickle as pkl, os, codecs, re
from commons import ORIG_PREFIX, RESULTS_SUFFIX, SOURCE, TARGET, get_unk_map, FINAL_RESULTS_SUFFIX, replace_line

'''
Commandline arguments
'''
def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('seq2seq_dir', help='Seq2seq data directory')
  parser.add_argument('-candidates', default='data.dev')
  parser.add_argument('-weight', default=0.5, type=float, help='LM Weight')
  parser.add_argument('-missing', dest='missing', action='store_true', default=False,
                      help='replace unresolved UNK symbols')
  parser.add_argument('-debug', dest='debug', help='debug mode', default=False, action='store_true')
  args = parser.parse_args()
  if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)
  return args


'''
Load Seq2seq UNK Candidates
'''
def get_seq2seq_candidates(args):
  candidates_path = os.path.join(args.seq2seq_dir, '%s.%s.%s'%(args.candidates, SOURCE, RESULTS_SUFFIX))
  candidates = pkl.load(open(candidates_path))
  logging.info('Seq2Seq Candidates file: %s #Candidates: %d'%(candidates_path, len(candidates)))
  return candidates

'''
Load inputs, inputs_unk
Needed to generate UNK Symbols
'''
def get_inputs(args):
  inputs_path = os.path.join(args.seq2seq_dir, '%s%s.%s'%(ORIG_PREFIX, args.candidates, SOURCE))
  inputs = codecs.open(inputs_path, 'r', 'utf-8').readlines()

  inputs_unk_path = os.path.join(args.seq2seq_dir, '%s.%s'%(args.candidates, SOURCE))
  inputs_unk = codecs.open(inputs_unk_path, 'r', 'utf-8').readlines()

  logging.info('#Inputs: %d'%len(inputs))
  return inputs, inputs_unk


def fill_missing_symbols(orig_candidates, unk_map):
  unk_candidates = []
  symbol_start = set(['Q', 'A', '_'])

  for (prob, candidate) in orig_candidates:
    tokens = candidate.split()
    symbols = set([token for token in tokens if token[0] in symbol_start])
    unresolved_symbols = symbols - set(unk_map.keys())
    logging.debug('C:%s Unresolved:%s'%(candidate, str(unresolved_symbols)))

    if len(unresolved_symbols) == 0:
      unk_candidates.append((prob, candidate))
      continue

    if len(unresolved_symbols) == 1:
      unresolved_symbol = list(unresolved_symbols)[0]
      unused_symbols = set(unk_map.keys()) - symbols
      for ununsed_symbol in unused_symbols:
        new_candidate = re.sub(unresolved_symbol, ununsed_symbol, candidate)
        logging.debug('New Candidate: %s'%new_candidate)
        unk_candidates.append((prob, new_candidate))
  return unk_candidates


def write_candidates(final_candidates, args):
  candidates_path = os.path.join(args.seq2seq_dir, '%s.%s.%s'%(args.candidates, SOURCE, FINAL_RESULTS_SUFFIX))
  logging.info('Writing Final Candidates to %s'%candidates_path)
  pkl.dump(final_candidates, open(candidates_path, 'w'))



def main():
  args = setup_args()
  seq2seq_candidates = get_seq2seq_candidates(args)
  inputs, inputs_unk = get_inputs(args)
  assert len(inputs) == len(inputs_unk)

  final_candidates = []
  for index, seq2seq_candidate in enumerate(seq2seq_candidates):
    unk_map = get_unk_map(inputs[index], inputs_unk[index])
    logging.debug('UNK_Map: %s'%str(unk_map))

    if args.missing:
      unk_candidates = fill_missing_symbols(seq2seq_candidate, unk_map)
    else:
      unk_candidates = seq2seq_candidate

    candidates = [replace_line(line, unk_map) for line in unk_candidates]
    final_candidates.append(candidates)
    logging.info('Orig candidates:%d New:%d' % (len(seq2seq_candidate), len(candidates)))

  write_candidates(final_candidates, args)

if __name__ == '__main__':
    main()