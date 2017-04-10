'''
Combine results from all parallel evaluation jobs
'''
import argparse, os, commands, logging, re

PART_PATTERN = 'p%d'
DATA_PARTS = ['report.txt', 'all_inputs.txt', 'all_hyp.txt']
DATA_PARTS_REP = ['all_ref%d.txt']

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir')
  parser.add_argument('-workers', type=int, default=100)
  parser.add_argument('-max_refs', default=10, type=int)
  args = parser.parse_args()
  return args


def join_outputs(model_dir, data_part, workers):
  command = 'cat '
  fname = os.path.join(model_dir, data_part)
  for part_num in range(1, workers+1):
    part_string = PART_PATTERN%part_num
    command += ' %s.%s '%(fname, part_string)

  command += '> %s'%fname
  logging.info('CMD: %s'%command)
  status, output = commands.getstatusoutput(command)
  if status:
    print output


def generate_ref_string(model_dir, workers):
  ref_file = ''
  for ref_num in range(workers):
    ref_file += os.path.join(model_dir, DATA_PARTS_REP[0]%ref_num ) + ' '
  return ref_file


def execute_bleu_command(ref_file, hyp_file):
  command = './multi-bleu.perl %s < %s' % (ref_file, hyp_file)
  status, output = commands.getstatusoutput(command)
  if status:
    print(output)

  match = re.search(r'BLEU\ \=\ (\d+\.\d+)', output)
  if match:
    return float(match.group(1))
  else:
    print 'BLEU not found! %s' % output
    return 0.0


def main():
  args = setup_args()
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  for data_part in DATA_PARTS:
    join_outputs(args.model_dir, data_part, args.workers)

  for rep_data_part in DATA_PARTS_REP:
    for ref_num in range(args.max_refs):
      data_part = rep_data_part%ref_num
      join_outputs(args.model_dir, data_part, args.workers)

  #Execute BLEU
  ref_file = generate_ref_string(args.model_dir, args.max_refs)
  hyp_file = os.path.join(args.model_dir, 'all_hyp.txt')

  bleu = execute_bleu_command(ref_file, hyp_file)
  logging.info('Final BLEU: %f'%bleu)

if __name__ == '__main__':
  main()

