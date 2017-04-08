#Only used for logging framework
import commands, logging, argparse, os, codecs

#Define command constants
EVAL_COMMAND = 'python sequence_generator.py -model_dir %s -bleu -eval_file %s'
EVAL_COMMAND_UNK_TX = EVAL_COMMAND + ' -unk_tx'
EVAL_COMMAND_UNK_SUFFIX= EVAL_COMMAND_UNK_TX + ' -suffix p%d'

JBSUB_COMMAND = 'jbsub -name %s -out %s -proj %s -cores 1x1+1 -q %s -mem %s %s'


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir', help='model directory')
  parser.add_argument('job_name', help='job name')
  parser.add_argument('-split', dest='split', default=False, action='store_true', help='Split eval file for workers')
  parser.add_argument('-eval_file', default='eval.data', help='Evaluation file')
  parser.add_argument('-q', default='x86_1h', help='Job queue')
  parser.add_argument('-proj', default='test', help='Project')
  parser.add_argument('-workers', default=1, help='Num of eval jobs', type=int)
  parser.add_argument('-mem', default='4g', help='Mem per job')
  args = parser.parse_args()
  return args


def submit_eval_jobs(args):
  if args.workers == 1:
    python_command = EVAL_COMMAND_UNK_TX%(args.model_dir, args.eval_file)
    out_file = os.path.join(args.model_dir, 'test.out')
    command = JBSUB_COMMAND%(args.job_name, out_file, args.proj, args.q, args.mem, python_command)
    logging.info('CMD: %s'%command)
    status, output = commands.getstatusoutput(command)
    if status:
      print output

  else:
    out_file = os.path.join(args.model_dir, 'test.out')
    for worker_num in range(1, args.workers+1):
      python_command = EVAL_COMMAND_UNK_SUFFIX % (args.model_dir, args.eval_file, worker_num)
      job_name = '%s-%d'%(args.job_name, worker_num)
      worker_out_file = '%s-%d'%(out_file, worker_num)
      command = JBSUB_COMMAND % (job_name, worker_out_file, args.proj, args.q, args.mem ,python_command)
      status, output = commands.getstatusoutput(command)
      if status:
        print output


def split_eval_file(args):
  eval_lines = codecs.open(args.eval_file, 'r', 'utf-8').readlines()

  if len(eval_lines) < args.workers:
    logging.error('Bad split Lines: %d Workers: %d'%(len(eval_lines), args.workers))
    return

  num_lines = len(eval_lines) / args.workers
  part_num = 1
  fw = None
  for index, line in enumerate(eval_lines):
    if index % num_lines == 0:
      if fw is not None:
        fw.close()
      fw = codecs.open('%s.p%d' % (args.eval_file, part_num), 'w', 'utf-8')
      part_num += 1
    fw.write(line)

  if fw is not None:
    fw.close()


def main():
  args = setup_args()
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  if args.split:
    split_eval_file(args)
  else:
    submit_eval_jobs(args)



if __name__ == '__main__':
  main()