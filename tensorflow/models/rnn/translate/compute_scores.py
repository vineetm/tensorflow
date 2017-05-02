'''

Inputs: Trained Seq2seq model, Trained Language Model

Generate candidates using beam_search from seq2seq model, and compute LM scores.

'''
import tensorflow as tf
import argparse, codecs, os, commands, tempfile, re
import cPickle as pkl
import numpy as np
from sequence_generator import SequenceGenerator
from language_model import LanguageModel

logging = tf.logging
logging.set_verbosity(logging.INFO)

SAVED_EVAL_FILE = 'eval.pkl'
NUM_PARTS_FILE = 'num_parts.txt'
PARTS_FORMAT = 'p%d.txt'
NOT_SET = -99.0

def execute_shell_command(command, debug=False):
  logging.info('Executing CMD: %s' % command)
  if debug:
    return
  status, output = commands.getstatusoutput(command)
  if status:
    logging.warning(output)

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


class Candidate(object):
  def __init__(self, text, seq2seq_score):
    self.text = text
    self.seq2seq_score = seq2seq_score
    self.bleu = NOT_SET
    self.lm_score = NOT_SET
    self.final_score = NOT_SET

  def set_lm_score(self, lm_score):
    self.lm_score = lm_score

  def set_bleu_score(self, bleu):
    self.bleu = bleu

  def score_str(self):
    return 'Final: %s[Seq: %.4f Lm_Score: %.4f] Bleu: %.4f '\
           %(self.final_score, self.seq2seq_score, self.lm_score, self.bleu)

  def __str__(self):
    return '%s %s'%(self.text, self.score_str())


class EvalDatum(object):
  def __init__(self, input_sentence, references):
    self.input_sentence = input_sentence
    self.references = references
    self.candidates = None

  def set_candidates(self, candidates):
    self.candidates = candidates

  def compute_bleu_score(self, hyp, references):
    def cleanup_files(file_names):
      for file_name in file_names:
        rm_cmd = 'rm -f %s'%file_name
        execute_shell_command(rm_cmd)

    file_names = []
    try:
      f_hyp = tempfile.NamedTemporaryFile(delete=False)
      file_names.append(f_hyp.name)
      f_hyp.write(hyp.strip() + '\n')
      hyp_fname = f_hyp.name

      f_hyp.close()


      ref_fnames = []
      for reference in references:
        f_ref = tempfile.NamedTemporaryFile(delete=False)
        file_names.append(f_ref.name)
        f_ref.write(reference.strip() + '\n')
        ref_fnames.append(f_ref.name)
        f_ref.close()

      bleu = execute_bleu_command(' '.join(ref_fnames), hyp_fname)
      cleanup_files(file_names)
      return bleu
    except Exception:
      cleanup_files(file_names)
      return 0.0


  '''
  Set bleu score for each candidate by comparing against the references
  '''
  def set_bleu_score(self):
    [candidate.set_bleu_score(self.compute_bleu_score(candidate.text, self.references)) for candidate in self.candidates]

  def __str__(self):
    return 'In: %s References: %s Candidates: %s'%\
           (self.input_sentence, ' '.join(self.references), ' '.join(self.candidates))


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-compute_all', default=False, help='Compute scores for all dir', action='store_true')

  parser.add_argument('-compute', default=False, help='Compute scores for a single file', action='store_true')
  parser.add_argument('-eval_dir', help='Main eval directory', default=None)
  parser.add_argument('-eval_file', help='Single eval file', default=None)
  parser.add_argument('-seq2seq_model_dir', default=None)
  parser.add_argument('-mem', default='16g', help='Reserved mem for jobs')
  parser.add_argument('-q', default='x86_1h', help='Job q')
  parser.add_argument('-proj', default='test', help='project name')


  parser.add_argument('-report', default=False, help='Summarize results', action='store_true')
  parser.add_argument('-beam_size', default=16, type=int)
  args = parser.parse_args()
  return args


def generate_fresh_candidates(model_dir, eval_dir, eval_file):
  logging.info('Compute_Scores: Loading Lang Model')
  lm = LanguageModel()
  logging.info('Compute_Scores: Loaded  Lang Model')

  logging.info('Compute_Scores: Loading Seq2seq Model from %s' % model_dir)
  sg = SequenceGenerator(model_dir)
  logging.info('Compute_Scores: Loaded  Seq2seq Model')

  eval_data = []
  for eval_line in codecs.open(os.path.join(eval_dir, eval_file), 'r', 'utf-8'):
    parts = eval_line.split('\t')
    parts = [part.strip() for part in parts]

    input_qs = parts[0]
    references = parts[1:]
    eval_datum = EvalDatum(input_qs, references)

    candidates = sg.generate_topk_sequences(input_qs)
    candidates = [Candidate(candidate[0], candidate[1]) for candidate in candidates]
    logging.info('Index: %d Generated seq2seq candidates'%(len(eval_data)))
    [candidate.set_lm_score(lm.compute_prob(candidate.text)) for candidate in candidates]
    logging.info('Index: %d Generated LM scores' % (len(eval_data)))

    eval_datum.set_candidates(candidates)
    eval_datum.set_bleu_score()
    eval_data.append(eval_datum)

    logging.info('Completed %d'%len(eval_data))
  return eval_data


def generate_candidates_and_compute_scores(model_dir, eval_dir, eval_file):
  saved_candidates_file = os.path.join(model_dir, eval_dir, '%s.%s'%(eval_file, SAVED_EVAL_FILE))
  if os.path.exists(saved_candidates_file):
    all_candidates = pkl.load(open(saved_candidates_file))
    logging.info('Compute_Scores: Loaded %d saved candidates from %s'%(len(all_candidates), saved_candidates_file))
  else:
    logging.info('Compute_Scores: Computing fresh candidates')
    all_candidates = generate_fresh_candidates(model_dir, eval_dir, eval_file)
    with open(saved_candidates_file, 'w') as fw:
      pkl.dump(all_candidates, fw)

def read_num_parts(eval_dir):
  fr = codecs.open(os.path.join(eval_dir, NUM_PARTS_FILE), 'r', 'utf-8')
  num_parts = int(fr.read())
  fr.close()
  return num_parts


def submit_compute_jobs(model_dir, eval_dir, mem, q, proj):
  num_parts = read_num_parts(eval_dir)
  logging.info('Num_parts: %d'%num_parts)
  mkdir_cmd = 'mkdir -p %s' % (os.path.join(model_dir, eval_dir))
  execute_shell_command(mkdir_cmd)

  for part in range(num_parts):
    eval_file = PARTS_FORMAT%part
    python_cmd = 'python compute_scores.py -compute -seq2seq_model_dir %s -eval_dir %s -eval_file %s'\
                 %(model_dir, eval_dir, eval_file)
    logging.info(python_cmd)
    out_file = os.path.join(model_dir, eval_dir, 'p%d.out'%part)
    job_name = 'ev-%d'%part
    jbsub_cmd = 'jbsub -q %s -mem %s -proj %s -out %s -cores 1x1+1 -name %s %s '\
                %(q, mem, proj, out_file, job_name, python_cmd)
    execute_shell_command(jbsub_cmd)


def generate_report(model_dir, eval_dir, beam_size):
  def get_header_data():
    header_data = []
    header_data.append('Input Qs')

    header_data.append('Avg Precision')

    for cand_num in range(beam_size):
      header_data.append('Cand%d'%cand_num)
      header_data.append('Score')

    header_data.append('References')
    return header_data

  num_parts = read_num_parts(eval_dir)
  logging.info('num_parts: %d'%num_parts)
  avg_bleu_scores = []
  report_fw = codecs.open(os.path.join(model_dir, eval_dir, 'report_%d.txt'%(beam_size)), 'w', 'utf-8')

  report_fw.write('\t'.join(get_header_data()) + '\n')
  for part in range(num_parts):
    eval_file = PARTS_FORMAT%part
    saved_candidates_file = os.path.join(model_dir, eval_dir, '%s.%s' % (eval_file, SAVED_EVAL_FILE))
    if not os.path.exists(saved_candidates_file):
      logging.error('Part %d not found at %s'%(part, saved_candidates_file))
      return -99.0
    eval_data = pkl.load(open(saved_candidates_file))

    for eval_datum in eval_data:
      report_part_data = []
      report_part_data.append(eval_datum.input_sentence)

      #Only consider candidates upto beam_size
      candidates = eval_datum.candidates[:beam_size]
      if len(candidates) > 0:
        avg_precision = np.average([candidate.bleu for candidate in candidates])
      else:
        avg_precision = 0.0
      avg_bleu_scores.append(avg_precision)

      report_part_data.append('%.4f'%avg_precision)
      for candidate in candidates:
        report_part_data.append(candidate.text)
        report_part_data.append(candidate.score_str())

      for _ in range(len(candidates), beam_size):
        report_part_data.append('')
        report_part_data.append('')

      [report_part_data.append(reference) for reference in eval_datum.references]
      report_fw.write('\t'.join(report_part_data) + '\n')
  avg_bleu_score = np.average(avg_bleu_scores)
  return avg_bleu_score


def main():
  args = setup_args()
  logging.info(args)

  if args.compute_all:
    submit_compute_jobs(args.seq2seq_model_dir, args.eval_dir, args.mem, args.q, args.proj)
  elif args.compute:
    generate_candidates_and_compute_scores(args.seq2seq_model_dir, args.eval_dir, args.eval_file)
  elif args.report:
    avg_precision = generate_report(args.seq2seq_model_dir, args.eval_dir, args.beam_size)
    logging.info('Model: %s Eval: %s Avg_Pr:%.4f [Beam_Size: %d]'
                 %(args.seq2seq_model_dir, args.eval_dir, avg_precision, args.beam_size))


if __name__ == '__main__':
  main()