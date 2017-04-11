import argparse, logging, os
import cPickle as pkl

#eval.data.p1.translations.pkl.p1
SUB_TX = '%s.p%d.translations.pkl.p%d'
def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir', help='Trained Model Directory')
  parser.add_argument('-final_tx', default='eval.translations.pkl')
  parser.add_argument('-workers', default=100, type=int)
  parser.add_argument('-eval_file', default='eval.data')
  args = parser.parse_args()
  return args

def main():
  args = setup_args()
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  logging.info(args)

  merged_translations = {}
  for cl_num in range(1, args.workers+1):
    sub_tx_fname = os.path.join(args.model_dir, SUB_TX%(args.eval_file, cl_num, cl_num))

    with open(sub_tx_fname) as f:
      sub_tx = pkl.load(f)
      for key in sub_tx:
        real_key = ((cl_num -1) * 10) + key
        merged_translations[real_key] = sub_tx[key]


  logging.info('Final tx: %d'%len(merged_translations))
  with open(os.path.join(args.model_dir, args.final_tx), 'w') as f:
    pkl.dump(merged_translations, f)


if __name__ == '__main__':
  main()