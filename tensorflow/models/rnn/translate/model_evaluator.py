'''
Evaluate a Trained seq2seq model
'''
import argparse, os
import tensorflow as tf
import cPickle as pkl
from commons import STOPW_FILE, CONFIG_FILE
from seq2seq_model import Seq2SeqModel
from data_utils import initialize_vocabulary
from symbol_assigner import SymbolAssigner

BEAM_SIZE_TEST = [16]
logging = tf.logging
logging.set_verbosity(logging.INFO)

class ModelEvaluator(object):
  def __init__(self, model_path):
    self.model_path = model_path
    self.config = pkl.load(open(os.path.join(self.model_path, CONFIG_FILE)))

    # Create model
    self.model = Seq2SeqModel(
      source_vocab_size=self.config['src_vocab_size'],
      target_vocab_size=self.config['target_vocab_size'],
      buckets=self.config['_buckets'],
      size=self.config['size'],
      num_layers=self.config['num_layers'],
      max_gradient_norm=self.config['max_gradient_norm'],
      batch_size=1,
      learning_rate=self.config['learning_rate'],
      learning_rate_decay_factor=self.config['learning_rate_decay_factor'],
      forward_only=True,
      compute_prob=True,
      num_samples=-1)

    # Restore Model from checkpoint file
    self.session = tf.Session()
    ckpt = tf.train.get_checkpoint_state(self.model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      logging.error('Model not found!')
      return None

    #Load Vocab, reverse vocab
    en_vocab_path = os.path.join(self.config['data_dir'],
                                 "vocab%d.en" % self.config['src_vocab_size'])
    fr_vocab_path = os.path.join(self.config['data_dir'],
                                 "vocab%d.fr" % self.config['target_vocab_size'])

    self.en_vocab, self.rev_en_vocab = initialize_vocabulary(en_vocab_path)
    self.fr_vocab, self.rev_fr_vocab = initialize_vocabulary(fr_vocab_path)
    logging.info('Vocab: Src/Tgt: %d/%d'%(len(self.en_vocab), len(self.fr_vocab)))

    #Setup symbol assigner
    stopw_file = os.path.join(self.config['data_dir'], STOPW_FILE)
    self.sa = SymbolAssigner(stopw_file, valid_entity_list=None, entity_mapping_file=None)


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', help='Trained seq2seq model directory')
  parser.add_argument('eval_file', help='Evaluation file')
  parser.add_argument('-max_refs', dest='max_refs', default=10, type=int, help='Maximum references')
  parser.add_argument('-unk_tx', dest='unk_tx', default=False, action='store_true')
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  me = ModelEvaluator(args.model_path)


if __name__ == '__main__':
  main()