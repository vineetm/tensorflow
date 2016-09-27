import cPickle as pkl, os
from collections import OrderedDict
from commons import DEFAULT_CANDIDATES, DEFAULT_TREE

def main():
  config = OrderedDict()
  config.train_dir = 'trained/phrases-fast'
  config.data_path = os.path.join(config.train_dir, 'data')
  config.model_path = os.path.join(config.train_dir, 'models/%s'%'translate.ckpt-1800')
  config.src_vocab_size = 202
  config.target_vocab_size = 202
  config.model_size = 128
  config.num_layers = 1
  config.candidates = os.path.join(config.train_dir, DEFAULT_CANDIDATES)
  config.prefix_tree = os.path.join(config.train_dir, DEFAULT_TREE)
  config.stopw_file = 'stopw.txt'

  pkl.dump(config, open('config.pkl', 'w'))

if __name__ == '__main__':
    main()