import codecs, logging, os
import cPickle as pkl

class SymbolAssigner(object):
  def __init__(self, stopw_file, valid_entity_list=None, entity_mapping=None, default_unk_symbol='UNK'):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    self.deault_unk_symbol = default_unk_symbol
    self.read_stopw(stopw_file)
    logging.info('#Stopwords: %d'%(len(self.stopw)))

    if valid_entity_list is None:
      self.entity_mapping = None
    else:
      self.entity_set = set(valid_entity_list)
      if entity_mapping is not None:
        with open(entity_mapping) as fr:
          self.entity_mapping = pkl.load(fr)
          self.max_entity_len = 0

      #Find longest *valid* entity
      for key in self.entity_mapping:
        if self.entity_mapping[key] not in self.entity_set:
          continue
        parts = key.split('_')
        if len(parts) > self.max_entity_len:
          self.max_entity_len = len(parts)
      logging.info('Entity_Map size: %d Max_len: %d'%(len(self.entity_mapping), self.max_entity_len))

  #Phrases are built using entity map
  def convert_to_phrases(self, sentence):
    if self.entity_mapping is None:
      logging.warn('No entity map found')
      return None



  def build_unk_map(self, sentence, existing_unk_map=None):
    unk_map = {}
    if existing_unk_map is not None:
      for key in existing_unk_map:
        unk_map[key] = existing_unk_map[key]

    tokens = sentence.split()
    for token in tokens:
      if token in unk_map:
        continue
      if token in self.stopw:
        continue
      unk_map[token] = '%s%d'%(self.deault_unk_symbol, len(unk_map) + 1)
    return unk_map


  def recover_orginal_sentence(self, unk_sentence, unk_map):
    rev_unk_map = dict(zip(unk_map.values(), unk_map.keys()))
    return self.assign_unk_symbols(unk_sentence, rev_unk_map)


  def assign_unk_symbols(self, sentence, unk_map):
    unk_tokens = [unk_map[token] if token in unk_map else token for token in sentence.split()]
    return ' '.join(unk_tokens)


  def read_stopw(self, stopw_file):
    self.stopw = set()
    if os.path.exists(stopw_file) is None:
      logging.warn('Stopwords file: %s does not exist'%stopw_file)
      return
    with codecs.open(stopw_file, 'r', 'utf-8') as fr:
      for line in fr:
        self.stopw.add(line.strip())

