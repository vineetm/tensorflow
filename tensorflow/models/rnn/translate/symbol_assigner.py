import codecs, logging, os
import cPickle as pkl

DET = set(['a', 'an', 'the'])

class SymbolAssigner(object):
  def __init__(self, stopw_file, valid_entity_list=None, entity_mapping_file=None, default_unk_symbol='UNK', max_len=8):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    self.max_len = max_len
    self.default_unk_symbol = default_unk_symbol
    self.read_stopw(stopw_file)
    logging.info('#Stopwords: %d'%(len(self.stopw)))

    if valid_entity_list is None:
      self.entity_mapping = None
    else:
      self.entity_set = set(valid_entity_list)
      self.filter_entity(entity_mapping_file)
      self.find_longest_entity()


  def find_longest_entity(self):
    longest_entity = None
    max_len = 0

    for entity in self.entity_mapping:
      num_tokens = len(entity.split('_'))
      if num_tokens > max_len:
        longest_entity = entity
        max_len = num_tokens
    logging.info('Max_len: %d Longest_entity: %s'%(max_len, longest_entity))

  def replace_phrase(self, sentence):
    return sentence.replace('_', ' ')

  def filter_entity(self, entity_mapping_file):
    self.entity_mapping = {}
    skipped_len = 0
    conflict = 0
    candidate_entity_map = pkl.load(open(entity_mapping_file))
    logging.info('Candidates: %d'%len(candidate_entity_map))
    problem_keys = set()
    for entity in candidate_entity_map:

      #Skip entities we are not interested in
      if candidate_entity_map[entity] not in self.entity_set:
        continue

      #Skip entities which have a disambiguation using '('
      if entity.find('(') > 0:
        continue

      parts = entity.split('_')
      num_tokens = len(parts)
      # if num_tokens > 1 and parts[0].lower() in DET:
      #   logging.info('Skip %s %s'%(entity, candidate_entity_map[entity]))
      #   continue

      if num_tokens > self.max_len:
        skipped_len += 1
        continue

      entity_lower = entity.lower()

      if entity_lower in self.entity_mapping and self.entity_mapping[entity_lower] != candidate_entity_map[entity]:
        # logging.warn('Key:%s entity_old:%s entity_new:%s'%(entity_lower,
        #                                                    self.entity_mapping[entity_lower], candidate_entity_map[entity]))
        conflict += 1
        problem_keys.add(entity_lower)
        continue

      self.entity_mapping[entity_lower] = candidate_entity_map[entity]

    logging.info('#Problem keys: %d'%len(problem_keys))
    for key in problem_keys:
      del self.entity_mapping[key]


    del candidate_entity_map
    logging.info('Entities: %d Skipped_len: %d Conflict:%d'%(len(self.entity_mapping), skipped_len, conflict))


  #Phrases are built using entity map
  def convert_to_phrases(self, sentence):
    if self.entity_mapping is None:
      logging.warn('No entity map found')
      return None

    tokens = sentence.split()
    entities_found = set()
    for max_len in reversed(range(1, self.max_len+1)):
      start_index = 0
      final_tokens = []
      while start_index < len(tokens):
        key = '_'.join(tokens[start_index: start_index+ max_len])
        if key in self.entity_mapping and key not in entities_found:
          final_tokens.append(key)
          start_index += len(key.split('_'))
          entities_found.add(key)
        else:
          final_tokens.append(tokens[start_index])
          start_index += 1
      tokens = final_tokens

    del entities_found
    return ' '.join(tokens)

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
      unk_map[token] = '%s%d'%(self.default_unk_symbol, len(unk_map) + 1)
    return unk_map


  def build_entity_unk_map(self, phrase_sentence, existing_unk_map=None, existing_unk_num_map=None):
    unk_map = {}
    unk_num_map = {}
    if existing_unk_map is not None:
      for key in existing_unk_map:
        unk_map[key] = existing_unk_map[key]

    if existing_unk_num_map is not None:
      for key in existing_unk_num_map:
        unk_num_map[key] = existing_unk_num_map[key]

    tokens = phrase_sentence.split()
    for token in tokens:
      if token in unk_map:
        continue
      elif token in self.entity_mapping:
        entity = self.entity_mapping[token]
        if entity not in unk_num_map:
          unk_num_map[entity] = 1
        else:
          unk_num_map[entity] = unk_num_map[entity] + 1
        unk_symbol = '%s_%s%d'%(self.default_unk_symbol, entity, unk_num_map[entity])
        unk_map[token] = unk_symbol

      elif token in self.stopw:
        continue
      else:
        if self.default_unk_symbol not in unk_num_map:
          unk_num_map[self.default_unk_symbol] = 1
        else:
          unk_num_map[self.default_unk_symbol] = unk_num_map[self.default_unk_symbol] + 1
        unk_symbol = '%s%d'%(self.default_unk_symbol, unk_num_map[self.default_unk_symbol])
        unk_map[token] = unk_symbol

    return unk_map, unk_num_map


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

