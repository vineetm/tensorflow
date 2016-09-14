import logging, argparse, codecs
from commons import replace_tokens
from collections import OrderedDict

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('src')
  parser.add_argument('symbols')
  parser.add_argument('input')
  args = parser.parse_args()
  return args

def find_unk_map(orig_line, symbol_line):
  orig_tokens = orig_line.split()
  tokens = symbol_line.split()
  unk_map = OrderedDict()

  for token, orig_token in zip(tokens, orig_tokens):
    if token != orig_token:
      unk_map[token] = orig_token

  return unk_map


def main():
  #Logging setup
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

  #Get all Arguments, and output them to log
  args = setup_args()
  logging.info(args)

  orig_lines = codecs.open(args.src, 'r', 'utf-8').readlines()
  symbol_lines = codecs.open(args.symbols,  'r', 'utf-8').readlines()

  assert len(orig_lines) == len(symbol_lines)

  fw = codecs.open(args.input + '.decoded', 'w', 'utf-8')
  lines = codecs.open(args.input, 'r', 'utf-8').readlines()

  for line, orig_line, symbol_line in zip(lines, orig_lines, symbol_lines):
    unk_map = find_unk_map(orig_line, symbol_line)
    line_tokens = replace_tokens(line.split(), unk_map)
    fw.write(' '.join(line_tokens) + '\n')


if __name__ == '__main__':
    main()