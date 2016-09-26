import argparse, logging, codecs

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Source file')
    parser.add_argument('output', help='Target file')

    args = parser.parse_args()
    return args

SYMBOLS_SET = set(['Q', 'A'])

def convert_phrases(inp, out):
    fr = codecs.open(inp, 'r', 'utf-8')
    fw = codecs.open(out,  'w', 'utf-8')

    for line in fr:
        tokens = line.split()
        target_tokens = []
        for token in tokens:
            if token[0] in SYMBOLS_SET:
                target_tokens.append(token)
            else:
                sub_tokens = token.split('_')
                if len(sub_tokens) == 1:
                    target_tokens.append(token)
                else:
                    target_tokens.extend(sub_tokens)
        fw.write(' '.join(target_tokens) + '\n')


def main():
    args = setup_args()
    logging.info(args)

    convert_phrases(args.input, args.output)

if __name__ == '__main__':
    main()