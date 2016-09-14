import codecs
from nltk.corpus import stopwords

def main():
  dev_lines = codecs.open('data/data.dev.en', 'r', 'utf-8').readlines()
  candidates = set(codecs.open('data/data.train.fr', 'r', 'utf-8').readlines())


  stopw = set(stopwords.words('english')) | set(['?'])
  stopw -= set(['when', 'how', 'what', 'why', 'which', 'who', 'while', 'whom', 'where', 'now', 'with'])


  print stopw
  for index, line in enumerate(dev_lines):
    q2 = line.split('EOS')[2]
    q2_tokens = set(q2.split()) - stopw
    #print q2_tokens
    counter = 0

    for candidate in candidates:
      tokens = set(candidate.split())
      if len(q2_tokens.intersection(tokens)) > 0:
        counter += 1
        # print(q2, candidate)
        # return


    print ('Index: %d Count: %d'%(index, counter))
    if counter == 0:
      print line, index

if __name__ == '__main__':
    main()