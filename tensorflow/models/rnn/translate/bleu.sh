#!/bin/sh
python get_bleu_score.py trained/diff-symbols/data/ data.dev.en orig.data.dev.fr 100 -replace -best
./multi-bleu.perl trained/diff-symbols/data/orig.data.dev.fr < trained/diff-symbols/data/data.dev.en.best_bleu 

