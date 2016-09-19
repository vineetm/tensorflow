#!/bin/sh
#python get_bleu_score.py trained/vocab-170/data/ data.dev.en orig.data.dev.fr 100 -replace
#./multi-bleu.perl trained/vocab-170/data/orig.data.dev.fr < trained/vocab-170/data/data.dev.en.best_bleu 

python get_bleu_score.py trained/vocab-170/test/ input.txt orig.gold.txt 100 -replace -best
./multi-bleu.perl trained/vocab-170/test/orig.gold.txt < trained/vocab-170/test/input.txt.best_bleu 
