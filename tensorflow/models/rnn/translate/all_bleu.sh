#!/bin/sh
python get_bleu_score.py trained/vocab-170/test/ input.txt orig.gold.txt 200 -replace -best
./multi-bleu.perl trained/vocab-170/test/orig.gold.txt < trained/vocab-170/test/input.txt.best_bleu 
