#!/bin/sh
TRAIN_PATH=trained/vocab-170
MODEL_PATH='translate.ckpt-1600'
VOCAB=170
MODEL_SIZE=128
INPUT=trained/vocab-170/test/input.txt
CANDIDATES=data/data.train.fr
python generate_scores.py $TRAIN_PATH $MODEL_PATH $VOCAB $VOCAB $MODEL_SIZE $INPUT $CANDIDATES
