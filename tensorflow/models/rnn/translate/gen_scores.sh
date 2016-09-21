#!/bin/sh
TRAIN_PATH=trained/diff-symbols
MODEL_PATH='models-b1/translate.ckpt-2000'
VOCAB=189
FR_VOCAB=189
MODEL_SIZE=128
NUM_LAYERS=1
INPUT=trained/diff-symbols/data/data.dev.en
K=5
python get_limited_scores.py $TRAIN_PATH $MODEL_PATH $VOCAB $FR_VOCAB $MODEL_SIZE $INPUT -num_layers $NUM_LAYERS -k $K 
