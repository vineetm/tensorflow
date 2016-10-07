#!/bin/sh
MODEL_DIR=$1
K=$2
python candidate_generator.py $MODEL_DIR -k $K

