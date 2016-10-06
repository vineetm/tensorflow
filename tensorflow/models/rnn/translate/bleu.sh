#!/bin/sh
MODEL_DIR=$1
K=$2
python translation_model.py $MODEL_DIR -k $K

