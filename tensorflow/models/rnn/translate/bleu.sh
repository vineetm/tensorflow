#!/bin/sh
BASE_DIR=$1
MODEL_DIR=$BASE_DIR/models
DATA_DIR=$BASE_DIR/data
python candidate_generator.py $MODEL_DIR

