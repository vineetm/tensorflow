#!/bin/sh
MODELS_DIR=$1
./multi-bleu.perl $MODELS_DIR/all_ref0.txt $MODELS_DIR/all_ref1.txt $MODELS_DIR/all_ref2.txt $MODELS_DIR/all_ref3.txt $MODELS_DIR/all_ref4.txt $MODELS_DIR/all_ref5.txt $MODELS_DIR/all_ref6.txt $MODELS_DIR/all_ref7.txt < $MODELS_DIR/all_hyp.txt
