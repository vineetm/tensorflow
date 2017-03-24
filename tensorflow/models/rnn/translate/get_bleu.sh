#!/bin/sh
MODELS_DIR=$1
BEAM_SIZE=$2
python sequence_generator.py -model_dir $MODELS_DIR -beam_size $BEAM_SIZE -progress -bleu
./multi-bleu.perl $MODELS_DIR/all_ref0.txt $MODELS_DIR/all_ref1.txt $MODELS_DIR/all_ref2.txt $MODELS_DIR/all_ref3.txt $MODELS_DIR/all_ref4.txt $MODELS_DIR/all_ref5.txt $MODELS_DIR/all_ref6.txt $MODELS_DIR/all_ref7.txt  $MODELS_DIR/all_ref8.txt $MODELS_DIR/all_ref9.txt < $MODELS_DIR/all_hyp.txt
