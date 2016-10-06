1. Train Model
python translate.py --data_dir trained/vocab-x/data --train_dir trained/vocab-x/models --size 128 --num_layers 1 --en_vocab_size 300 --fr_vocab_size 30

2. Process Candidates: Build a Prefix Tree
python process_candidates.py <train_dir> <target_vocab_size>

3. Pick K candidates for each sentence in input
./gen_scores.sh

4. Get BELU Scores
