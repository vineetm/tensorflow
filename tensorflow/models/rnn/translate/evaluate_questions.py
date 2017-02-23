from sequence_generator import SequenceGenerator
from progress.bar import Bar
import os, codecs
import tensorflow as tf
import cPickle as pkl

logging = tf.logging
logging.set_verbosity(logging.INFO)




tf.app.flags.DEFINE_string('models_dir', None, 'Models directory')
tf.app.flags.DEFINE_string('out_file', None, 'model predictions file')
tf.app.flags.DEFINE_string('eval_file', 'data.dev.en', 'Evaluation source file')
tf.app.flags.DEFINE_boolean('splitqs', False, 'split output on ?')
FLAGS = tf.app.flags.FLAGS


def main(_):
  #Load Model config
  config = pkl.load(open(os.path.join(FLAGS.models_dir, 'config.ckpt')))
  logging.info(config)

  sg = SequenceGenerator(FLAGS.models_dir, beam_size=1)

  #Test Question
  answer = 'sky is blue because of refraction.'
  question = sg.generate_output_sequence(answer, unk_tx=False)
  logging.info('Test Ans: %s'%answer)
  logging.info('Test Qs : %s'%question)

  fw = codecs.open(FLAGS.out_file, 'w', 'utf-8')
  fr = codecs.open(os.path.join(config['data_dir'], FLAGS.eval_file), 'r', 'utf-8')

  N = 10000
  bar = Bar('Evaluating Data', max=N)
  for eval_line in fr:
    # logging.info('Eval_line: %s'%eval_line)
    model_output = sg.generate_output_sequence(eval_line, unk_tx=False)
    # logging.info('out_line: %s' % model_output)
    if FLAGS.splitqs:
      qs_index = model_output.find('?')
      if qs_index > 0:
        model_output = model_output[:qs_index+1]
    fw.write(model_output + '\n')
    bar.next()


if __name__ == '__main__':
    tf.app.run()