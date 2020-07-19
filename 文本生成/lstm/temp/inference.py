import tensorflow as tf
from data_utils import Vocabulary
from model import LSTMModel
import os
from IPython import embed

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('vocab_file', '', 'vocabulary pkl file')
tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')


def main(_):

    vocabulary = Vocabulary()
    vocabulary.load_vocab(FLAGS.vocab_file)

    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = LSTMModel(vocabulary.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = vocabulary.encode(FLAGS.start_string)
    arr = model.predict(FLAGS.max_length, start, vocabulary.vocab_size)
    print(vocabulary.decode(arr))


if __name__ == '__main__':
    tf.app.run()
