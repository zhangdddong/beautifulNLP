import tensorflow as tf
from data_utils import Vocabulary, batch_generator
from model import LSTMModel
import os
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_path', 'checkpoint/base', 'model path')
tf.flags.DEFINE_integer('batch_size', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.flags.DEFINE_string('vocab_file', '', 'vocabulary pkl file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    if os.path.exists(checkpoint_path) is False:
        os.makedirs(checkpoint_path)
    
    # 读取训练文本
    with open(datafile, 'r', encoding='utf-8') as f:
        train_data = f.read()

    # 加载/生成 词典
    vocabulary = Vocabulary()
    if FLAGS.vocab_file:
        vocabulary.load_vocab(FLAGS.vocab_file)
    else:
        vocabulary.build_vocab(train_data)
    vocabulary.save(FLAGS.vocab_file)

    input_ids = vocabulary.encode(train_data)

    g = batch_generator(input_ids, FLAGS.batch_size, FLAGS.num_steps)

    model = LSTMModel(vocabulary.vocab_size,
                    batch_size=FLAGS.batch_size,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                checkpoint_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
