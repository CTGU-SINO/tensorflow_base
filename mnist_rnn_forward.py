import tensorflow as tf

INPUT_SIZE = 28
HIDDEN_NODE = 128
OUTPUT_SIZE = 10
N_STEP = 28


def _RNN(X):
    basic_Cell = tf.contrib.rnn.BasicRNNCell(num_units=HIDDEN_NODE, activation=tf.nn.relu)
    outputs1, states = tf.nn.dynamic_rnn(basic_Cell, X, dtype=tf.float32)

    logits = tf.layers.dense(states, OUTPUT_SIZE)

    return logits