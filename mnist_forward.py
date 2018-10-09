import tensorflow as tf

INPUT_SIZE = 784
FULLY_NODE1 = 1600
FULLY_NODE2 = 800
FULLY_NODE3 = 100
OUTPUT_SIZE = 10
INITIALIZER_FULLY = tf.contrib.layers.xavier_initializer()


def W_variable(name, shape, wd=None):
    var = tf.get_variable(name, shape, initializer=INITIALIZER_FULLY)
    if not wd is None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(wd)(var))
    return var


def B_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def bn(x, beta_name, gamma_name, name="bn"):
    axes = [d for d in range(len(x.get_shape()))]
    beta = tf.get_variable(beta_name, shape=[], initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable(gamma_name, shape=[], initializer=tf.constant_initializer(1.0))
    x_mean, x_variance = tf.nn.moments(x, axes)
    y = tf.nn.batch_normalization(x, x_mean, x_variance, beta, gamma, 1e-10, name)
    return y


def forward(x, regularizer):
    with tf.name_scope('Fully_Connection1'):
        w1 = W_variable('f_w1', [INPUT_SIZE, FULLY_NODE1], regularizer)
        b1 = B_variable('f_b1', [FULLY_NODE1])
        f1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        d1 = tf.nn.dropout(f1, 1.0)
    with tf.name_scope('Fully_Connection2'):
        w2 = W_variable('f_w2', [FULLY_NODE1, FULLY_NODE2], regularizer)
        b2 = B_variable('f_b2', [FULLY_NODE2])
        f2 = tf.nn.relu(tf.matmul(d1, w2) + b2)
        d2 = tf.nn.dropout(f2, 0.5)
    with tf.name_scope('Fully_Connection3'):
        w3 = W_variable('f_w3', [FULLY_NODE2, FULLY_NODE3], regularizer)
        b3 = B_variable('f_b3', [FULLY_NODE3])
        f3 = tf.nn.relu(tf.matmul(d2, w3) + b3)
        d3 = bn(f3, 'beta3', 'gamma3', 'batch_normalization3')
        # d3 = tf.nn.dropout(f3, 0.5)
    with tf.name_scope('SoftMax'):
        sw = W_variable('s_w', [FULLY_NODE3, OUTPUT_SIZE])
        sb = B_variable('s_b', [OUTPUT_SIZE])
        prediction = tf.nn.softmax(tf.matmul(d3, sw) + sb)
    return prediction
