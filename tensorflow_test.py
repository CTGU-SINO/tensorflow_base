import tensorflow as tf


def compute(x):
    # sigmoid = tf.div(1.0, tf.add(1.0, tf.exp(tf.negative(x))))
    y = tf.div(9.0, tf.add(9.0, tf.exp(tf.negative(x))))
    return tf.where(tf.less_equal(y, 0.9), x=tf.zeros_like(y), y=tf.ones_like(y))       # 阈值为 0.9


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 2], name='X')
    y_ = compute(x)
    result = sess.run([y_], feed_dict={x: [[0.1, 0.2], [-0.1, 0.1]]})
    print(result)