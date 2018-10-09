import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn_forward
import mnist_cnn_backward

TEST_INTERVAL_SECS = 20
IMAGE_SIZE = 28


def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE])
        x_ = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        y_ = tf.placeholder(tf.float32, [None, mnist_cnn_forward.OUTPUT_SIZE])
        y = mnist_cnn_forward.forward(x_)

        ema = tf.train.ExponentialMovingAverage(mnist_cnn_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_cnn_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images[:3000], y_: mnist.test.labels[:3000]})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
