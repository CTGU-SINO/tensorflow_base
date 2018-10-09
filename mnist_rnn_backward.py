import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_rnn_forward
import os

LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
TRAIN_EPOCHS = 5
BATCH_SIZE = 200
STEPS = 100000
MODEL_SAVE_PATH = "./rnn_model/"
MODEL_NAME = "mnist_rnn_model"


def backward(mnist):
    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, mnist_rnn_forward.N_STEP, mnist_rnn_forward.INPUT_SIZE], name='X')
        y_ = tf.placeholder(tf.float32, [None, mnist_rnn_forward.OUTPUT_SIZE], name='Y')
        y = mnist_rnn_forward._RNN(x)
        global_step = tf.Variable(0, trainable=False)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
        # loss = tf.reduce_mean(ce, name="loss")
        cem = tf.reduce_mean(ce)
        loss = cem + tf.reduce_sum(tf.get_collection('losses'))

        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            if not os.path.exists('./rnn'):
                os.mkdir('./rnn')

            tf.summary.merge_all()
            tf.summary.FileWriter("rnn/summary", graph)

            for i in range(STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                xs = xs.reshape((BATCH_SIZE, mnist_rnn_forward.N_STEP, mnist_rnn_forward.INPUT_SIZE))
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 1000 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    print(mnist.train.num_examples / BATCH_SIZE)
    backward(mnist)


if __name__ == '__main__':
    main()