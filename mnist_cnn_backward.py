import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn_forward
import os

IMAGE_SIZE = 28
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.9
STEPS = 1000000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./cnn_model/"
CHECK_POINT = './cnn'
MODEL_NAME = "mnist_model"


def backward(mnist):
    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE*IMAGE_SIZE])
        x_ = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        y_ = tf.placeholder(tf.float32, [None, mnist_cnn_forward.OUTPUT_SIZE])
        y = mnist_cnn_forward.forward(x_)
        global_step = tf.Variable(0, trainable=False)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
        # loss = tf.reduce_mean(ce, name="loss")
        cem = tf.reduce_mean(ce)
        loss = cem + tf.reduce_sum(tf.get_collection('losses'))

        tf.summary.scalar('loss', loss)

        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True)

        tf.summary.scalar('learning_rate', learning_rate)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # 为可训练的变量添加直方图
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            if not os.path.exists(CHECK_POINT):
                os.mkdir(CHECK_POINT)

            summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(CHECK_POINT+'/summary', graph)

            for i in range(STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 20 == 0:
                    summary_str = sess.run(summary, feed_dict={x: xs, y_: ys})
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()
                if i % 500 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    print(mnist.train.num_examples / BATCH_SIZE)
    backward(mnist)


if __name__ == '__main__':
    main()
