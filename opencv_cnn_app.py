import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import mnist_cnn_forward
import mnist_cnn_backward
import os

IMAGE_SIZE = 28

def restore_model(picPath):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE])
        x_ = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        y = mnist_cnn_forward.forward(x_)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_cnn_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_cnn_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                img = Image.open(picPath)
                reIm = img.resize((28, 28), Image.ANTIALIAS)
                im_arr = np.array(reIm.convert('L'))
                threshold = 50
                for i in range(28):
                    for j in range(28):
                        im_arr[i][j] = 255 - im_arr[i][j]
                        if im_arr[i][j] < threshold:
                            im_arr[i][j] = 0
                        else:
                            im_arr[i][j] = 255

                nm_arr = im_arr.reshape([1, 784])
                nm_arr = nm_arr.astype(np.float32)
                img_ready = np.multiply(nm_arr, 1.0 / 255.0).reshape(
                    (-1, IMAGE_SIZE*IMAGE_SIZE))
                preValue = sess.run(preValue, feed_dict={x: img_ready})
                # cv_img = cv2.imread(picPath)
                # cv2.imshow(str(preValue[0]), cv_img)
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def main():
    for i in os.listdir('pic'):
        result = restore_model(os.path.join('pic', i))
        print(os.path.join('pic', i), result)


if __name__ == '__main__':
    main()