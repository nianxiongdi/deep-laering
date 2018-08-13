

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


'''
    一张图片的卷积
'''

im = Image.open('timg.jpg')
images = np.asarray(im)
print(images.shape)

images = np.reshape(images,[1,750,500,3])

img = tf.Variable(images,dtype=tf.float32)
# kernel = tf.get_variable(name='a',shape=[3, 3, 3, 3], dtype=tf.float32,
#                                   initializer=tf.contrib.layers.xavier_initializer())

kernel = tf.get_variable(name='a',shape=[3, 3, 3, 64], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())

#tf.nn.conv2d(input=input_op, filter=weights, strides=[1, dh, dw, 1], padding="SAME")

conv1 = tf.nn.conv2d(input=img, filter=kernel,strides=[1, 1, 1, 1], padding="SAME")

print(conv1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    '''
    # print(sess.run(kernel).shape)
    # print(sess.run(conv1).shape)
    mat = sess.run(conv1)
    print(mat)
    mat = np.squeeze(mat, axis=(0))
    # print( np.squeeze(mat, axis=(0)))
    plt.imshow(mat)
    plt.show()
    '''
    print('conv1:' , conv1)

    conv1_convert = sess.run(tf.transpose(conv1, [3, 0, 1, 2]))
    print(conv1_convert.shape)

    fig6,ax6 = plt.subplots(nrows=8, ncols=8, figsize = (8, 8))
    plt.title('conv1')
    for i in range(8):
        for j in range(8):
            # print(type(conv1_convert[i][0]))
            '''
             fig6,ax6 = plt.subplots(nrows=1, ncols=32, figsize = (32, 1))  单行图片的imshow
             plt.title('Pool2 32x7x7')
             for i in range(32):
                 ax6[i].imshow(pool2_transpose[i][0])
            '''
            ax6[i][j].imshow(conv1_convert[(i+1)*j][0])

    plt.show()
