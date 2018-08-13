

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


'''
    一张图片的反卷积
'''

im = Image.open('timg.jpg')
images = np.asarray(im)
print(images.shape)

images = np.reshape(images,[1,750,500,3])

img = tf.Variable(images,dtype=tf.float32)
# kernel = tf.get_variable(name='a',shape=[3, 3, 3, 3], dtype=tf.float32,
#                                   initializer=tf.contrib.layers.xavier_initializer())

# 卷积核
kernel = tf.get_variable(name='a',shape=[3, 3, 3, 64], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())

# 反卷积核
kernel2 = tf.get_variable(name='a1',shape=[3, 3, 64, 64], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())

#tf.nn.conv2d(input=input_op, filter=weights, strides=[1, dh, dw, 1], padding="SAME")

# 卷积
conv1 = tf.nn.conv2d(input=img, filter=kernel,strides=[1, 1, 1, 1], padding="SAME")
# print(conv1)
# 池化
pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

shape_ = pool.get_shape().as_list()
print(shape_) #[1, 375, 250, 64]
output_shape = [shape_[0], shape_[1] * 2, shape_[2] * 2, shape_[3]]

print('pool:',pool.get_shape())
# 反卷积操作
conts = tf.nn.conv2d_transpose(pool,kernel2,output_shape,strides=[1,2,2,1],padding='SAME')

# print(conv1.get_shape())



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    conv1_convert = sess.run(tf.transpose(conts, [0, 3, 1, 2]))

    print(conv1_convert.shape)

    fig6, ax6 = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
    plt.title('Pool2 32x7x7')
    for i in range(8):
        for j in range(8):
            ax6[i][j].imshow(conv1_convert[0][(i + 1) * j])

    plt.show()


