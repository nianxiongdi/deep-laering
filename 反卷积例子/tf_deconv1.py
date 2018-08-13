'''
一 反卷积实例
'''
import tensorflow as tf
import numpy as np

# 模拟数据
img = tf.Variable(tf.constant([[[-2,-2],
                                [-2,-2]]
                               ],tf.float32),tf.float32)
img=tf.transpose(img,perm=(1,2,0))
img = tf.expand_dims(img, 0)
kernel = tf.Variable(tf.constant([[[[-2,-1],
                                   [0,1]]]
                                  ],tf.float32),tf.float32)
print(kernel.get_shape())
kernel = tf.transpose(kernel, perm=(2, 3, 0, 1))

print(img.get_shape())

# 在进行反卷积操作
contv = tf.nn.conv2d_transpose(img, kernel, [1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('kernel:\n', sess.run(kernel))

    # print('contv:\n', sess.run(contv).shape)
    cont = tf.transpose(contv, [0,3,1,2])
    print( sess.run(cont) )