import tensorflow as tf
import numpy as np


input_data=[
               [[1,0,1],
                [0,2,1],
                [1,1,0]],

               [[2,0,2],
                [0,1,0],
                [1,0,0]],

               [[1,1,1],
                [2,2,0],
                [1,1,1]],

               [[1,1,2],
                [1,0,1],
                [0,2,2]]

            ]
weights_data=[
              [[[ 1, 0, 1],
                [-1, 1, 0],
                [ 0,-1, 0]],
               [[-1, 0, 1],
                [ 0, 0, 1],
                [ 1, 1, 1]],
               [[ 0, 1, 1],
                [ 2, 0, 1],
                [ 1, 2, 1]],
               [[ 1, 1, 1],
                [ 0, 2, 1],
                [ 1, 0, 1]]],

              [[[ 1, 0, 2],
                [-2, 1, 1],
                [ 1,-1, 0]],
               [[-1, 0, 1],
                [-1, 2, 1],
                [ 1, 1, 1]],
               [[ 0, 0, 0],
                [ 2, 2, 1],
                [ 1,-1, 1]],
               [[ 2, 1, 1],
                [ 0,-1, 1],
                [ 1, 1, 1]]]
           ]



def tf_conv2d_transpose(input, weights):
    # input_shape=[n,height,width,channel]
    #print(input.shape)
    input_shape = input.get_shape().as_list()
   # print(input_shape)
    # weights shape=[height,width,out_c,in_c]
    weights_shape = weights.get_shape().as_list()
    output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, weights_shape[2]]

    print("output_shape:", output_shape)

    deconv = tf.nn.conv2d_transpose(input, weights, output_shape=output_shape,
                                    strides=[1, 2, 2, 1], padding='SAME')
    return deconv


def main():
    weights_np = np.asarray(weights_data, np.float32)
    # 将输入的每个卷积核旋转180°
    weights_np = np.rot90(weights_np, 2, (2, 3))
    #print(weights_np)
    const_input = tf.constant(input_data, tf.float32)
    const_weights = tf.constant(weights_np, tf.float32)

    input = tf.Variable(const_input, name="input")
    # [c,h,w]------>[h,w,c]
    input = tf.transpose(input, perm=(1, 2, 0))
    # [h,w,c]------>[n,h,w,c]
    input = tf.expand_dims(input, 0)

    # weights shape=[out_c,in_c,h,w]
    weights = tf.Variable(const_weights, name="weights")
    # [out_c,in_c,h,w]------>[h,w,out_c,in_c]
    weights = tf.transpose(weights, perm=(2, 3, 0, 1))

    # 执行tensorflow的反卷积
    deconv = tf_conv2d_transpose(input, weights)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    deconv_val = sess.run(deconv)

    hwc = deconv_val[0]
    # print(hwc.shape)
    print(deconv.get_shape())
    cont = tf.transpose(deconv, [0,3,1,2])
    print(sess.run(cont))
if __name__ == '__main__':
    main()
