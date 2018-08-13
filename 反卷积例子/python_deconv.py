# 根据输入map([h,w])和卷积核([k,k]),计算卷积后的feature map
import numpy as np


def compute_conv(fm, kernel):
    [h, w] = fm.shape
    [k, _] = kernel.shape
    r = int(k / 2)
    # 定义边界填充0后的map
    padding_fm = np.zeros([h + 2, w + 2], np.float32)
    # 保存计算结果
    rs = np.zeros([h, w], np.float32)
    # 将输入在指定该区域赋值，即除了4个边界后，剩下的区域
    padding_fm[1:h + 1, 1:w + 1] = fm
    # 对每个点为中心的区域遍历
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            # 取出当前点为中心的k*k区域
            roi = padding_fm[i - r:i + r + 1, j - r:j + r + 1]
            # 计算当前点的卷积,对k*k个点点乘后求和
            rs[i - 1][j - 1] = np.sum(roi * kernel)

    return rs


# 填充0
def fill_zeros(input):
    [c, h, w] = input.shape
    rs = np.zeros([c, h * 2 + 1, w * 2 + 1], np.float32)

    for i in range(c):
        for j in range(h):
            for k in range(w):
                rs[i, 2 * j + 1, 2 * k + 1] = input[i, j, k]
    return rs


def my_deconv(input, weights):
    # weights shape=[out_c,in_c,h,w]
    [out_c, in_c, h, w] = weights.shape
    print(weights.shape)
    out_h = h * 2
    out_w = w * 2
    rs = []
    for i in range(out_c):
        w = weights[i]
        tmp = np.zeros([out_h, out_w], np.float32)
        for j in range(in_c):
            conv = compute_conv(input[j], w[j])
            # 注意裁剪，最后一行和最后一列去掉
            tmp = tmp + conv[0:out_h, 0:out_w]
        rs.append(tmp)

    return rs

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

def main():
    input = np.asarray(input_data, np.float32)
    input = fill_zeros(input)
    print('full_input:' , input.shape)
    weights = np.asarray(weights_data, np.float32)
    deconv = my_deconv(input, weights)

    print(np.asarray(deconv))


if __name__ == '__main__':
    main()
