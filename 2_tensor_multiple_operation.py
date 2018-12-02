# -*- encoding: UTF-8 -*-
import tensorflow as tf
import numpy as np

# 注意这里必须使用dtype定义数据类型为int32, float32等tf.matmul支持的数据类型，int64是不支持的
c1 = tf.constant(np.arange(24).reshape((6, 4)), dtype=tf.int32)
c2 = tf.constant(np.arange(12).reshape((4, 3)), dtype=tf.int32)

# tf.matmul运算是举证乘法，参数中的矩阵a的列数必须等于参数中的矩阵b的行数，同时矩阵都必须是秩 >= 2的
# tf.multiply运算时矩阵级别的点乘，两个矩阵必须是同型矩阵
multiple = tf.matmul(c1, c2)
session = tf.Session()
result = session.run(multiple)
print('Output 1:', result)
session.close()

# 第二种方法管理session
with tf.Session() as sess:
    print('Output 2:', sess.run(multiple))
