import tensorflow as tf

# 构造graph的结构，用一个线性方程的例子
# y = w*x + b
w = tf.Variable(2.0, dtype=tf.float32, name="weight") # 权重
b = tf.Variable(1.0, dtype=tf.float32, name="bias") # 偏差
x = tf.placeholder(dtype=tf.float32, name="input") # 输入

with tf.name_scope('output'):
    y = w * x + b

path = './log' # 定义log目录路径

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x: 3.0})
    print('y = {}'.format(result))


