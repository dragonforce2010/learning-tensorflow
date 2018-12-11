# 使用梯度下降的优化方法快速解决线性回归问题
import os
# silence warning: tensorflow using SSE4.1, SSE4.2, and AVX
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf 

x_data = np.random.normal(0, 0.67, 100)
y_data = 0.1 * x_data + 0.2 + np.random.normal(0, 0.04, 100)

plt.plot(x_data, y_data, 'r*', label='original data')
plt.title('Linear Regression using gradient descent')
plt.legend()
plt.show()

# 使用tensorflow构建线性回归模型
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - y_data))

# 使用梯度下降的优化器对loss进行优化, 设置学习效率为0.1
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 创建会话
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)

    for step in range(20):
        # 在每一次训练迭代中进行优化
        session.run(train)
        # 打印当前步的损失，权重和偏差
        print("Step={}, Loss={}, [weight={}, bias={}]".format(step, session.run(loss), session.run(w), session.run(b)))

        plt.plot(x_data, y_data, 'r*', label="Original data")
        plt.title('Linear Regression using Gradient Descent')
        plt.plot(x_data, session.run(w) * x_data + session.run(b), label='Fitted line')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()