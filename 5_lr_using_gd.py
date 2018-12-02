# 使用梯度下降的优化方法快速解决线性回归问题

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

point_num = 100
vectors = []

for i in range(point_num):
    x1 = np.random.normal(0, 0.67)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

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