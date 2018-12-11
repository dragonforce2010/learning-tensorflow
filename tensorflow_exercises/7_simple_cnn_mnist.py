import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# download the mnist data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# None 表示输入的张量的数量可以是任意长度
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255
input_x = tf.reshape(input_x, [-1, 28, 28, 1])
output_y = tf.placeholder(tf.int32, [None, 10])

# 从测试数据集中选取3000个手写数字的图片和对应标签
test_x = mnist.test.images[:3000] # 图片
test_y = mnist.test.labels[:3000] # 标签

# 构建神经网路
# 构建第一层卷积网路
conv1 = tf.layers.conv2d(
    inputs=input_x,         # 形状 [28, 28, 1]
    filters=32,             # 32个过滤器，输出的深度（depth）是32
    kernel_size=[5, 5],     # 过滤器的二维大小是（5 * 5）
    strides=1,              # 步长为1
    padding='same',         # same 表示输出的大小不变，因此需要在外围补2圈0
    activation=tf.nn.relu   # 激活函数是Relu
) # 最终输出是[28, 28, 32]

# 第一层池化 （亚采样）
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,            # shape [28, 28, 32]
    pool_size=[2, 2],        # the filter shape in 2d space is (2, 2)
    strides=2                # stride is 2
) # shape [14, 14, 32]

# The second conv layer 
conv2 = tf.layers.conv2d(
    inputs=pool1,            # shape [14, 14, 32]
    filters=64,              # the depth of the filter is 64
    kernel_size=[5, 5],      # the filter shape in 2d is (5, 5)
    strides=1,               # stride is 1
    padding='same',          # same means the output size the same
    activation=tf.nn.relu    # the ativation function is relu
) # shape [14, 14, 64]

# The second pooling layer
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,            # shape [14, 14, 64]
    pool_size=[2, 2],        
    strides=2
) # shape [7, 7, 64]

# flat
flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # shape [7 * 7 * 64, ]

# fully connected layer of 1024 neurons
dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)

# dropout
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# output layer, also fully connected layer, size is 10, since we want to recognize 10 digits
logits = tf.layers.dense(inputs=dropout, units=10)

# calculate cross entropy and then use softmax to calculate the probablity
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# use Adam optimzer to minimize the loss, learning rate is 0.001
train_optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

# calcuate the accuracy
# return (accuracy, update_op), will create two local variables
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1), 
    predictions=tf.argmax(logits, axis=1)
)

# create a session
with tf.Session() as sess:
    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    sess.run(init)

    # for visualization 
    train_loss_values = []
    train_accuracy_values = []

    for i in range(100):
        # fetch 50 samples from mnist train dataset
        batch = mnist.train.next_batch(50)
        train_loss, _ = sess.run(
            [loss, train_optimizer],
            {
                input_x: np.reshape(batch[0], [-1, 28, 28, 1]),
                output_y: batch[1]
            }
        )

        if i % 2 == 0:
            test_accuracy, _ = sess.run(accuracy, {
                input_x: np.reshape(test_x, [-1, 28, 28, 1]),
                output_y: test_y
            })
            print('Step=%d, Train loss=%.4f, [Test accuracy=%.2f]' % (i, train_loss, test_accuracy))

            # test: print 20 prediction values and real values for comparasion
            test_output = sess.run(logits, { input_x: np.reshape(test_x[:20], [-1, 28, 28, 1])})
            predicted_y = np.argmax(test_output, 1)
            print(predicted_y, 'Predicted numbers')
            print(np.argmax(test_y[:20], 1), 'Real values')    

            train_accuracy_values.append(test_accuracy)
            train_loss_values.append(train_loss)

    plt.plot(np.arange(len(train_accuracy_values)), train_accuracy_values, 'x-', label="training accuracy")
    plt.plot(np.arange(len(train_loss_values)), train_loss_values, '+-', label="training loss")
    plt.show()

          
        
