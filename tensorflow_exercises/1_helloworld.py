# -*- coding: utf-8 -*-
import os
# silence warning: tensorflow using SSE4.1, SSE4.2, and AVX
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 引入tensorflow库
import tensorflow as tf

# 创建一个常量操作
hw = tf.constant('Hello World! I love tensorflow!')

# 启动tensorflow的一个回话
session = tf.Session()

#运行graph （计算图）
print(session.run(hw))

# 关闭tensorflow的回话
session.close()