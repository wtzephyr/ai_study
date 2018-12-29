
# coding: utf-8

# In[1]:




import numpy as np
import tensorflow as tf

EPOCHS = 20000
STEP = 2000
Num_hidden = 2
Learning_rate=0.005


data_train = np.array([[4,3,1,2],[2,5,3,1],[3,4,2,3],[6,3,2,2],
                       [4,1,2,0],[2,2,1,1],[1,5,2,2],[2,4,3,1],[5,8,1,2]])

data_label = np.array([[6],[9],[6],[9],[7],[4],[6],[8],[12]])

x_ = tf.placeholder(tf.float32, [None, data_train.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.LSTMCell(Num_hidden)

outputs, states = tf.contrib.rnn.static_rnn(cell, [x_], dtype=tf.float32)
outputs = outputs[-1]

W= tf.Variable(tf.truncated_normal([Num_hidden, int(y_.get_shape()[1])]))
b= tf.Variable(tf.constant(0.1, shape=[y_.get_shape()[1]]))
y = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.square((y ) - y_))
train_p = tf.train.AdamOptimizer(learning_rate=Learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("Training Start!!")
print("==============================================")
   
for i in range(EPOCHS):
    sess.run(train_p, feed_dict={x_:data_train, y_:data_label})
    if i % STEP == 0:
        c = sess.run(cost, feed_dict={x_:data_train, y_:data_label})
        print('Cost for training data :', c)
a = sess.run(y, feed_dict={x_:[[5,1,2,1]]})
print(a)


# In[ ]:



