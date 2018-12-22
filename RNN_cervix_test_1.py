
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf


EPOCHS = 36000
STEP = 3000
Num_hidden = 150
Learning_rate=0.005
num_layers = 3
dropout =0.9
data_input =np.array([[0,4,2,0,3,0]])

data_file_name ='RNN_cervix_data2.csv'
data_length =6

data = np.genfromtxt(data_file_name, delimiter=',')
data_train =data[:,0:data_length]
data_label =data[:,data_length]
temp = data_label.shape
data_label = data_label.reshape(temp[0], 1)

x_ = tf.placeholder(tf.float32, [None, data_train.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 1])

def lstm_cell():
   lstm = tf.contrib.rnn.LSTMCell(Num_hidden)
   return lstm
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])


outputs, states = tf.contrib.rnn.static_rnn(cell, [x_], dtype=tf.float32)
outputs = outputs[-1]

W= tf.Variable(tf.truncated_normal([Num_hidden, int(y_.get_shape()[1])]))
b= tf.Variable(tf.constant(0.1, shape=[y_.get_shape()[1]]))
y = tf.tanh(tf.matmul(outputs, W) + b)

cost = tf.reduce_mean(tf.square((y * 100) - y_))
train_p = tf.train.AdamOptimizer(learning_rate=Learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


saver = tf.train.Saver()
saver.restore(sess, 'RNN_cervix')

c = sess.run(y*100, feed_dict={x_:data_input})

print('Expected survival :')
print(c)



# In[ ]:
