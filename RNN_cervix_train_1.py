import tensorflow as tf
import numpy as np

EPOCHS = 12000 #학습할 에포크의 수를 설정
STEP = 3000
Num_hidden = 150 # 하나의 은닉층 내부에 배치할 재귀노드 수
Learning_rate = 0.005 #학습률
num_layers = 3 #전체 은닉 cell 수
dropout = 0.9 # 드롭아웃비율
data_file_name = 'RNN_cervix_data2.csv' #불러들일 파일 이름
data_length = 6 #목적변수 제외한 속성의 수

data = np.genfromtxt(data_file_name, delimiter=',') #데이터파일을 numpy 로딩
data_train =data[:,0:data_length]
data_label =data[:,data_length]
#print(data_label)
temp = data_label.shape
data_label = data_label.reshape(temp[0], 1)
#print(data_label)

x_ = tf.placeholder(tf.float32, [None, data_train.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 1])

#아래꺼로 하면 오류남. 아마 MultiRNNCell 때문
#cell = tf.contrib.rnn.LSTMCell(Num_hidden) #셀 성격과 노드 수 설정
#cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout) #드롭아웃 적용
#cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True) #다층 재귀층 설정

#lstm = tf.contrib.rnn.LSTMCell(Num_hidden) #셀 성격과 노드 수 설정

def lstm_cell():
   lstm = tf.contrib.rnn.LSTMCell(Num_hidden)
   return lstm
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])



outputs, states = tf.contrib.rnn.static_rnn(cell, [x_], dtype=tf.float32)
outputs = outputs[-1]

W= tf.Variable(tf.truncated_normal([Num_hidden, int(y_.get_shape()[1])]))
b= tf.Variable(tf.constant(0.1, shape=[y_.get_shape()[1]]))
y = tf.tanh(tf.matmul(outputs, W) + b)

cost = tf.reduce_mean(tf.square((y*100)-y_)) #cost 정의
train_p = tf.train.AdamOptimizer(learning_rate=Learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print("Training Start!!")
print("============================================================")


for i in range(EPOCHS) :
    sess.run(train_p, feed_dict ={x_:data_train, y_:data_label})
    if i % STEP ==0:
        c = sess.run(cost, feed_dict= {x_:data_train, y_:data_label})
        print('Cost for training data:', c)
saver = tf.train.Saver()
saver.save(sess, './'+'RNN_cervix')

print("Training finished & Saved")
