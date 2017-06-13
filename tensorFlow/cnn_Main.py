#!/usr/bin/env python
# -*- coding: utf-8 -*-
##https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf18_CNN3/full_code.py
##http://darren1231.pixnet.net/blog/post/332753859-tensorflow
##https://github.com/Chunshan-Theta/DeepLearning/blob/master/tensorFlow/cnn_Main.py
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# draw pic from dataset  
##using:draw_digit(mnist.data[5])
import matplotlib.pyplot as plt
import numpy as np
def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

    plt.show()


#prediction a pic
def GiveAnswer(pic):
    global prediction
    pic = np.array([pic])
    prediction_answer = sess.run(prediction,feed_dict={xs: pic, keep_prob: 1})

    #print("y_pre[0]\n",y_pre[0])
    print('I think this is ',list(prediction_answer[0]).index(max(prediction_answer[0])),' with',max(prediction_answer[0])*100,'% confident')
    

#compute accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})

    #show detail of computation
    #print("y_pre[0]\n",y_pre[0])
    #print('guess:',list(y_pre[0]).index(max(y_pre[0])))
    #print("Ans:",list(v_ys[0]).index(1))

    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#隨機變量
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1  !! the note from Document
	
	# x 是指圖片數值
	# W 是指weight
	# strides 是指步長，需輸入要是四個維度，而第一個維度與最後一個維度必須為1。第二維度是指X方向，第三維度則是Y方向
	# padding方式，padding='SAME' or 'VALID'
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
	# 用來避免strides過大導致丟失太多特徵
	# 這邊可以選用 tf.nn.avg_pool 或是 tf.nn.max_pool，官方範例是用 tf.nn.max_pool
	# strides 是指步長，需輸入要是四個維度，而第一個維度與最後一個維度必須為1。第二維度是指X方向，第三維度則是Y方向
	# strides 第二維度與第三維度設定2是為了減小圖像大小
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])   # -1是指放棄資料原有的所有維度，28,28則是新給維度，1則是指說資料只有一個數值(黑白)，若是彩色則為3(RGB)
										    # x_image.shape = [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32])      # patch 5x5, in size 1, out size 32
										    # 這邊用5x5是Mnist官方建議的參數，若是圖片較大可以向上調整
											# 1 是 image的輸入時的厚度,輸出則要求要是32的厚度 
b_conv1 = bias_variable([32])				# 設定為輸出的厚度
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
														  # tf.nn.relu() 將內容改成非線性資料
														 
h_pool1 = max_pool_2x2(h_conv1)                           # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) 				  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                           # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])                   # 讓資料厚度變成更大(1024)
b_fc1 = bias_variable([1024])                   
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])          # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
												          # 轉換為一維數值
														  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)			  # 避免過度學習
														  # keep_prob = tf.placeholder(tf.float32)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
#if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#    init = tf.initialize_all_variables()
#else:
#    init = tf.global_variables_initializer()
#sess.run(init)



GiveAnswer(mnist.test.images[3])
print("ans:",list(mnist.test.labels[3]).index(1))


for i in range(101):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 20 == 0:
	#if you get "Killed" message , you should upgrade limit of memory
        print(str(i),",",compute_accuracy(mnist.test.images,mnist.test.labels))	
	#print("ans",list(mnist.test.labels[0+i]).index(1))
	#draw_digit(mnist.test.images[0+i])

GiveAnswer(mnist.test.images[3])
print("ans:",list(mnist.test.labels[3]).index(1))        
draw_digit(mnist.test.images[3])
