#coding:utf-8
import matplotlib.pyplot as plt
# 手書き数字データを描画する関数
import numpy as np
import Image
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


 
def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,27)				  # 畫布大小
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off",labelleft="off")
    #plt.show()
    

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
Source = mnist.test.images[0]
#print type(Source)
#print len(Source)
print Source

data = Image.open('SourceImg/11.jpg')
gray = data.convert('L')
gray_28_28 = gray.resize((28,28), Image.ANTIALIAS)
np_gray= np.array(gray_28_28,float)
np_gray = np_gray.ravel()
np_gray /= 255
#np_gray -= 1
#np_gray *= -1
#x_image = tf.reshape(data, [-1, 28, 28, 1])
#x_image = x_image.convert('L')
#x_image = x_image.ravel()

#print type(np_gray)
#print len(np_gray)
print np_gray

draw_digit(Source) 

