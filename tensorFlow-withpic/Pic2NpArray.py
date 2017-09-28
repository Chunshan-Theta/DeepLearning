#coding:utf-8
import matplotlib.pyplot as plt
# 手書き数字データを描画する関数
import numpy as np
import Image


 
def numpy_digit_draw(data,X_size,Y_size,alert = True):    
    
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(X_size),range(Y_size))
    Z = data.reshape(X_size,Y_size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                     # flip vertical
    plt.xlim(0,(X_size-1)	)		  # 畫布大小
    plt.ylim(0,(Y_size-1))
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off",labelleft="off")
    plt.show()
    if alert: print len(data),data


def ResizeAndConToNumpyArray(pic,X_size,Y_size):
    gray = pic.convert('L')
    gray_rz = gray.resize((X_size,Y_size), Image.ANTIALIAS)
    np_gray= np.array(gray_rz,float)
    np_gray = np_gray.ravel()
    np_gray /= 255
    return np_gray

'''
using:

#open a pic
data = Image.open('SourceImg/11.jpg')

# Resize pic to 56x56
NData = ResizeAndConToNumpyArray(data,56,56)

# Draw pic, pic's size = 56x56
numpy_digit_draw(NData,56,56) 

'''

