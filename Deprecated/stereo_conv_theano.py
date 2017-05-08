import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import skimage as ski
from skimage.transform import resize
from skimage import io, data
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.signal.conv import conv2d
#from theano.tensor.nnet import conv2d


def image_preprocessor(pic_height_pix):
	#Load images and normalize their sizes
	leftim = io.imread('Data/Pictures_Data/left.jpeg')
	rightim = io.imread('Data/Pictures_Data/right.jpeg')

	#The aspect ratio of the original images
	aspect_ratio = np.shape(leftim[0,:,1])[0]/np.shape(leftim[:,0,1])[0]
	pic_width_pix = int(np.floor(pic_height_pix * aspect_ratio))

	#Normalize the sizes of the images
	leftim_n = \
	ski.transform.resize(leftim,(pic_height_pix,pic_width_pix))
	rightim_n = \
	ski.transform.resize(rightim,(pic_height_pix,pic_width_pix))

	return leftim_n, rightim_n, aspect_ratio, pic_width_pix


def filter_maker(leftim_n, pic_height_pix, pic_width_pix,filter_size):
	#Makes bank of filters for convolution
	filter_spacer = int(np.floor((filter_size )/2))
	#the number of filters that will fit vertically
	filter_number_h = int(np.floor(pic_height_pix/filter_size))
	#the number of filters that will fit horizontally
	filter_number_w =  int(np.floor(pic_width_pix/filter_size))
	filter_number = filter_number_h*filter_number_w
	#Create collection of filters in filter bank, each will generate a feature map
	filter_bank = []
	#parameter to keep track of filter number in for loop
	#this clearly needs to be improved upon
	for i in range(filter_number_h):
		for j in range(filter_number_w):
			filter_bank.append(leftim_n[i*filter_size:(i+1)*filter_size,\
				j*filter_size:(j+1)*filter_size])

	filter_bank = np.array(filter_bank)
	return filter_bank, filter_number


def main(debug = True):
	#The length in pixels of the height the images will be adjusted to
	pic_height_pix = 1000
	#Filters for convolution will be of size filter_size x filter_size.
	#Choose a number that will divide both pic_height_pix and pic_width_pix
	filter_size = 25

	#Resize images
	leftim_n, rightim_n, aspect_ratio, \
	pic_width_pix = image_preprocessor(pic_height_pix)

	#Make the filter bank
	filter_bank, filter_number= \
	filter_maker(leftim_n, pic_height_pix, pic_width_pix,filter_size)

	#Cast all np arrays to Theano tensors. We also separate into RGB values.
	rightim_n_R = T.as_tensor_variable(rightim_n[:,:,0], ndim = 2)
	filter_bank_R = T.as_tensor_variable(filter_bank[:,:,:,0], ndim =3)
	rightim_n_G = T.as_tensor_variable(rightim_n[:,:,1], ndim = 2)
	filter_bank_G = T.as_tensor_variable(filter_bank[:,:,:,1], ndim =3)
	rightim_n_B = T.as_tensor_variable(rightim_n[:,:,2], ndim = 2)
	filter_bank_B = T.as_tensor_variable(filter_bank[:,:,:,2], ndim =3)

	X = T.matrix('X')
	Y = T.tensor3('Y')

	con_inp = X.reshape(pic_height_pix,pic_width_pix)
	filt_inp = Y.reshape(filter_number,filter_size,filter_size)
	#Convolve each channel
	conv_out_R = conv2d(
        input=con_inp,
        filters=filt_inp,
        filter_shape=(filter_number,filter_size,filter_size),
        image_shape=(1,pic_height_pix,pic_width_pix)
        )

	# conv_out_G = conv2d(
 #        input=rightim_n_G,
 #        filters=filter_bank_G,
 #        filter_shape=(filter_number,filter_size,filter_size),
 #        image_shape=(pic_height_pix,pic_width_pix)
 #        )

	# conv_out_B = conv2d(
 #        input=rightim_n_B,
 #        filters=filter_bank_B,
 #        filter_shape=(filter_number,filter_size,filter_size),
 #        image_shape=(pic_height_pix,pic_width_pix)
 #        )

	
	conv_out_f = theano.function(inputs = [], outputs = conv_out_R,
	givens = {
		X: rightim_n[:,:,0],
		Y: filter_bank[:,:,:,0]
	})
	conv_out = conv_out_f()
	# if debug == True:
	# 	print("aspect_ratio: %s" % aspect_ratio)
	# 	print("filter_number: %d" % filter_number)
	# 	print("filter_size: %d" % filter_size)
	# print(np.shape(rightim_n[:,:,0]))
	# print(np.shape(filter_bank[:,:,:,0]))
	# print(np.shape(filter_bank))
	# print(np.shape(filter_bank[2119]))
	# plt.imshow(filter_bank[2119])
	# plt.imshow(leftim_n)
	# plt.imshow(conv_out_R)
	#plt.show()
	#print(np.shape(conv_out))
	print(conv_out_R.shape.eval())
if __name__ == "__main__":
	main()