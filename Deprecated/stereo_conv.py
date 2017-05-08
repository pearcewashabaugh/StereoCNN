import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import skimage as ski
from skimage.transform import resize
from skimage import io, data
from scipy.signal import convolve2d
from PIL import Image


def image_preprocessor(pic_height_pix):
	#Load images and normalize their sizes
	leftim = io.imread('Data/Pictures_Data/left.jpeg')
	rightim = io.imread('Data/Pictures_Data/right.jpeg')

	#The aspect ratio of the original images
	aspect_ratio = np.shape(leftim[0,:,1])[0]/np.shape(leftim[:,0,1])[0]
	pic_width_pix = int(np.floor(pic_height_pix * aspect_ratio))
	pic_dim = [pic_height_pix,pic_width_pix]
	#Normalize the sizes of the images
	leftim_n = \
	ski.transform.resize(leftim,(pic_height_pix,pic_width_pix))
	rightim_n = \
	ski.transform.resize(rightim,(pic_height_pix,pic_width_pix))

	return leftim_n, rightim_n, pic_dim


def filter_maker(leftim_n, pic_dim,filter_size):
	#Makes bank of filters for convolution
	filter_spacer = int(np.floor((filter_size )/2))
	#the number of filters that will fit vertically
	filter_number_h = int(np.floor(pic_dim[0]/filter_size))
	#the number of filters that will fit horizontally
	filter_number_w =  int(np.floor(pic_dim[1]/filter_size))
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

def image_convolver(rightim_n,filter_bank,filter_number,pic_dim,filter_size):
	#Outputs feature maps
	feature_maps = np.zeros((filter_number,pic_dim[0]+filter_size-1,pic_dim[1]+filter_size-1,3))
	print("filter_number: %d" % filter_number)

	# for i in range(filter_number):
	for i in range(1):
		print("Current convolution iteration: %d" % i)
		conv_out_R = convolve2d(rightim_n[:,:,0],filter_bank[i,:,:,0])
		conv_out_G = convolve2d(rightim_n[:,:,1],filter_bank[i,:,:,1])
		conv_out_B = convolve2d(rightim_n[:,:,2],filter_bank[i,:,:,2])

		feature_maps[i,:,:,0] = conv_out_R 
		feature_maps[i,:,:,1] = conv_out_G
		feature_maps[i,:,:,2] = conv_out_B
	return feature_maps

###############################################################################
#EVALUATE
###############################################################################

def main(debug = True):
	#The length in pixels of the height the images will be adjusted to
	pic_height_pix = 600

	#Filters for convolution will be of size filter_size x filter_size.
	#Choose a number that will divide both pic_height_pix and pic_width_pix
	filter_size = 25

	#Resize images
	leftim_n, rightim_n, pic_dim = image_preprocessor(pic_height_pix)

	#Make the filter bank
	filter_bank, filter_number= \
	filter_maker(leftim_n, pic_dim,filter_size)
	
	feature_maps = image_convolver(rightim_n,filter_bank,filter_number,
		pic_dim,filter_size)

	print(np.shape(rightim_n))
	print(np.shape(feature_maps[0,:,:,:]))
	#plt.imshow(feature_maps[0,:,:,:])
	plt.imshow(feature_maps[0,:,:,0])
	#plt.imshow(conv_out_R)
	plt.show()

	# if debug == True:
	# 	print("aspect_ratio: %s" % aspect_ratio)
	# 	print("filter_number: %d" % filter_number)
	# 	print("filter_size: %d" % filter_size)
if __name__ == "__main__":
	main()

# r, g, and b are 512x512 float arrays with values >= 0 and < 1.

# rgbArray = np.zeros((512,512,3), 'uint8')
# rgbArray[..., 0] = r*256
# rgbArray[..., 1] = g*256
# rgbArray[..., 2] = b*256
# img = Image.fromarray(rgbArray)
# img.save('myimg.jpeg'