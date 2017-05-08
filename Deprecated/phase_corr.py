#Taken from 
#http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html#
#sphx-glr-auto-examples-transform-plot-register-translation-py

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

import skimage as ski
from skimage import io
from skimage.transform import resize

def image_preprocessor(pic_height_pix):
	#Load images and normalize their sizes
	leftim = io.imread('Data/Pictures_Data/left.jpeg')
	rightim = io.imread('Data/Pictures_Data/right.jpeg')

	#The aspect ratio of the original images
	aspect_ratio = np.shape(leftim[0,:,1])[0]/np.shape(leftim[:,0,1])[0]
	pic_width_pix = int(np.floor(pic_height_pix * aspect_ratio))
	pic_dim = [pic_height_pix,pic_width_pix]
	#Normalize the sizes of the images
	image = \
	ski.transform.resize(leftim,(pic_height_pix,pic_width_pix))
	offset_image = \
	ski.transform.resize(rightim,(pic_height_pix,pic_width_pix))

	return image, offset_image, pic_dim

image, offset_image, pic_dim = image_preprocessor(500)
# image = data.camera()
# shift = (-22.4, 13.32)
# # The shift corresponds to the pixel offset relative to the reference image
# offset_image = fourier_shift(np.fft.fftn(image), shift)
# offset_image = np.fft.ifftn(offset_image)
# print("Known offset (y, x): {}".format(shift))

# pixel precision first
shift, error, diffphase = register_translation(image, offset_image)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

plt.show()

print("Detected pixel offset (y, x): {}".format(shift))

# subpixel precision
shift, error, diffphase = register_translation(image, offset_image, 100)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Calculate the upsampled DFT, again to show what the algorithm is doing
# behind the scenes.  Constants correspond to calculated values in routine.
# See source code for details.
cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Supersampled XC sub-area")


plt.show()

print("Detected subpixel offset (y, x): {}".format(shift))