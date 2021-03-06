import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.transform import resize
from skimage import io
from scipy.signal import convolve2d


###############################################################################
# Process original images to standard sizes

def image_preprocessor(pic_height_pix):
    # Load images and normalize their sizes
    leftim = io.imread('Data/Pictures_Data/left.jpeg')
    rightim = io.imread('Data/Pictures_Data/right.jpeg')

    # The aspect ratio of the original images
    aspect_ratio = np.shape(leftim[0, :, 1])[0] / np.shape(leftim[:, 0, 1])[0]
    pic_width_pix = int(np.floor(pic_height_pix * aspect_ratio))
    pic_dim = [pic_height_pix, pic_width_pix]
    # Normalize the sizes of the images
    leftim_n = \
        ski.transform.resize(leftim, (pic_height_pix, pic_width_pix))
    rightim_n = \
        ski.transform.resize(rightim, (pic_height_pix, pic_width_pix))

    return leftim_n, rightim_n, pic_dim


###############################################################################
#
class ConvPoolLayer(object):
    # This class will take two given pictures, divide one up into subpictures, each
    # of which will be used as a filter, and convolve each filter against the other image

    def __init__(self, im1, im2, pic_dim, filter_size, subpic_pad):
        self.im1 = im1
        self.im2 = im2
        self.pic_dim = pic_dim
        self.filter_size = filter_size
        self.subpic_pad = subpic_pad

        # Construct collection of filters by calling filter_maker
        self.filter_bank, self.im_bank, self.filter_number, self.filter_bank_L2, \
        self.im_bank_l2 = self.filter_maker

        # Convolve images by calling image_convolver()
        self.feature_maps = self.image_convolver()

        # Construct pooling layer by calling feature_pooler()
        self.pooled_features = self.feature_pooler()

    @property
    def filter_maker(self):
        # Makes bank of filters for convolution
        # filter_spacer = int(np.floor((self.filter_size) / 2))
        # the number of filters that will fit vertically
        filter_number_h = int(np.floor(self.pic_dim[0] / self.filter_size))
        # the number of filters that will fit horizontally
        filter_number_w = int(np.floor(self.pic_dim[1] / self.filter_size))
        filter_number = filter_number_h * filter_number_w
        # the length of a subpicture
        subpic_size = self.filter_size+2*self.subpic_pad
        # this is useful for the boundary case
        subpic_p_size = self.filter_size+self.subpic_pad
        # Create collection of filters in filter bank, each will generate a feature map
        filter_bank = []
        # The collection of pieces im2 that will be convolved against each filter
        im_bank = np.zeros((filter_number, subpic_size, subpic_size, 3))
        # parameter to keep track of filter number in for loop
        # this clearly needs to be improved upon

        for i in range(filter_number_h):
            for j in range(filter_number_w):
                filter_bank.append(self.im1[i * self.filter_size:(i+1) * self.filter_size,
                                   j * self.filter_size:(j+1) * self.filter_size])
                # (a,b) are the coordinates of the upper left point of the current filter
                a = i * self.filter_size
                b = j * self.filter_size
                # if the subpicture runs over the boundary:
                if (a < self.subpic_pad) & (b < self.subpic_pad):
                    im_bank[len(filter_bank)-1, (self.subpic_pad-a):, (self.subpic_pad-b):, :]\
                        = self.im2[0:subpic_p_size+a, 0:subpic_p_size+b]

                elif (a < self.subpic_pad) & (b > self.subpic_pad) & (b < (self.pic_dim[1]-self.subpic_pad-self.filter_size)):
                    im_bank[len(filter_bank)-1, (self.subpic_pad-a):, :, :]\
                        = self.im2[0:subpic_p_size+a, (b-self.subpic_pad):(subpic_p_size+b)]

                elif (a > self.subpic_pad) & (b < self.subpic_pad) & (a < (self.pic_dim[0]-self.subpic_pad-self.filter_size)):
                    im_bank[len(filter_bank) - 1, :, (self.subpic_pad-b):, :] \
                        = self.im2[(a - self.subpic_pad):subpic_p_size + a, 0:(subpic_p_size + b)]

                elif (a > self.subpic_pad) & (b > self.subpic_pad) & (a < (self.pic_dim[0]-self.subpic_pad-self.filter_size)) \
                & (b < (self.pic_dim[1]-self.subpic_pad-self.filter_size)):
                    im_bank[len(filter_bank) - 1, :, :, :] \
                        = self.im2[(a - self.subpic_pad):subpic_p_size + a, (b-self.subpic_pad):(subpic_p_size+b)]

                elif (a > (self.pic_dim[0]-self.subpic_pad)) & (b > (self.pic_dim[1]-self.subpic_pad)):
                    im_bank[len(filter_bank) - 1, :(a - (self.pic_dim[0]-self.subpic_pad)), :(b - (self.pic_dim[1]-self.subpic_pad)), :] \
                        = self.im2[(a - self.subpic_pad):, (b-self.subpic_pad):]

                elif (a > (self.pic_dim[0]-self.subpic_pad)) & (b < (self.pic_dim[1]-self.subpic_pad)):
                    im_bank[len(filter_bank) - 1, :(a - (self.pic_dim[0]-self.subpic_pad)), :(b - (self.pic_dim[1]-self.subpic_pad)), :] \
                        = self.im2[(a - self.subpic_pad):, (b-self.subpic_pad):(subpic_p_size+b)]

                elif (a < (self.pic_dim[0]-self.subpic_pad)) & (b > (self.pic_dim[1]-self.subpic_pad)):
                    im_bank[len(filter_bank) - 1, :(a - (self.pic_dim[0]-self.subpic_pad)), :(b - (self.pic_dim[1]-self.subpic_pad)), :] \
                        = self.im2[(a - self.subpic_pad):subpic_p_size + a, (b-self.subpic_pad):]

        filter_bank = np.asarray(filter_bank)
        print(len(im_bank))

        print("filter_number_h: %d" % filter_number_h)
        print("filter_number_w: %d" % filter_number_w)
        print("filter_number: %d" % filter_number)
        # print("im_number_h: %d" % int(round(filter_number_h * .5 + .00000001)))
        # print("im_number_w: %d" % int(round(filter_number_w * .5 + .00000001)))
        # print("im_number: % d" % (
        # int(round(filter_number_h * .5 + .00000001)) * int(round(filter_number_w * .5 + .00000001))))
        print(np.shape(filter_bank))
        im_bank = np.asarray(im_bank)
        print(np.shape(im_bank))
        # for i in range(filter_number):
        #     print(np.shape(im_bank[i]))



        # A list which contains the average L2 norm of every filter in filter_bank
        filter_bank_l2 = []
        im_bank_l2 = []
        for i in range(filter_number):
            filter_bank_l2.append(np.linalg.norm(filter_bank[i]) ** 2)
            im_bank_l2.append(np.linalg.norm(im_bank[i]) ** 2)
            # Normalize the size of the entries of the filter
            filter_bank[i] = filter_bank[i] / (self.filter_size ** 2)

        return filter_bank, im_bank, filter_number, filter_bank_l2, im_bank_l2

    def image_convolver(self):
        # Outputs feature maps
        feature_maps = np.zeros((self.filter_number, np.shape(self.im_bank[0])[0], np.shape(self.im_bank[0])[1], 3))
        print("filter_number: %d" % self.filter_number)

        # space = int(np.floor(float(self.filter_size)/2))
        space = int(np.floor(float(self.filter_size) / 2.0 + .00000001))
        # for i in range(filter_number):
        for i in range(self.filter_number):
            print("Current convolution iteration: %d" % i)

            feature_maps[i, :, :, 0] = convolve2d(self.im_bank[i, :, :, 0], self.filter_bank[i, :, :, 0], mode="same",
                                    boundary='fill', fillvalue=.2)
            feature_maps[i, :, :, 1] = convolve2d(self.im_bank[i, :, :, 1], self.filter_bank[i, :, :, 1], mode="same",
                                    boundary='fill', fillvalue=.2)
            feature_maps[i, :, :, 2] = convolve2d(self.im_bank[i, :, :, 2], self.filter_bank[i, :, :, 2], mode="same",
                                    boundary='fill', fillvalue=.2)

        return feature_maps

    def feature_pooler(self):
        # Pools results of feature_maps
        pooled_features = np.zeros((1, 1))
        return pooled_features


###############################################################################
# EVALUATE
###############################################################################

def main(plotter = 'basic'):
    # The length in pixels of the height the images will be adjusted to
    pic_height_pix = 200

    # Filters for convolution will be of size filter_size x filter_size.
    # Choose a number that will divide both pic_height_pix and pic_width_pix
    filter_size = 50
    # the number of pixels to pad each filter by to create each subpicture
    subpic_pad = 25

    # Resize images
    leftim_n, rightim_n, pic_dim= image_preprocessor(pic_height_pix)

    # Make Conv/Pool layer object
    layer_0 = ConvPoolLayer(leftim_n, rightim_n, pic_dim, filter_size, subpic_pad)

    if plotter == 'basic':
        plt.imshow(layer_0.filter_bank[0, :, :, :]*layer_0.filter_size ** 2)
        plt.show()

    if plotter == 'full':
        #if layer_0.filter_number > 10:
            #print('Warning: You are about to print %d images' % layer_0.filter_number)
            #proceed = input('Proceed?')

        subpltdim = int(np.ceil(np.sqrt(layer_0.filter_number)))
        # plot the filters
        fig = plt.figure(1)
        plt.title('Filters')
        plt.axis('off')
        for i in range(layer_0.filter_number):
            fig.add_subplot(subpltdim, subpltdim, (i+1))
            plt.imshow(layer_0.filter_bank[i, :, :, :]*(layer_0.filter_size ** 2))
        # plot the image bank
        fig = plt.figure(2)
        plt.title('Sub-images')
        plt.axis('off')
        for i in range(layer_0.filter_number):
            fig.add_subplot(subpltdim, subpltdim, (i+1))
            plt.imshow(layer_0.im_bank[i, :, :, :])
        #plot the feature maps
        fig = plt.figure(3)
        plt.title('Feature Maps')
        plt.axis('off')
        for i in range(layer_0.filter_number):
            fig.add_subplot(subpltdim, subpltdim, (i+1))
            plt.imshow(layer_0.feature_maps[i, :, :, :])

        plt.show()

if __name__ == "__main__":
    main(plotter='full')
    # main()
