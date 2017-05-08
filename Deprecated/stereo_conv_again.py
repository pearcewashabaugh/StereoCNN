
###############################################################################
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
# This class name is misleading at the moment as we're not making a CNN

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
        subpic_size = self.filter_size + 2 * self.subpic_pad
        # Create collection of filters in filter bank, each will generate a feature map
        filter_bank = []
        # The collection of pieces im2 that will be convolved against each filter
        im_bank = []
        # pad im2 with its mean along each axis
        im2_pad = np.pad(self.im2, ((self.subpic_pad, self.subpic_pad), (self.subpic_pad, self.subpic_pad), (0, 0)),
                         'mean')
        # Load up the filter and image banks
        for i in range(filter_number_h):
            for j in range(filter_number_w):
                filter_bank.append(self.im1[i * self.filter_size:(i + 1) * self.filter_size,
                                   j * self.filter_size:(j + 1) * self.filter_size])
                # (a,b) are the coordinates of the upper left point of the current filter
                a = i * self.filter_size
                b = j * self.filter_size
                im_bank.append(im2_pad[a:(a + subpic_size), b:(b + subpic_size), :])

        filter_bank = np.asarray(filter_bank)
        im_bank = np.asarray(im_bank)
        print("filter_number_h: %d" % filter_number_h)
        print("filter_number_w: %d" % filter_number_w)
        print("filter_number: %d" % filter_number)
        print(np.shape(filter_bank))
        print(np.shape(im_bank))

        # A list which contains the average L2 norm of every filter in filter_bank.
        # Each entry is a triple containing the R, G, and B L^2 norms
        filter_bank_l2 = []
        # Each entry here contains the L^2 norm of the R,G,B values in a region the size of a filter around
        # point (i,j) in each kth subimage
        im_bank_l2 = []
        # fill up filter_bank_l2
        for i in range(filter_number):
            filter_bank_l2.append(np.linalg.norm(filter_bank[i, :, :, :], axis=(0, 1)))
            # Normalize the size of the entries of the filter and filter l2
            filter_bank[i] /= (self.filter_size ** 2)
            filter_bank_l2[i] /= (self.filter_size)

        # fill up im_bank_l2
        for k in range(filter_number):
            print("Computing l2 norms of subpicture %d" % k)
            im_l2_mat = np.zeros((subpic_size, subpic_size, 3))
            for i in range(subpic_size):
                for j in range(subpic_size):
                    temp = np.pad(im_bank[k, :, :, :],
                                  ((self.filter_size, self.filter_size), (self.filter_size, self.filter_size), (0, 0)),
                                  'mean')
                    im_l2_mat[i, j, :] = np.linalg.norm(temp[i:i+self.filter_size, j:j+self.filter_size], axis=(0, 1))
            # normalize to account for size of filter
            im_l2_mat /= (self.filter_size)
            im_bank_l2.append(im_l2_mat)

        im_bank_l2 = np.asarray(im_bank_l2)
        print(filter_bank_l2)

        return filter_bank, im_bank, filter_number, filter_bank_l2, im_bank_l2

    def image_convolver(self):
        # Initialize array of feature maps
        feature_maps = np.zeros((self.filter_number, np.shape(self.im_bank[0])[0], np.shape(self.im_bank[0])[1], 3))
        # Convolve each filter across its corresponding subimage
        for i in range(self.filter_number):
            print("Current convolution iteration: %d" % i)

            feature_maps[i, :, :, 0] = np.sqrt(convolve2d(self.im_bank[i, :, :, 0], self.filter_bank[i, :, :, 0], mode="same",
                                                  boundary='fill', fillvalue=.2))
            feature_maps[i, :, :, 1] = np.sqrt(convolve2d(self.im_bank[i, :, :, 1], self.filter_bank[i, :, :, 1], mode="same",
                                                  boundary='fill', fillvalue=.2))
            feature_maps[i, :, :, 2] = np.sqrt(convolve2d(self.im_bank[i, :, :, 2], self.filter_bank[i, :, :, 2], mode="same",
                                                  boundary='fill', fillvalue=.2))

        # Now re-weight each feature map point according to how different its L^2 norm is from the filter
        for i in range(self.filter_number):
            l2_diff = 100 * np.abs(self.im_bank_l2[i, :, :, :]-self.filter_bank_L2[i])
            feature_maps[i, :, :, :] *= (1/(1+l2_diff))

        return feature_maps

    def feature_pooler(self):
        # Pools results of feature_maps
        pooled_features = np.zeros((1, 1))
        return pooled_features


###############################################################################
# EVALUATE
###############################################################################

def main(plotter='basic'):
    # The length in pixels of the height the images will be adjusted to
    pic_height_pix = 100
    # Filters for convolution will be of size filter_size x filter_size.
    # Choose a number that will divide both pic_height_pix and pic_width_pix
    filter_size = 20
    # the number of pixels to pad each filter by to create each subpicture
    subpic_pad = 10

    # Resize images
    leftim_n, rightim_n, pic_dim = image_preprocessor(pic_height_pix)

    # Make Conv/Pool layer object
    layer_0 = ConvPoolLayer(leftim_n, leftim_n, pic_dim, filter_size, subpic_pad)
    # print(np.max(leftim_n))
    # print(np.max(layer_0.filter_bank))
    # print(np.max(layer_0.im_bank))
    # print(np.max(layer_0.feature_maps))
    # This just helps make sure the user doesn't accidentally print far too many images.
    # This bit can clearly be improved upon
    if plotter == 'none':
        return
    if plotter == 'basic':
        plt.figure()
        plt.imshow(layer_0.filter_bank[0, :, :, :] * layer_0.filter_size ** 2)
        plt.show()

    if plotter == 'full':
        if layer_0.filter_number > 50:
            print('WARNING: You are about to print %d images to your screen.' % (3 * layer_0.filter_number))
            proceed = input('Proceed? (y/n)\n')
            if (proceed == 'n') | (proceed == 'no'):
                print('To print fewer to no images call main with arg plotter = \'basic\' or \'none\' ')
                return

        subpltdim = int(np.ceil(np.sqrt(layer_0.filter_number)))
        # plot the filters
        fig = plt.figure(1)
        plt.title('Filters')
        plt.axis('off')
        for i in range(layer_0.filter_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer_0.filter_bank[i, :, :, :] * (layer_0.filter_size ** 2))
        # plot the image bank
        fig = plt.figure(2)
        plt.title('Sub-images')
        plt.axis('off')
        for i in range(layer_0.filter_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer_0.im_bank[i, :, :, :])
        # plot the feature maps
        fig = plt.figure(3)
        plt.title('Feature Maps')
        plt.axis('off')
        for i in range(layer_0.filter_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer_0.feature_maps[i, :, :, :])

        plt.show()
        # fig = plt.figure(4)
        # plt.title('imbankl2')
        # plt.axis('off')
        # for i in range(layer_0.filter_number):
        #     fig.add_subplot(subpltdim, subpltdim, (i + 1))
        #     plt.imshow(np.abs(layer_0.feature_maps[i, :, :, :]-layer_0.im_bank_l2[i, :, :, :]))
        #
        # plt.show()


if __name__ == "__main__":
    main(plotter='full')
    # main(plotter = 'none')
