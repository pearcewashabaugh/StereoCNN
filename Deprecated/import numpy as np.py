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
    print(aspect_ratio)
    return leftim_n, rightim_n, pic_dim


###############################################################################
#
class conv_pool_layer(object):
    # This class will take two given pictures, divide one up into subpictures, each
    # of which will be used as a filter, and convolve each filter against the other image

    def __init__(self, im1, im2, pic_dim, filter_size):
        self.im1 = im1
        self.im2 = im2
        self.pic_dim = pic_dim
        self.filter_size = filter_size

        # Construct collection of filters by calling filter_maker
        self.filter_bank, self.im_bank, self.filter_number, self.filter_bank_L2, \
        self.im_bank_L2 = self.filter_maker()

        # Convolve images by calling image_convolver()
        self.feature_maps = self.image_convolver()

        # Construct pooling layer by calling feature_pooler()
        self.pooled_features = self.feature_pooler()

    def filter_maker(self):
        # Makes bank of filters for convolution
        filter_spacer = int(np.floor((self.filter_size) / 2))
        # the number of filters that will fit vertically
        filter_number_h = int(np.floor(self.pic_dim[0] / self.filter_size))
        # the number of filters that will fit horizontally
        filter_number_w = int(np.floor(self.pic_dim[1] / self.filter_size))
        filter_number = filter_number_h * filter_number_w
        # Create collection of filters in filter bank, each will generate a feature map
        filter_bank = []
        # The collection of pieces im2 that will be convolved against each filter
        im_bank = []
        # parameter to keep track of filter number in for loop
        # this clearly needs to be improved upon
        for i in range(filter_number_h):
            for j in range(filter_number_w):
                filter_bank.append(self.im1[i * self.filter_size:(i + 1)
                                                                 * self.filter_size,
                                   j * self.filter_size:(j + 1) * self.filter_size])

        for i in range(int(round(filter_number_h * .5+.00000001))):
            for j in range(int(round(filter_number_w * .5+.00000001))):
                im_bank.append(self.im2[int(np.floor(i * self.filter_size * 2)):int(np.floor((i + 1)
                                                                                             * self.filter_size * 2)),
                               int(np.floor(j * self.filter_size * 2)):int(np.floor((j + 1)
                                                                                    * self.filter_size * 2))])
                im_bank.append(im_bank[-1])
                im_bank.append(im_bank[-1])
                im_bank.append(im_bank[-1])

        filter_bank = np.array(filter_bank)
        im_bank = np.array(im_bank)
        print("filter_number_h: %d" % filter_number_h)
        print("filter_number_w: %d" % filter_number_w)
        print("filter_number: %d" % filter_number)
        print("im_number_h: %d" % int(round(filter_number_h * .5+.00000001)))
        print("im_number_w: %d" % int(round(filter_number_w * .5+.00000001)))
        print("im_number: % d" % (int(round(filter_number_h * .5+.00000001)) * int(round(filter_number_w * .5+.00000001))))
        print(np.shape(filter_bank))
        print(np.shape(im_bank))

        # A list which contains the average L2 norm of every filter in filter_bank
        filter_bank_L2 = []
        im_bank_L2 = []
        for i in range(filter_number):
            filter_bank_L2.append(np.linalg.norm(filter_bank[i]) ** 2)
            im_bank_L2.append(np.linalg.norm(im_bank[i]) ** 2)
            # Normalize the size of the entries of the filter
            filter_bank[i] = filter_bank[i] / (self.filter_size ** 2)

        return filter_bank, im_bank, filter_number, filter_bank_L2, im_bank_L2

    def image_convolver(self):
        # Outputs feature maps
        feature_maps = np.zeros((self.filter_number, np.shape(self.im_bank[0])[0], np.shape(self.im_bank[0])[1], 3))
        print("filter_number: %d" % self.filter_number)

        # space = int(np.floor(float(self.filter_size)/2))
        space = int(np.floor(float(self.filter_size) / 2))
        # for i in range(filter_number):
        for i in range(1):
            print("Current convolution iteration: %d" % i)
            # convolve_R = convolve2d(self.im2[:,:,0],self.filter_bank[i,:,:,0],
            # 	boundary = 'fill', fillvalue = .2)
            # convolve_G = convolve2d(self.im2[:,:,1],self.filter_bank[i,:,:,1],
            # 	boundary = 'fill', fillvalue = .2)
            # convolve_B = convolve2d(self.im2[:,:,2],self.filter_bank[i,:,:,2],
            # 	boundary = 'fill', fillvalue = .2)
            convolve_R = convolve2d(self.im_bank[i, :, :, 0], self.filter_bank[i, :, :, 0],
                                    boundary='fill', fillvalue=.2)
            convolve_G = convolve2d(self.im_bank[i, :, :, 1], self.filter_bank[i, :, :, 1],
                                    boundary='fill', fillvalue=.2)
            convolve_B = convolve2d(self.im_bank[i, :, :, 2], self.filter_bank[i, :, :, 2],
                                    boundary='fill', fillvalue=.2)

            print(np.shape(convolve_R))
            # Cut out unnecessary boundary and re-weigh by difference in L^2 norm
            feature_maps[i, :, :, 0] = convolve_R[space: -space, space: -space]
            feature_maps[i, :, :, 1] = convolve_G[space: -space, space: -space]
            feature_maps[i, :, :, 2] = convolve_B[space: -space, space: -space]
            return feature_maps

    def feature_pooler(self):
        # Pools results of feature_maps
        pooled_features = np.zeros((1, 1))
        return pooled_features


###############################################################################
# EVALUATE
###############################################################################

def main(debug=True):
    # The length in pixels of the height the images will be adjusted to
    pic_height_pix = 200

    # Filters for convolution will be of size filter_size x filter_size.
    # Choose a number that will divide both pic_height_pix and pic_width_pix
    filter_size = 50

    # Resize images
    leftim_n, rightim_n, pic_dim = image_preprocessor(pic_height_pix)

    # Make Conv/Pool layer object
    layer_0 = conv_pool_layer(leftim_n, rightim_n, pic_dim, filter_size)

    # print(np.shape(rightim_n))
    # print(np.shape(layer_0.feature_maps[0,:,:,:]))
    # print(np.max(layer_0.feature_maps))
    # print(np.shape(leftim_n))
    # print(np.shape(layer_0.filter_bank[0]))
    # print(np.shape(layer_0.feature_maps[0,:,:,:]))
    # plt.imshow(layer_0.filter_bank[0])
    # fig = plt.figure()
    #    ax1 = plt.subplot2grid((6,1), (0,0),rowspan = 1, colspan = 1)
    #    ax2 = plt.subplot2grid((6,1), (1,0),rowspan = 4, colspan = 1, sharex = ax1)
    #    ax3 = plt.subplot2grid((6,1), (5,0),rowspan = 1, colspan = 1, sharex = ax1)

    plt.imshow(layer_0.feature_maps[0, :, :, :])
    # plt.imshow(layer_0.feature_maps[0,:,:,0])
    # plt.imshow(leftim_n)
    plt.show()


# if debug == True:
# 	print("aspect_ratio: %s" % aspect_ratio)
# 	print("filter_number: %d" % filter_number)
# 	print("filter_size: %d" % filter_size)
if __name__ == "__main__":
    main()
