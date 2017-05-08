# A sum of squared distance stereo algorithm written in python
###############################################################################


import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.transform import resize
from skimage import io


###############################################################################
# Process original images to standard sizes
###############################################################################


def image_preprocessor(pic_height_pix, im1loc, im2loc):
    # Load images and normalize their sizes
    leftim = io.imread(im1loc)
    rightim = io.imread(im2loc)

    # The aspect ratio of the original images
    aspect_ratio = np.shape(leftim[0, :, 1])[0] / np.shape(leftim[:, 0, 1])[0]
    pic_width_pix = int(np.floor(pic_height_pix * aspect_ratio))
    # Normalize the sizes of the images
    leftim_n = \
        ski.transform.resize(leftim, (pic_height_pix, pic_width_pix))
    rightim_n = \
        ski.transform.resize(rightim, (pic_height_pix, pic_width_pix))

    return leftim_n, rightim_n


###############################################################################
# A class that splits up the images
###############################################################################


class ImSplitDecisionLayer(object):
    # Parameters:
    #
    #   im1: First image to be divided up.
    #   im2: Second image to be divided up. We will use its subimages as windows
    #       and drag im1 across them taking differences along the way
    #   subim1_size: How large each subimage of im1 to be dragged along im2 is.
    #   subim2_pad: We take subim1_size and add 2*subim2_pad (one for each side)
    #       to get the size of each subim2, the window of the second image.
    #
    # Methods:
    #
    #   subim_maker: Makes the collections of subimages whose differences will be
    #       compared
    #
    #   image_diff: Takes the differences of the given images and outputs. Outputs
    #       Feature_maps which have larger values for regions with closer l2 distance
    #
    #   decision_maker: Integrates an indicator function * Feature_maps where the
    #       indicator function is over a square region of side length int_length.
    #       Then chooses the region with the larges value as where the upper left
    #       pixel of subim1 should go for each subim2.

    def __init__(self, im1, im2, subim1_size, subim2_pad):
        self.im1 = im1
        self.im2 = im2
        self.pic_dim = np.shape(im1[:, :, 0])

        # Construct collections of subimages by calling subim_maker
        self.subim1_bank, self.subim2_bank, self.subim1_number, \
            self.subim2_size, self.im2_pad = self.subim_maker(im1, im2, subim1_size, subim2_pad)

        # Find the squared sum of differences by calling image_diff()
        self.feature_maps = self.image_diff(subim1_size)

        # Decide where the upper right corner of each subim1 belongs
        self.local_disparity, self.global_disparity, self.global_disp_map,\
            self.offset_map = self.decision_maker(subim1_size)

    def subim_maker(self, im1, im2, subim1_size, subim2_pad):

        # the number of subim1 that will fit vertically
        subim1_number_h = int(np.floor(self.pic_dim[0] / subim1_size))
        # the number of subim1 that will fit horizontally
        subim1_number_w = int(np.floor(self.pic_dim[1] / subim1_size))
        subim1_number = subim1_number_h * subim1_number_w
        # the length of a subpicture
        subim2_size = subim1_size + 2 * subim2_pad
        # Create collection of subimages in subim1_bank, each will generate a feature map
        subim1_bank = []
        # The collection of pieces im2 that will act as windows against each subim1
        subim2_bank = []
        # pad im2 with its mean along each axis
        im2_pad = np.pad(im2, ((subim2_pad, subim2_pad), (subim2_pad, subim2_pad), (0, 0)),
                         'constant')
        # Load up the subim1 and subim2 banks
        for i in range(subim1_number_h):
            for j in range(subim1_number_w):
                subim1_bank.append(im1[i * subim1_size:(i + 1) * subim1_size,
                                   j * subim1_size:(j + 1) * subim1_size,:])
                # (a,b) are the coordinates of the upper left point of the current subim1
                a = i * subim1_size
                b = j * subim1_size
                subim2_bank.append(im2_pad[a:(a + subim2_size), b:(b + subim2_size), :])

        subim1_bank = np.asarray(subim1_bank)
        subim2_bank = np.asarray(subim2_bank)

        return subim1_bank, subim2_bank, subim1_number, subim2_size, im2_pad

    def image_diff(self, subim1_size):
        # Initialize array of unpadded feature maps
        feature_maps_pre = np.zeros(
            (self.subim1_number, self.subim2_size - subim1_size, self.subim2_size - subim1_size, 3))
        # Convolve each subim1 across its corresponding subim2

        for k in range(self.subim1_number):
            if k % 100 == 0:
                print("Computing l2 differences of subpictures and filters: %d/%d" % (k,self.subim1_number))
            diff_mat = np.zeros((self.subim2_size - subim1_size, self.subim2_size - subim1_size, 3))
            for i in range(self.subim2_size - subim1_size):
                for j in range(self.subim2_size - subim1_size):
                    temp = self.subim2_bank[k, :, :, :]
                    tempdiff = temp[i:i + subim1_size, j:j + subim1_size, :] - self.subim1_bank[k, :, :, :]
                    diff_mat[i, j, :] = np.linalg.norm(tempdiff, axis=(0, 1))

            # normalize the l2 norms for plotting purposes. take 1/ to penalize larger distances
            feature_maps_pre[k, :, :, :] = 1 / (diff_mat / self.subim1_number)
            # compress to 0 to 1
            for l in range(3):
                feature_maps_pre[k, :, :, l] /= np.max(feature_maps_pre[k, :, :, l])

        # Final array of padded feature maps. We pad to get the full range of pixels in decision_maker
        feature_maps = np.pad(feature_maps_pre, ((0, 0), (0, subim1_size), (0, subim1_size), (0, 0)),
                              'mean')
        # Enhance contrast if desired:
        # feature_maps = 1 / (1 + np.exp(5*(1-2*feature_maps_pad)))
        return feature_maps

    def decision_maker(self, subim1_size):
        # initialize array of max values of all integrals over feature_maps with squares with length int_len.
        # points to where in each subim2 the algorithm thinks the upper left corner of subim1 lies
        local_disparity = []
        # global_disparity points to where the algorithm thinks each subim1 lies in im2.
        global_disparity = []
        # This will be used to go from the subim2 coord system to the im2 coord system with the disparity map
        subim1_number_h = int(np.floor(self.pic_dim[0] / subim1_size))
        subim1_number_w = int(np.floor(self.pic_dim[1] / subim1_size))
        # this will be an array whose (i,j)th component has a vector [i0,j0,i1,j1] where i0,j0 is the original
        # coordinate of a point in im1 and (i1,j1) is its proposed new coordinate in im2.
        global_disp_map = np.zeros((subim1_number_h, subim1_number_w, 4))
        # this will be an array containing the distance that each point moves
        offset_map = np.zeros((subim1_number_h, subim1_number_w))
        for k in range(self.subim1_number):
            if k % 100 == 0:
                print("Computing integrals of feature map %d" % k)
            int_mat = np.zeros((self.subim2_size - subim1_size, self.subim2_size - subim1_size))
            int_len = 4
            for i in range(self.subim2_size - subim1_size):
                for j in range(self.subim2_size - subim1_size):
                    temp = self.feature_maps[k, :, :, :]
                    int_mat[i, j] = np.sum(temp[i:i + int_len, j:j + int_len, :])
            #local_disparity.append(np.where(int_mat == int_mat.max())[0])
            local_disparity.append(np.unravel_index(int_mat.argmax(), int_mat.shape))

            # the positions of the kth subim1
            ipos = int(np.floor(k / subim1_number_w))
            jpos = int(k - ipos*subim1_number_w)

            a = ipos*subim1_size
            b = jpos*subim1_size

            #print(local_disparity[k][0])
            global_disparity.append([int(local_disparity[k][0])+a, int(local_disparity[k][1])+b])

            global_disp_map[ipos, jpos, :] = np.array([a, b, global_disparity[k][0], global_disparity[k][1]])

            offset_map[ipos, jpos] = np.linalg.norm(global_disparity[k] - np.array([a, b]))

        return local_disparity, global_disparity, global_disp_map, offset_map


###############################################################################
# Function for plotting
###############################################################################

def plotter(layer, plotlev='basic'):
    # This just helps make sure the user doesn't accidentally print far too many images.
    # This bit can clearly be improved upon
    if plotlev == 'none':
        return
    if plotlev == 'basic':
        plt.figure()
        plt.title('Displacement Distance')
        plt.imshow(layer.offset_map[:, :], cmap='hot', interpolation='nearest')

        plt.show()

    if plotlev == 'full':
        if layer.subim1_number > 50:
            print('WARNING: You are about to print %d images to your screen.' % (3 * layer.subim1_number))
            proceed = input('Proceed? (y/n)\n')
            if (proceed == 'n') | (proceed == 'no'):
                print('To print fewer to no images call main with arg plotter = \'basic\' or \'none\' ')
                return

        subpltdim = int(np.ceil(np.sqrt(layer.subim1_number)))
        # plot the subim1 bank
        fig = plt.figure(1)
        plt.title('subim1')
        plt.axis('off')
        for i in range(layer.subim1_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer.subim1_bank[i, :, :, :])

        # plot the subim2 bank
        fig = plt.figure(2)
        plt.title('subim2')
        plt.axis('off')
        for i in range(layer.subim1_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer.subim2_bank[i, :, :, :])
            plt.axhline(y=layer.local_disparity[i][0])
            plt.axvline(x=layer.local_disparity[i][1])

        # plot the feature maps
        fig = plt.figure(3)
        plt.title('Feature Maps')
        plt.axis('off')
        for i in range(layer.subim1_number):
            fig.add_subplot(subpltdim, subpltdim, (i + 1))
            plt.imshow(layer.feature_maps[i, :, :, :])

        # plot the final disparity map
        plt.figure(4)
        plt.title('im2finaldisp')
        plt.axis('off')
        plt.imshow(layer.im2_pad[:, :, :])
        width = float(layer.im2_pad[0, :, 0].shape[0])
        height = float(layer.im2_pad[:, 0, 0].shape[0])
        for i in range(layer.subim1_number):
            xmin1 = layer.global_disparity[i][1]/width
            xmax1 = layer.global_disparity[i][1]/width+.05
            ymin1 = 1 - layer.global_disparity[i][0]/height-.05
            ymax1 = 1 - layer.global_disparity[i][0]/height
            plt.axhline(y=layer.global_disparity[i][0], xmin=xmin1, xmax=xmax1)
            plt.axvline(x=layer.global_disparity[i][1], ymin=ymin1, ymax=ymax1)

        # a heat map of the distances of each pixel
        plt.figure(5)
        plt.title('Displacement Distance')
        plt.imshow(layer.offset_map[:, :], cmap='hot', interpolation='nearest')

        plt.show()


###############################################################################
# EVALUATE
###############################################################################


def main():
    # The length in pixels of the height the images will be adjusted to
    pic_height_pix = 100
    # Filters for convolution will be of size filter_size x filter_size.
    # Choose a number that will divide both pic_height_pix and pic_width_pix
    subim1_size = 20
    # the number of pixels to pad each filter by to create each subpicture
    subim2_pad = 10
    # Resize images
    leftim_n, rightim_n = image_preprocessor(pic_height_pix,
                                             'Data/Pictures_Data/left.jpeg',
                                             'Data/Pictures_Data/right.jpeg')
    # Make Conv/Pool layer object
    layer_0 = ImSplitDecisionLayer(leftim_n, rightim_n, subim1_size, subim2_pad)

    # The new location of the upper left corner of subim1
    # print(layer_0.global_disparity)

    plotter(layer_0, plotlev = "full")


if __name__ == "__main__":
    main()
