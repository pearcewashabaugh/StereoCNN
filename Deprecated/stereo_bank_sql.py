# A sum of squared distance stereo algorithm written in python
###############################################################################


import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.transform import resize
from skimage import io
import sqlite3

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

        self.conn = sqlite3.connect('image_quantities.db')
        self.c = self.conn.cursor()


        # Construct collections of subimages by calling subim_maker
        self.subim1_number, self.subim2_size, self.im2_pad \
            = self.subim_maker(im1, im2, subim1_size, subim2_pad)

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
        self.c.execute('CREATE TABLE IF NOT EXISTS subim1bank(im_index INTEGER, subim1_ij TEXT)')
        # delete current contents of SQL subim1bank table to make way for new computations
        self.c.execute('DELETE FROM subim1bank')
        #subim1_bank = []
        # The collection of pieces im2 that will act as windows against each subim1
        self.c.execute('CREATE TABLE IF NOT EXISTS subim2bank(im_index INTEGER, subim2_ij TEXT)')
        # delete current contents of SQL subim2bank table
        self.c.execute('DELETE FROM subim2bank')
        #subim2_bank = []
        # pad im2 with its mean along each axis
        im2_pad = np.pad(im2, ((subim2_pad, subim2_pad), (subim2_pad, subim2_pad), (0, 0)),
                         'constant')
        # Keeps track of subim number
        im_index = 0

        # Load up the subim1 and subim2 banks
        for i in range(subim1_number_h):
            for j in range(subim1_number_w):
                # subim1_bank.append(im1[i * subim1_size:(i + 1) * subim1_size,
                #               j * subim1_size:(j + 1) * subim1_size,:])
                subim1_ij = im1[i * subim1_size:(i + 1) * subim1_size,
                                   j * subim1_size:(j + 1) * subim1_size, :].tostring()
                self.c.execute("INSERT INTO subim1bank (im_index, subim1_ij) VALUES (?, ?)",
                          (im_index, subim1_ij))
                # (a,b) are the coordinates of the upper left point of the current subim1
                a = i * subim1_size
                b = j * subim1_size
                # subim2_bank.append(im2_pad[a:(a + subim2_size), b:(b + subim2_size), :])
                subim2_ij = im2_pad[a:(a + subim2_size), b:(b + subim2_size), :].tostring()
                self.c.execute("INSERT INTO subim2bank (im_index, subim2_ij) VALUES (?, ?)",
                          (im_index, subim2_ij))
                self.conn.commit()
                im_index += 1

        # This code recovers subim1[0] from the SQL database:
        # self.c.execute("SELECT subim1_ij FROM subim1bank")
        # tempfilt = self.c.fetchall()[0][0]
        # newfilt = np.fromstring(tempfilt, dtype=float)
        # newfilt2 = newfilt.reshape(subim1_size,subim1_size,3)
        # plt.figure()
        # plt.imshow(newfilt2)
        # subim1_bank = np.asarray(subim1_bank)
        # subim2_bank = np.asarray(subim2_bank)

        return subim1_number, subim2_size, im2_pad

    def image_diff(self, subim1_size):
        # Initialize array of unpadded feature maps
        feature_maps_pre = np.zeros(
            (self.subim1_number, self.subim2_size - subim1_size, self.subim2_size - subim1_size, 3))
        # SQL table of feature maps before padding
        self.c.execute('CREATE TABLE IF NOT EXISTS feature_maps_pre(im_index INTEGER, fmappre TEXT)')
        self.c.execute('DELETE FROM feature_maps_pre')
        # SQL table of feature maps after padding
        self.c.execute('CREATE TABLE IF NOT EXISTS feature_maps(im_index INTEGER, fmap TEXT)')
        self.c.execute('DELETE FROM feature_maps')
        # Convolve each subim1 across its corresponding subim2
        for k in range(self.subim1_number):
            #if k % 100 == 0:
            print("Computing l2 differences of subpictures and filters: %d/%d" % (k,self.subim1_number))
            diff_mat = np.zeros((self.subim2_size - subim1_size, self.subim2_size - subim1_size, 3))
            for i in range(self.subim2_size - subim1_size):
                for j in range(self.subim2_size - subim1_size):
                    self.c.execute("SELECT subim2_ij FROM subim2bank")
                    temp21 = self.c.fetchall()[k][0]
                    temp22 = np.fromstring(temp21, dtype=float)
                    temp = temp22.reshape(self.subim2_size,self.subim2_size,3)
                    #temp = self.subim2_bank[k, :, :, :]
                    self.c.execute("SELECT subim1_ij FROM subim1bank")
                    temp11 = self.c.fetchall()[k][0]
                    temp12 = np.fromstring(temp11, dtype=float)
                    sub1bank = temp12.reshape(subim1_size, subim1_size,3)
                    tempdiff = temp[i:i + subim1_size, j:j + subim1_size, :] - sub1bank
                    diff_mat[i, j, :] = np.linalg.norm(tempdiff, axis=(0, 1))


            # normalize the l2 norms for plotting purposes. take 1/ to penalize larger distances
            feature_maps_pre[k, :, :, :] = 1 / (diff_mat / self.subim1_number)
            im_index = k
            fmappre = feature_maps_pre[k].tostring()
            self.c.execute("INSERT INTO feature_maps_pre (im_index, fmappre) VALUES (?, ?)",
                           (im_index, fmappre))
            self.conn.commit()
            # compress to 0 to 1
            for l in range(3):
                feature_maps_pre[k, :, :, l] /= np.max(feature_maps_pre[k, :, :, l])

        # Final array of padded feature maps. We pad to get the full range of pixels in decision_maker
        feature_maps = np.pad(feature_maps_pre, ((0, 0), (0, subim1_size), (0, subim1_size), (0, 0)),
                              'mean')
        for k in range(self.subim1_number):
            fmap = feature_maps[k].tostring()
            im_index = k
            self.c.execute("INSERT INTO feature_maps (im_index, fmap) VALUES (?, ?)",
                       (im_index, fmap))
            self.conn.commit()
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
            local_disparity.append(np.unravel_index(int_mat.argmax(), int_mat.shape))
            # the positions of the kth subim1
            ipos = int(np.floor(k / subim1_number_w))
            jpos = int(k - ipos*subim1_number_w)

            a = ipos*subim1_size
            b = jpos*subim1_size

            global_disparity.append([int(local_disparity[k][0])+a, int(local_disparity[k][1])+b])

            global_disp_map[ipos, jpos, :] = np.array([a, b, global_disparity[k][0], global_disparity[k][1]])

            offset_map[ipos, jpos] = np.linalg.norm(global_disparity[k] - np.array([a, b]))
        self.c.close()
        self.conn.close()
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

###############################################################################
# EVALUATE
###############################################################################


def main():
    # The length in pixels of the height the images will be adjusted to
    pic_height_pix = 100
    # Filters for convolution will be of size filter_size x filter_size.
    # Choose a number that will divide both pic_height_pix and pic_width_pix
    subim1_size = 10
    # the number of pixels to pad each filter by to create each subpicture
    subim2_pad = 20
    # Resize images
    leftim_n, rightim_n = image_preprocessor(pic_height_pix,
                                             'Data/Pictures_Data/left.jpeg',
                                             'Data/Pictures_Data/right.jpeg')
    # Make Conv/Pool layer object
    layer_0 = ImSplitDecisionLayer(leftim_n, rightim_n, subim1_size, subim2_pad)

    # The new location of the upper left corner of subim1
    # print(layer_0.global_disparity)

    plotter(layer_0)


if __name__ == "__main__":
    main()
