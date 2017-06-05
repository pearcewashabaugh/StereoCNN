
###############################################################################
# Import general modules

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
import sys

from sklearn.decomposition import PCA

###############################################################################
# Import other files from Bird's eye

import CNN.dual_neural as dnn
from Image_PrePost_Processing.image_preprocessor import image_resizer, subim_maker_trainer

###############################################################################
# Global constants

# # Re-scale all training, testing, and evaluation data to these dimensions.
# pic_h_px = 50
# pic_w_px = int(pic_h_px*(1+1/3.0))

# # The size of a subimage of the left picture (in pixels)
# s_lIm_size = 10

# # The s_rIm_pad + s_lIm_size = s_lIm_w (the left subim width)
# s_rIm_pad = 5

###############################################################################
def trainer(pic_h_px, pic_w_px, s_lIm_size, s_rIm_pad):
    # Load up training Data
    
    # This list will hold the resized training data. It's ith element will be
    # a list with the left and right images.
    Im_bank = []
    
    # This list will hold all the disparity maps
    disp_bank = []
    
    # This list will hold the training data chopped up into subimages. Each entry
    # will be of the form [s_lIm_bank,s_rIm_bank]
    s_Im_bank = []
    
    im_index = 0
    print("Loading and chopping training images.")
    for filename in os.listdir("../../../Projects_Data/StereoCNN_Data/middlebury-perfect"):
        print('image', im_index, 'out of', len(os.listdir("../../../Projects_Data/StereoCNN_Data/middlebury-perfect"))-1)
        im_index += 1
        calib = open('../../../Projects_Data/StereoCNN_Data/middlebury-perfect/%s/calib.txt' % filename, 'r')
        calibr = calib.read()
        vminind = calibr.find('vmin=')
        vmaxind = calibr.find('vmax=')
        dyavgind = calibr.find('dyavg')
        vmin = int(calibr[vminind+5:vmaxind])
        vmax = int(calibr[vmaxind+5:dyavgind])
        widthind = calibr.find('width')
        heightind = calibr.find('height')
        oldw = calibr[widthind+6:heightind]
        # Resize images, output left image = lIm and right image = rIm
        # lIm, rIm = image_resize("../Data/middlebury-perfect/%s/im0.png" % filename,
        #                       "../Data/middlebury-perfect/%s/im1.png" % filename,
        #                       pic_height_pix)
        lIm = image_resizer("../../../Projects_Data/StereoCNN_Data/middlebury-perfect/%s/im0.png" % filename,
                                pic_h_px,
                                pic_width_pix = pic_w_px)
        rIm = image_resizer("../../../Projects_Data/StereoCNN_Data/middlebury-perfect/%s/im1.png" % filename,
                                pic_h_px,
                                pic_width_pix = pic_w_px)
    
        Im_bank.append([lIm,rIm])
        # Resize disparity to correct dimensions of other images.
        disp= image_resizer("../../../Projects_Data/StereoCNN_Data/middlebury-perfect/%s/disp0_000.png" % filename,
                                pic_h_px, 
                                pic_width_pix = pic_w_px,
                                displacement = True)
        disp_bank.append(disp)
        # Split images up into subimages
        s_lIm_bank, s_rIm_bank, s_num_h, s_num_w = subim_maker_trainer(lIm, rIm, 
                                                                        disp, 
                                                                        s_lIm_size,
                                                                        vmin, vmax,
                                                                        oldw)
        s_Im_bank.append([s_lIm_bank,s_rIm_bank])
    
    ###############################################################################
    # Re-organize s_Im_bank into lists containing positive/negative pairs and locations
    # fig = plt.figure(1)
    # plt.title('s_Im')
    # plt.imshow( s_Im_bank[0][0][2])
    # fig = plt.figure(2)
    # plt.title('s_rIm_pos')
    # plt.imshow( s_Im_bank[0][1][2][0])
    # fig = plt.figure(3)
    # plt.title('s_rIm_neg')
    # plt.imshow( s_Im_bank[0][1][2][1])
    # plt.show()
    # sys.exit()


    s_Im_pos_bank_fin = []
    s_Im_neg_bank_fin = []

    for i in range(s_num_h):
        for j in range(s_num_w):
            for k in range(len(s_Im_bank)):
                temppos = np.append(s_Im_bank[k][0][i*s_num_w+j],s_Im_bank[k][1][i*s_num_w+j][0], axis =1)
                s_Im_pos_bank_fin.append([temppos,[i,j]])
                tempneg = np.append(s_Im_bank[k][0][i*s_num_w+j],s_Im_bank[k][1][i*s_num_w+j][1], axis =1)
                s_Im_neg_bank_fin.append([tempneg,[i,j]])

    x_input = s_Im_pos_bank_fin + s_Im_neg_bank_fin
    y_input_pos = [1 for i in range(len(s_Im_pos_bank_fin))]
    y_input_neg = [0 for i in range(len(s_Im_neg_bank_fin))]
    y_input = y_input_pos + y_input_neg


    ###############################################################################
    # Whiten Data (i.e. perform a PCA and re-project so that we have mean zero
    # and variance 1)

    print("Whitening Training Data...")
    if s_lIm_size*2*s_lIm_size*3 > len(x_input):
        print("Not enough samples. Increase image size and/or decrease subim size")
        sys.exit()
    pca = PCA(n_components = s_lIm_size*2*s_lIm_size*3, whiten = True)

    # x_input = np.array(x_input)
    # x_input = x_input.reshape(x_input.shape[0],s_lIm_size*2*s_lIm_size*3)
    # x_input = pca.fit_transform(x_input)
    # x_input = x_input.reshape(x_input.shape[0],s_lIm_size, 2*s_lIm_size, 3)
    x_flat = np.zeros((len(x_input), s_lIm_size*2*s_lIm_size*3))

    for (iindex, i) in enumerate(x_input):
        x_flat[iindex,:] = (i[0].reshape(s_lIm_size*2*s_lIm_size*3))

    x_flat = pca.fit_transform(x_flat)
    for iindex in range(len(x_input)):
        x_input[iindex][0] = np.reshape(x_flat[iindex,:],(s_lIm_size, 2*s_lIm_size, 3))

    ###############################################################################
    # Create a dual-conv-neural net.
    print("Creating and training neural net...")
    dnn.dual_conv_net(x_input, y_input, s_num_h, s_num_w)
    
