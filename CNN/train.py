
###############################################################################
# Import general modules

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
import sys

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
    print("Loading and chopping images.")
    for filename in os.listdir("../Data/middlebury-perfect"):
    
        calib = open('../Data/middlebury-perfect/%s/calib.txt' % filename, 'r')
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
        lIm = image_resizer("../Data/middlebury-perfect/%s/im0.png" % filename,
                                pic_h_px,
                                pic_width_pix = pic_w_px)
        rIm = image_resizer("../Data/middlebury-perfect/%s/im1.png" % filename,
                                pic_h_px,
                                pic_width_pix = pic_w_px)
    
        Im_bank.append([lIm,rIm])
        # Resize disparity to correct dimensions of other images.
        disp= image_resizer("../Data/middlebury-perfect/%s/disp0_000.png" % filename,
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

    s_Im_pos_bank_fin = []
    s_Im_neg_bank_fin = []

    for i in range(s_num_h):
        for j in range(s_num_w):
            for k in range(len(s_Im_bank)):
                temppos = np.append(s_Im_bank[k][0][i*s_num_w+j],s_Im_bank[k][1][i*s_num_w+j][0], axis =1)
                s_Im_pos_bank_fin.append([temppos,[i,j]])
                tempneg = np.append(s_Im_bank[k][0][i*s_num_w+j],s_Im_bank[k][1][i*s_num_w+j][1], axis =1)
                s_Im_neg_bank_fin.append([tempneg,[i,j]])

    # fig = plt.figure(1)
    # plt.title('s_lIm')
    # plt.imshow( s_lIm_bank_ij[2,3,-3,:,:,:])
    # fig = plt.figure(2)
    # plt.title('s_rIm_pos')
    # plt.imshow( s_rIm_pos_bank_ij[2,3,-3,:,:,:])
    # fig = plt.figure(3)
    # plt.title('s_rIm_neg')
    # plt.imshow( s_rIm_neg_bank_ij[2,3,-3,:,:,:])
    # fig = plt.figure(4)
    # plt.title('lIm')
    # plt.imshow(lIm)
    # fig = plt.figure(5)
    # plt.title('rIm')
    # plt.imshow(rIm)
    
    # plt.show()
    
    #sys.exit()
    ###############################################################################
    # Create a dual-conv-neural net.
    print("Creating and training neural net:")
    dnn.dual_conv_net(s_Im_pos_bank_fin, s_Im_neg_bank_fin)
    
