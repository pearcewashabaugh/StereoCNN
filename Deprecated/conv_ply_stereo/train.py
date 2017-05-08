
###############################################################################
# Import general modules

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
import sys

###############################################################################
# Import other files from Bird's eye

import dual_neural as dnn
from image_preprocessor import image_resizer, subim_maker_trainer

###############################################################################
# Global constants

# Re-scale all training, testing, and evaluation data to these dimensions.
pic_h_pix = 100
pic_w_pix = int(pic_h_pix*(1+1/3.0))

# The size of a subimage of the left picture (in pixels)
s_lIm_size = 20

# The s_rIm_pad + s_lIm_size = s_lIm_w (the left subim width)
s_rIm_pad = 10

###############################################################################
# Load up training data

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
                            pic_h_pix,
                            pic_width_pix = pic_w_pix)
    rIm = image_resizer("../Data/middlebury-perfect/%s/im1.png" % filename,
                            pic_h_pix,
                            pic_width_pix = pic_w_pix)

    Im_bank.append([lIm,rIm])
    # Resize disparity to correct dimensions of other images.
    disp= image_resizer("../Data/middlebury-perfect/%s/disp0_000.png" % filename,
                            pic_h_pix, 
                            pic_width_pix = pic_w_pix,
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
# Re-organize s_Im_bank into arrays that allow us to select all training
# examples associated to a position i,j]

s_lIm_bank_ij = np.zeros((s_num_h, s_num_w, len(s_Im_bank),
                        s_lIm_size,s_lIm_size,3 ))
s_rIm_pos_bank_ij = np.zeros((s_num_h, s_num_w, len(s_Im_bank),
                        s_lIm_size,s_lIm_size,3 ))
s_rIm_neg_bank_ij = np.zeros((s_num_h, s_num_w, len(s_Im_bank),
                        s_lIm_size,s_lIm_size,3 ))

for i in range(s_num_h):
    for j in range(s_num_w):
        for k in range(len(s_Im_bank)):
            s_lIm_bank_ij[i,j,k,:,:,:] = s_Im_bank[k][0][i*s_num_w+j]
            s_rIm_pos_bank_ij[i,j,k,:,:,:] = s_Im_bank[k][1][i*s_num_w+j][0]
            s_rIm_neg_bank_ij[i,j,k,:,:,:] = s_Im_bank[k][1][i*s_num_w+j][1]

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
# Create a dual-conv-neural net for each subimage position and train.
print("Creating and training neural nets:")
dualnet_list = []
for i in range(s_num_h):
    for j in range(s_num_w):
        print("neural net ", i*s_num_w+j+1, " out of ",s_num_h * s_num_w)
        dualnet_list.append(dnn.dual_conv_net(s_lIm_bank_ij[i,j,:,:,:,:],
            s_rIm_pos_bank_ij[i,j,:,:,:,:],s_rIm_neg_bank_ij[i,j,:,:,:,:], (i,j)))

