import numpy as np
import skimage as ski
from skimage.transform import resize
import cv2
import sqlite3
import numpy.random
import sys
from skimage import io

def image_resize(im1loc, im2loc, pic_height_pix, pic_width_pix = False):
    # Load images and normalize their sizes so their height is pic_height_pix.

    # If pic_width_pix is declared, then set size of picture to 
    #   pix_width_pix by pix_height_pix
    leftim = io.imread(im1loc)
    rightim = io.imread(im2loc)
    # The aspect ratio of the original images
    aspect_ratio = np.shape(leftim[0, :, 1])[0] / np.shape(leftim[:, 0, 1])[0]
    if not pic_width_pix:
        pic_width_pix = int(np.floor(pic_height_pix * aspect_ratio))
    # Normalize the sizes of the images
    leftim_n = \
        ski.transform.resize(leftim, (pic_height_pix, pic_width_pix))
    rightim_n = \
        ski.transform.resize(rightim, (pic_height_pix, pic_width_pix))

    return leftim_n, rightim_n

def image_resizer(im1loc, pic_height_pix, pic_width_pix = False, displacement = False):
    # Load images and normalize their sizes so their height is pic_height_pix.

    # If pic_width_pix is declared, then set size of picture to 
    #   pix_width_pix by pix_height_pix
    leftim = io.imread(im1loc)
    # The aspect ratio of the original image
    aspect_ratio = np.shape(leftim[0, :, 1])[0] / np.shape(leftim[:, 0, 1])[0]
    if not pic_width_pix:
        pic_width_pix = int(np.floor(pic_height_pix * aspect_ratio))
    # Normalize the sizes of the images
    im = \
        ski.transform.resize(leftim, (pic_height_pix, pic_width_pix), mode = 'constant')

    # If the image represents a displacement map, make sure that it has only one
    # value for each pixel location.
    if displacement == True:

        if im[:,:].shape[2] == 3:
            im = im[:,:,0]
            
        #im *= (pic_height_pix / leftim.shape[0])


    return im

def subim_maker(im1, im2, subim1_size, subim2_pad):
    # the number of subim1 that will fit vertically
    subim1_number_h = int(np.floor(im2.shape[0] / subim1_size))
    # the number of subim1 that will fit horizontally
    subim1_number_w = int(np.floor(im2.shape[1] / subim1_size))
    # the total number of subimages
    subim1_number = subim1_number_h * subim1_number_w
    # the collection of subimages of im1
    subim1_bank = []
    # the width of a padded subimage
    subim2_width = subim2_pad + subim1_size
    # the height of a padded subimage. Should be the same as subim1 assuming 
    # rectification
    subim2_height = subim1_size
    # the collection of subimages of im2
    subim2_bank = []
    # pad im2 with its mean along middle axis
    im2_pad = np.pad(im2, ((0, 0), (subim2_pad, subim2_pad), (0, 0)),
                        'constant')
    # Keeps track of subim number
    im_index = 0
    # Load up the subim1 and subim2 banks
    for i in range(subim1_number_h):
        for j in range(subim1_number_w):
            subim1_bank.append(im1[i * subim1_size:(i + 1) * subim1_size,
                                j * subim1_size:(j + 1) * subim1_size,:])

        # (a,b) are the coordinates of the upper left point of the current subim1
        # imposed on the padded subim2. Hence each subim2 will be shifted to the left
        # by subim2_pad and contain subim1 to the right. 
            a = i * subim1_size
            b = j * subim1_size
            subim2_bank.append(im2_pad[a:(a + subim2_height), b:(b + subim2_width), :])
            im_index += 1

    return subim1_bank, subim2_bank, subim1_number_h, subim1_number_w

def subim_maker_trainer(im1, im2, disp, subim1_size, vmin, vmax, oldw):
    # the number of subim1 that will fit vertically
    subim1_number_h = int(np.floor(im2.shape[0] / subim1_size))
    # the number of subim1 that will fit horizontally
    subim1_number_w = int(np.floor(im2.shape[1] / subim1_size))
    # the total number of subimages
    subim1_number = subim1_number_h * subim1_number_w
    # the collection of subimages of im1
    subim1_bank = []
    # subimage1 and contains a list of both a positive and negative example.
    subim2_bank = []
    # pad im2 with its mean along each axis
    im2_pad = np.pad(im2, ((subim1_size, subim1_size), (subim1_size, subim1_size), (0, 0)),
                        'constant')

    disp_pad = np.pad(disp, ((subim1_size, subim1_size), (subim1_size, subim1_size)),
                        'constant')
    #disp_pad = disp_pad / np.max(disp_pad)
    # Keeps track of subim number
    im_index = 0
    # Load up the subim1 and subim2 banks
    for i in range(subim1_number_h):
        for j in range(subim1_number_w):
            subim1_bank.append(im1[i * subim1_size:(i + 1) * subim1_size,
                                j * subim1_size:(j + 1) * subim1_size,:])

            # (a,b) are the coordinates of the upper left point of the current subim1
            a = (i) * subim1_size + subim1_size
            b = (j) * subim1_size + subim1_size
            a_cent = a + int(subim1_size*.5)
            b_cent = b + int(subim1_size*.5)
            # almost certainly should add doffs to this but I won't for now
            d = disp_pad[a_cent,b_cent]

            #d = int(max(0,min(255,int(255*(1.5-4*abs(d-.5))))))
            d= int(d*(vmax-vmin)*im2.shape[1]/float(oldw)+(vmin)*im2.shape[1]/float(oldw))
            # if i == 4:
            #     if j == 5:
            #         print(d)

            if (b-d)<0:
                d = b
            subim2_pos = im2_pad[a:(a+subim1_size), (b-d):(b - d + subim1_size), :]
            if subim2_pos.shape == (0,10,3):
                print(i,j,a,b,d)
            whichoneg = np.random.random_sample([1])
            if whichoneg < .5:
                oneg = -np.random.random_sample([1])*3*(subim1_size+4)-4
            else:
                oneg = np.random.random_sample([1])*3*(subim1_size+4)+4
            oneg = int(oneg)
            if (b-d+oneg)<0:
                oneg = d-b

            elif (b-d+oneg+subim1_size)> im2_pad.shape[1]:
                oneg = d-b + im2_pad.shape[1] - subim1_size
            oneg = int(oneg)
            subim2_neg = im2_pad[a:(a + subim1_size), b-d+oneg:(b - d + oneg + subim1_size), :]
            subim2_bank.append([subim2_pos, subim2_neg])
            im_index += 1
            if subim2_neg.shape != (subim1_size,subim1_size,3):
                print("Size mismatch at:")
                print("i:", i,"j:", j, "a:", a, "b:", b, d, oneg)
                print("Try using a larger subim size.")
                sys.exit()

    return subim1_bank, subim2_bank, subim1_number_h, subim1_number_w

# def subim_maker_sql(im1, im2, subim1_size, subim2_pad):
#     conn = sqlite3.connect('image_quantities.db')
#     c = conn.cursor()
#     # the number of subim1 that will fit vertically
#     subim1_number_h = int(np.floor(im2.shape[0] / subim1_size))
#     # the number of subim1 that will fit horizontally
#     subim1_number_w = int(np.floor(im2.shape[1] / subim1_size))
#     # the total number of subimages
#     subim1_number = subim1_number_h * subim1_number_w
#     # the width of a padded subimage
#     subim2_width = subim2_pad + subim1_size
#     # the height of a padded subimage. Will be smaller than width assuming 
#     # rectification. Currently set to 1/4 of size
#     subim2_height = int(subim2_pad/4) + subim1_size
#     # Create collection of subimages in subim1_bank, each will generate a feature map
#     c.execute('CREATE TABLE IF NOT EXISTS subim1bank(im_index INTEGER, subim1_ij TEXT)')
#     # delete current contents of SQL subim1bank table to make way for new computations
#     c.execute('DELETE FROM subim1bank')
#     #subim1_bank = []
#     # The collection of pieces im2 that will act as windows against each subim1
#     c.execute('CREATE TABLE IF NOT EXISTS subim2bank(im_index INTEGER, subim2_ij TEXT)')
#     # delete current contents of SQL subim2bank table
#     c.execute('DELETE FROM subim2bank')
#     #subim2_bank = []
#     # pad im2 with its mean along each axis
#     im2_pad = np.pad(im2, ((subim2_pad, subim2_pad), (subim2_pad, subim2_pad), (0, 0)),
#                         'constant')
#     # Keeps track of subim number
#     im_index = 0
#     # Load up the subim1 and subim2 banks
#     for i in range(subim1_number_h):
#         for j in range(subim1_number_w):
#             # subim1_bank.append(im1[i * subim1_size:(i + 1) * subim1_size,
#                                 #j * subim1_size:(j + 1) * subim1_size,:])
#             subim1_ij = im1[i * subim1_size:(i + 1) * subim1_size,
#                                 j * subim1_size:(j + 1) * subim1_size,:].tostring()
#             c.execute("INSERT INTO subim1bank (im_index, subim1_ij) VALUES (?, ?)",
#                     (im_index, subim1_ij))
#         # (a,b) are the coordinates of the upper left point of the current subim1
#         a = i * subim1_size
#         b = j * subim1_size
#         # subim2_bank.append(im2_pad[a:(a + subim2_size), b:(b + subim2_size), :])
#         subim2_ij = im2_pad[a:(a + subim2_height), b:(b + subim2_width), :].tostring()
#         c.execute("INSERT INTO subim2bank (im_index, subim2_ij) VALUES (?, ?)",
#                     (im_index, subim2_ij))
#         conn.commit()
#         im_index += 1

#     c.close()
#     conn.close()

#     return subim1_number, subim2_height, subim2_width

if __name__ == "main":
    print("Don't run this as main.")