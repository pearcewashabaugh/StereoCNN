import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import disp_predict as dsp
from image_preprocessor import image_resizer, subim_maker

pic_h_pix = 100
pic_w_pix = int(pic_h_pix*(1+1/3.0))

# The size of a subimage of the left picture (in pixels)
s_lIm_size = 20

# The s_rIm_pad + s_lIm_size = s_rIm_w (the right subim width)
s_rIm_pad = 10

# The number of samples to take along the width of each s_rIm. Must divide
# s_rIm_pad
pred_samp_w = 5

# Resize images, output left image = lIm and right image = rIm
lIm = image_resizer("../Data/Pictures_Data/left.jpeg",
                        pic_h_pix,
                        pic_width_pix = pic_w_pix)
rIm = image_resizer("../Data/Pictures_Data/right.jpeg",
                        pic_h_pix,
                        pic_width_pix = pic_w_pix)
# Split images up into subimages
s_lIm_bank, s_rIm_bank, s_num_h, s_num_w = subim_maker(lIm, rIm, s_lIm_size, s_rIm_pad)

s_num = s_num_h * s_num_w

# for i in range(s_num_h):
#     for j in range(s_num_w):
#         print("prediction ",(i,j))
#         which = dsp.disp_predictor(s_lIm_bank[i*s_num_w + j], 
#                                     s_rIm_bank[i*s_num_w + j], (i,j))
#         print(which.dual_conv_pred())


# for k in range(s_num):
#     if k % 100 == 0:
#         print("Computing l2 differences of subpictures and filters: %d/%d" % (k, s_num))
#     diff_mat = np.zeros((subim2_size - subim1_size, subim2_size - subim1_size, 3))
#     for i in range(subim2_size - subim1_size):
#         for j in range(subim2_size - subim1_size):
#             temp = subim2_bank[k, :, :, :]
#             tempdiff = temp[i:i + subim1_size, j:j + subim1_size, :] - subim1_bank[k, :, :, :]
#             diff_mat[i, j, :] = np.linalg.norm(tempdiff, axis=(0, 1))

pred_stride = int(s_rIm_pad / float(pred_samp_w))

prediction_arr = np.zeros((s_num_h,s_num_w,1+pred_samp_w,2))
for i in range(s_num_h):
    for j in range(s_num_w):
        print("predicting for subimage ", (i,j))
        for k in range(pred_samp_w+1):
            which = dsp.disp_predictor(s_lIm_bank[i*s_num_w + j], 
                s_rIm_bank[i*s_num_w + j][:,k*pred_stride:k*pred_stride+s_lIm_size,:], (i,j))
            prediction_arr[i,j,k,:] = which.dual_conv_pred()

pred_fin = np.zeros((s_num_h,s_num_w))
for i in range(s_num_h):
    for j in range(s_num_w):
        ijmax = np.argmax(prediction_arr[i,j,:,1])
        pred_fin[i,j] = 10 - pred_stride * ijmax

print(pred_fin.shape[0])
print(pred_fin.shape[1])
conn = sqlite3.connect('image_quantities.db')
c = conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS offset_map(data TEXT, shape0 INT, shape1 INT)')
c.execute('DELETE FROM offset_map')
c.execute("INSERT INTO offset_map (data, shape0, shape1) VALUES (?,?,?)", (pred_fin.tostring(),\
    pred_fin.shape[0], pred_fin.shape[1]))
conn.commit()
#c.close()
conn.close()


#np.savetxt('disp_fin.t')
plt.figure()
plt.title('Displacement Distance')
plt.imshow(pred_fin, cmap='hot', interpolation='nearest')
plt.show()