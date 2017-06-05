import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.decomposition import PCA
import CNN.disp_predict as dsp
from Image_PrePost_Processing.image_preprocessor import image_resizer, subim_maker
from keras.models import Model, load_model
from sklearn.decomposition import PCA
import sys

def evaluator(s_lIm_size, s_rIm_pad, pred_samp_w, lIm, rIm):
    
    s_lIm_bank, s_rIm_bank, s_num_h, s_num_w = subim_maker(lIm, rIm, s_lIm_size, s_rIm_pad)        
    s_num = s_num_h * s_num_w
    
    pred_stride = int(s_rIm_pad / float(pred_samp_w))
    
    prediction_arr = np.zeros((s_num_h,s_num_w,1+pred_samp_w,2))
    # for i in range(s_num_h):
    #     for j in range(s_num_w):
    #         print("predicting for subimage ", (i,j))
    #         for k in range(pred_samp_w+1):
    #             which = dsp.disp_predictor(s_lIm_bank[i*s_num_w + j], 
    #                 s_rIm_bank[i*s_num_w + j][:,k*pred_stride:k*pred_stride+s_lIm_size,:], (i,j))
    #             prediction_arr[i,j,k,:] = which.dual_conv_pred()
    model = load_model('CNN/dual_neural_models/model')
    ###########################################################################
    print("Whitening Evaluation Data...")
    if s_lIm_size*2*s_lIm_size*3 > s_num_h * s_num_w * (pred_samp_w+1):
        print("Not enough samples. Increase sampling rate of right image.")
        sys.exit()
    pca = PCA(n_components = s_lIm_size*2*s_lIm_size*3, whiten = True)

    x_flat = np.zeros((len(s_lIm_bank)*(pred_samp_w+1), s_lIm_size*2*s_lIm_size*3))

    for i in range(s_num_h):
        for j in range(s_num_w):
            for k in range(pred_samp_w+1):
                temp = np.append(s_lIm_bank[i*s_num_w + j],
                    s_rIm_bank[i*s_num_w + j][:,k*pred_stride:k*pred_stride+s_lIm_size,:],
                    axis=1)
                x_flat[(i*s_num_w+j)*(pred_samp_w+1)+k,:] = temp.reshape(s_lIm_size*2*s_lIm_size*3)

    x_flat = pca.fit_transform(x_flat)

    eval_white_input = []
    for iindex in range(len(s_lIm_bank)*(pred_samp_w+1)):
        eval_white_input.append(np.reshape(x_flat[iindex,:],(s_lIm_size, 2*s_lIm_size, 3)))

    s_im_left_white_bank = []
    s_im_right_white_bank = []

    for i in range(s_num_h):
        for j in range(s_num_w):
            s_im_left_white_bank.append(eval_white_input[(i*s_num_w + j)*(pred_samp_w+1)][:,:s_lIm_size,:]) 
            for k in range(pred_samp_w+1):
                s_im_right_white_bank.append(eval_white_input[(i*s_num_w + j)*(pred_samp_w+1)+k][:,:s_lIm_size,:])
    ###########################################################################
    # # Evaluation code without whitening:
    # for i in range(s_num_h):
    #     print("predicting for subimage ", i*s_num_w ,"out of %d" % (s_num_h*s_num_w))
    #     for j in range(s_num_w):
    #         for k in range(pred_samp_w+1):
    #             which = dsp.disp_predictor(model, s_lIm_bank[i*s_num_w + j], 
    #                 s_rIm_bank[i*s_num_w + j][:,k*pred_stride:k*pred_stride+s_lIm_size,:], [i,j])
    #             prediction_arr[i,j,k,:] = which.dual_conv_pred()
    for i in range(s_num_h):
        print("predicting for subimage ", i*s_num_w ,"out of %d" % (s_num_h*s_num_w))
        for j in range(s_num_w):
            for k in range(pred_samp_w+1):
                which = dsp.disp_predictor(model, s_im_left_white_bank[i*s_num_w + j], 
                    s_im_right_white_bank[(i*s_num_w + j)*(pred_samp_w+1)+k], [i,j])
                prediction_arr[i,j,k,:] = which.dual_conv_pred()


    pred_fin = np.zeros((s_num_h,s_num_w))
    for i in range(s_num_h):
        for j in range(s_num_w):
            ijmax = np.argmax(prediction_arr[i,j,:,1])
            pred_fin[i,j] = 10 - pred_stride * ijmax
    
    conn = sqlite3.connect('CNN/image_quantities.db')
    c = conn.cursor()
    
    c.execute('CREATE TABLE IF NOT EXISTS offset_map(data TEXT, shape0 INT, shape1 INT)')
    c.execute('DELETE FROM offset_map')
    c.execute("INSERT INTO offset_map (data, shape0, shape1) VALUES (?,?,?)", (pred_fin.tostring(),\
        pred_fin.shape[0], pred_fin.shape[1]))
    conn.commit()
    conn.close()
    
    plt.figure()
    plt.title('Displacement Distance')
    plt.imshow(pred_fin, cmap='hot', interpolation='nearest')
    plt.show()