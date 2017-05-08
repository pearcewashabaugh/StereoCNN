import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import CNN.disp_predict as dsp
from Image_PrePost_Processing.image_preprocessor import image_resizer, subim_maker
from keras.models import Model, load_model

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

    for i in range(s_num_h):
        for j in range(s_num_w):
            print("predicting for subimage ", (i,j))
            for k in range(pred_samp_w+1):
                which = dsp.disp_predictor(model, s_lIm_bank[i*s_num_w + j], 
                    s_rIm_bank[i*s_num_w + j][:,k*pred_stride:k*pred_stride+s_lIm_size,:], [i,j])
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