from __future__ import print_function
import keras
import sys
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model, load_model
from keras import backend as K

class disp_predictor(object):
    def __init__(self, model, s_lIm_ij, s_rIm_ij, ind):

        self.model = model
        self.s_lIm_size = s_lIm_ij.shape[1]
        self.s_lIm_ij = np.reshape(s_lIm_ij,(1,self.s_lIm_size,self.s_lIm_size,3))
        self.s_rIm_ij = np.reshape(s_rIm_ij,(1,self.s_lIm_size,self.s_lIm_size,3))
        self.ind = np.array(ind)

    def dual_conv_pred(self):
        temp = self.ind
        temp = temp.reshape(1,2)
        comparison = self.model.predict([self.s_lIm_ij, self.s_rIm_ij, temp])
        return [comparison[0][0], comparison[0][1]]
