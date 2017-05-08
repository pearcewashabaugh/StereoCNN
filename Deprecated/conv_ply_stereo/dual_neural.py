'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import sys
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import train_test_split

class dual_conv_net(object):
    def __init__(self, s_lIm_bank, s_rIm_pos_bank, s_rIm_neg_bank, ind):
        self.s_lIm_bank = s_lIm_bank
        self.s_rIm_pos_bank = s_rIm_pos_bank
        self.s_rIm_neg_bank = s_rIm_neg_bank
        self.s_lIm_size = s_lIm_bank.shape[2]

        self.num_classes = 2
        self.epochs = 12
        self.batch_size = 8

        self.x_train_l, self.x_train_r, self.y_train, \
            self.x_test_l, self.x_test_r, self.y_test = self.train_test_separator()
               
        self.dual_model = self.dual_conv_creator()

        self.dual_conv_train()

        self.model_saver(ind)

    def train_test_separator(self):
        x_train_l = self.s_lIm_bank[:20,:,:,:]
        x_train_l = np.append(x_train_l,x_train_l, axis = 0)
        x_train_r_pos = self.s_rIm_pos_bank[:20,:,:,:] 
        x_train_r_neg = self.s_rIm_neg_bank[:20,:,:,:]
        x_train_r = np.append(x_train_r_pos,x_train_r_neg, axis = 0)

        # x_trainneg = np.concatenate((x_train_l,x_train_r_neg), axis = 1)
        # x_trainpos = np.concatenate((x_train_l,x_train_r_pos), axis = 1)

        # x_train = np.concatenate(x_trainneg,x_trainpos)

        y_train_pos = np.ones((20))
        y_train_neg = np.zeros((20))
        y_train = np.append(y_train_neg, y_train_pos, axis = 0)

        x_test_l = self.s_lIm_bank[21:23,:,:,:]
        x_test_l = np.append(x_test_l,x_test_l,axis = 0)
        x_test_r = np.append(self.s_rIm_pos_bank[21:23,:,:,:] ,
                                  self.s_rIm_neg_bank[21:23,:,:,:], axis = 0)
        y_test_pos = np.ones((2))
        y_test_neg = np.zeros((2))
        y_test = np.append(y_test_neg,y_test_pos, axis = 0)

        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

        return x_train_l, x_train_l, y_train, x_test_l, x_test_r, y_test

    def dual_conv_creator(self):
        # First, define the vision modules
        digit_input = Input(shape=(self.s_lIm_size, self.s_lIm_size,3))
        x = Conv2D(128, (3, 3), activation='relu')(digit_input)
        x = Conv2D(64, (3, 3))(x)
        x = Dropout(0.25)(x)
        x = MaxPooling2D((2, 2))(x)
        out = Flatten()(x)
        
        vision_model = Model(digit_input, out)
        
        # Then define the tell-digits-apart model
        digit_a = Input(shape=(self.s_lIm_size, self.s_lIm_size,3))
        digit_b = Input(shape=(self.s_lIm_size, self.s_lIm_size,3))
        
        # The vision model will be shared, weights and all
        out_a = vision_model(digit_a)
        out_b = vision_model(digit_b)
        
        concatenated = keras.layers.concatenate([out_a, out_b])
        out = Dense(2, activation='sigmoid')(concatenated)
    
        classification_model = Model([digit_a, digit_b], out)

        classification_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

        return classification_model

    def dual_conv_train(self):
        self.dual_model.fit([self.x_train_l,self.x_train_r], self.y_train,
                        batch_size = self.batch_size,
                        epochs = self.epochs,
                        verbose = 1)
                        #validation_data = ([self.x_test_l,self.x_test_r], self.y_test))
        score = self.dual_model.evaluate([self.x_test_l,self.x_test_r], self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def model_saver(self, ind):
        i = ind[0]
        j = ind[1]
        self.dual_model.save('dual_neural_models\model_%d_%d' % (i,j))