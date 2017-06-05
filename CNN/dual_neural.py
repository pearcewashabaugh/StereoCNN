'''Modified from keras MNIST example
'''

from __future__ import print_function
import keras
import sys
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Model
from keras import backend as K
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sqlite3

import Interface.interface as uinp

class dual_conv_net(object):
    def __init__(self, x_input, y_input, s_num_h, s_num_w):
        self.x_input = x_input
        self.y_input = y_input
        self.s_num_h = s_num_h
        self.s_num_w = s_num_w
        self.s_Im_size = x_input[0][0].shape[0]

        self.num_classes = 2

        # Parameters loaded from interface
        self.epochs = uinp.epochs
        self.batch_size = uinp.batch_size
        self.test_size = uinp.test_size
        self.random_state = uinp.random_state

        self.x_train_l, self.x_train_r, self.y_train, \
            self.x_test_l, self.x_test_r, self.y_test, \
            self.x_train_loc, self.x_test_loc = self.train_test_separator()
               
        self.dual_model = self.dual_conv_creator()

        self.dual_conv_train()

        self.model_saver()

    def train_test_separator(self):
        # x_input_l = self.s_lIm_bank
        # x_input_l = np.append(x_input_l,x_input_l, axis = 0)
        # x_input_r_pos = self.s_rIm_pos_bank
        # x_input_r_neg = self.s_rIm_neg_bank
        # x_input_r = np.append(x_input_r_pos,x_input_r_neg, axis = 0)
        # x_input_lab = np.append(s_Im_lab, s_Im_lab, axis = 0)
        # x_input_im = np.append(x_input_l, x_input_r, axis = 2)
        # x_input = [x_input_im,x_input_lab]
        ############################

        # x_input_pos = np.array(self.s_Im_pos_bank)
        # x_input_neg = np.array(self.s_Im_neg_bank)
        # x_input = np.append(x_input_pos,x_input_neg, axis = 0)

        ############################

        # x_input = self.s_Im_pos_bank + self.s_Im_neg_bank
        # x_trainneg = np.concatenate((x_train_l,x_train_r_neg), axis = 1)
        # x_trainpos = np.concatenate((x_train_l,x_train_r_pos), axis = 1)

        # x_train = np.concatenate(x_trainneg,x_trainpos)

        ############################
        # y_input_pos = np.ones((len(self.s_Im_pos_bank)))
        # y_input_neg = np.zeros((len(self.s_Im_pos_bank)))
        # y_input = np.append(y_input_neg, y_input_pos, axis = 0)
        ############################
        # y_input_pos = [1 for i in range(len(self.s_Im_pos_bank))]
        # y_input_neg = [0 for i in range(len(self.s_Im_neg_bank))]

        # y_input = y_input_pos + y_input_neg

        x_train, x_test, y_train, y_test = train_test_split(
            self.x_input, self.y_input, test_size = self.test_size, random_state = self.random_state)
        # x_test_l = self.s_lIm_bank[:,:,21:23,:,:,:]
        # x_test_l = np.append(x_test_l,x_test_l,axis = 0)
        # x_test_r = np.append(self.s_rIm_pos_bank[:,:,21:23,:,:,:] ,
        #                           self.s_rIm_neg_bank[:,:,21:23,:,:,:], axis = 0)
        # y_test_pos = np.ones((2))
        # y_test_neg = np.zeros((2))
        # y_test = np.append(y_test_neg,y_test_pos, axis = 0)
        # x_train_l = x_train[:,:,:x_input_l.shape[2],:]
        # x_train_r = x_train[:,:,x_input_l.shape[2]:,:]
        # x_test_l = x_test[:,:,x_input_l.shape[2]:,:]
        # x_test_r = x_test[:,:,:x_input_l.shape[2],:]

        x_train_l = np.array([k[0][:,:self.s_Im_size,:] for k in x_train])
        x_test_l = np.array([k[0][:,:self.s_Im_size,:] for k in x_test])
        x_train_r = np.array([k[0][:,self.s_Im_size:,:] for k in x_train])
        x_test_r = np.array([k[0][:,self.s_Im_size:,:] for k in x_test])
        x_train_loc = np.array([[k[1][0]/(self.s_num_h-1),k[1][1]/(self.s_num_w-1)] for k in x_train])
        x_test_loc = np.array([[k[1][0]/(self.s_num_h-1),k[1][1]/(self.s_num_w-1)] for k in x_test])


        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

        return x_train_l, x_train_l, y_train, x_test_l, x_test_r, y_test, x_train_loc, x_test_loc

    def dual_conv_creator(self):
        # First, define the vision modules
        subim_input = Input(shape=(self.s_Im_size, self.s_Im_size,3))
        x = Conv2D(64, (5, 5), activation='relu')(subim_input)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        out = Dropout(0.5)(x)
        #out = Flatten()(x)
        
        vision_model = Model(subim_input, out)
        
        # Then define the tell-digits-apart model
        subim_l = Input(shape=(self.s_Im_size, self.s_Im_size,3))
        subim_r = Input(shape=(self.s_Im_size, self.s_Im_size,3))
        subim_pos = Input(shape=(2,))
        
        # The vision model will be shared, weights and all
        out_a = vision_model(subim_l)
        out_b = vision_model(subim_r)
        
        concatenated = keras.layers.concatenate([out_a, out_b, subim_pos])
        concatenated = Dense(128, activation = 'relu')(concatenated)
        out = Dense(2, activation='sigmoid')(concatenated)
    
        classification_model = Model([subim_l, subim_r, subim_pos], out)

        #############################################
        # Choose method to compile model

        learning_rate = uinp.learning_rate
        #decay_rate = learning_rate/self.epochs
        decay_rate = uinp.decay_rate
        momentum = uinp.momentum
        sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        classification_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


        # classification_model.compile(loss=keras.losses.categorical_crossentropy,
        #       optimizer=keras.optimizers.Adadelta(),
        #       metrics=['accuracy'])

        return classification_model

    def dual_conv_train(self):
        self.dual_model.fit([self.x_train_l,self.x_train_r,self.x_train_loc], self.y_train,
                        batch_size = self.batch_size,
                        epochs = self.epochs,
                        verbose = 1,
                        validation_data = ([self.x_test_l,self.x_test_r,self.x_test_loc], self.y_test))
        score = self.dual_model.evaluate([self.x_test_l,self.x_test_r, self.x_test_loc], self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Save the network architecture to a json string
        json_model = self.dual_model.to_json()

        conn = sqlite3.connect('CNN/hyperparameters.db')
        c = conn.cursor()

        c.execute(
        'CREATE TABLE IF NOT EXISTS neural_scores(job_id INT, test_loss REAL, test_accuracy REAL)')

        c.execute("INSERT INTO neural_scores (job_id, test_loss, test_accuracy) VALUES (?,?,?)", 
            (uinp.job_id, score[0], score[1]))
        c.execute("UPDATE neural_params SET json_model = ? WHERE job_id = ?", 
            (json_model, uinp.job_id))
        conn.commit()
        conn.close()

    def model_saver(self):
        self.dual_model.save('CNN/dual_neural_models/model')