from keras.layers import Input, Dense, TimeDistributed, Embedding
from keras.layers import Concatenate, Reshape, Lambda, Multiply, multiply, concatenate
from keras.models import Model
from keras import backend as K

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class Human_Terms_Network():
    def __init__(self, input_shape, human_terms_shape, 
                loss_function='mse', 
                optimizer='Adagrad', 
                trainable=True):
        self.input_shape = input_shape
        self.human_terms_shape = human_terms_shape
        self.loss_function = loss_function
        self.optimizer = optimizer

        # Build model here
        self.base_combined, self.combined = self.build_combined_model()

        self.base_combined.compile(loss=self.loss_function,
                        optimizer=self.optimizer,
                        metrics=['mae', 'acc'])

        # set the trainable, whether train or not for the combined model. 
        self.base_combined.trainable = trainable

        self.combined.compile(loss=self.loss_function,
                        optimizer=self.optimizer,
                        metrics=['mae','acc'])

    def build_base_model(self):
        input_layer = Input(shape=(self.input_shape,))
        tanh_output = Dense(1, activation='tanh', name='tanh_output')(input_layer)
        
        model = Model(inputs=input_layer, outputs=tanh_output)
        model.summary()
        
        return model

    def build_combined_model(self):

        # input for base model
        base_model = self.build_base_model()
        combined_input_layer = Input(shape=(self.input_shape,))

        # build the hard coded weight for human terms and split the input 
        ht_input_layer = Input(shape=(self.human_terms_shape,))
        split = Lambda( lambda x: tf.split(x,num_or_size_splits=self.human_terms_shape,axis=1))(ht_input_layer)

        # get the document prediction
        label_layer = base_model(combined_input_layer)
        
        # multiply the predicion and the human terms absence -> pass it to relu
        dense_layer = []
        for i in range(self.human_terms_shape):
            dense_layer.append(Dense(
                1, 
                activation='relu', 
                use_bias=False, 
                kernel_initializer='ones')(Multiply()([split[i], label_layer])))

        # concat all the result and pass it to sigmoid layer
        concat = Lambda( lambda x: tf.concat(x, axis=1), name='concatenate')(dense_layer)
        output_layer = Dense(1, activation='sigmoid')(concat)

        # build model
        combined_model = Model(inputs=[combined_input_layer, ht_input_layer], outputs=output_layer)
        combined_model.summary()

        
        return base_model, combined_model
    
    def set_data(self, X_train, X_test, y_train_agreement, y_test_agreement, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_agreement = y_train_agreement
        self.y_test_agreement = y_test_agreement
        self.y_train = y_train
        self.y_test = y_test

        # set the y_train tanh
        self.y_tanh_train = self.y_train
        self.y_tanh_train[self.y_tanh_train == 0] = -1

        self.y_tanh_test = self.y_test
        self.y_tanh_test[self.y_tanh_test == 0] = -1


    def train(self, epochs=10, verbose=0, batch_size=1, show=True):
        # Train the base model first with target label [-1, 1]
        self.base_history = self.base_combined.fit(self.X_train, self.y_tanh_train,
                                                    validation_data=(self.X_test, self.y_test),
                                                    epochs=epochs, verbose=verbose, batch_size=batch_size)

        self.combined_history = self.combined.fit([self.X_train, self.y_train_agreement], self.y_train, 
                                                    validation_data=([self.X_test, self.y_test_agreement], self.y_test),
                                                    epochs=epochs, verbose=verbose, batch_size=batch_size)

        if show:
            self.history_plot(self.base_history, 'base')
            self.history_plot(self.combined_history, 'combined')

    def history_plot(self, history, model_name):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.plot(history.history['loss'], 'm--')
        plt.plot(history.history['val_loss'], 'y--')

        plt.title('model loss history')
        plt.xlabel('epoch')
        plt.legend(['tr_acc', 'te_acc', 'tr_loss', 'te_loss'], loc='upper left')
        plt.show()
        plt.savefig('./',model_name,'.png')
        plt.clf()

