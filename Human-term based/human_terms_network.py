from keras.layers import Input, Dense, TimeDistributed, Embedding
from keras.layers import Concatenate, Reshape, Lambda, Multiply, multiply, concatenate
from keras.models import Model
from keras import backend as K

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
        # self.base = self.build_base_model()


        self.base_combined, self.combined = self.build_combined_model()

        self.base_combined.compile(loss=self.loss_function,
                        optimizer=self.optimizer,
                        metrics=['mae', 'acc'])

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

    #input_shape, human_terms_shape
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

    
        