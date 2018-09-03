from keras.layers import Input, Dense, TimeDistributed, Embedding
from keras.layers import Concatenate, Reshape, Lambda, Multiply, multiply, concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class Human_Terms_Network():
    def __init__(self, input_shape, human_terms_shape, domain,
                loss_function='mse', 
                optimizer='adam', 
                trainable=True):
        self.input_shape = input_shape
        self.human_terms_shape = human_terms_shape
        self.loss_function = loss_function
        self.optimizer = optimizer

        # Build model here
        self.base_combined, self.combined = self.build_combined_model()

        self.base_combined.compile(loss=self.loss_function,
                        optimizer=self.optimizer,
                        metrics=['acc'])

        # set the trainable, whether train or not for the combined model. 
        self.base_combined.trainable = trainable

        self.combined.compile(loss=self.loss_function,
                        optimizer=self.optimizer,
                        metrics=['acc'])

        self.domain = domain
        

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
        #split = Lambda( lambda x: tf.split(x,num_or_size_splits=self.human_terms_shape,axis=1))(ht_input_layer)
        split = Lambda(self.layer_split)(ht_input_layer)

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
        concat = Lambda(self.layer_concat, name='concatenate')(dense_layer)
        output_layer = Dense(1, activation='sigmoid')(concat)

        # build model
        combined_model = Model(inputs=[combined_input_layer, ht_input_layer], outputs=output_layer)
        combined_model.summary()

        
        return base_model, combined_model
    
    def layer_split(self, x):
        return tf.split(x,num_or_size_splits=self.human_terms_shape,axis=1)

    def layer_concat(self, x):
        return tf.concat(x, axis=1)

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


    def train(self, epochs=10, verbose=1, batch_size=1, show_graph=True, save_model=True):
        # Train the base model first with target label [-1, 1]
        self.base_history = self.base_combined.fit(self.X_train, self.y_tanh_train,
                                                    validation_split=0.33, shuffle=True,
                                                    epochs=epochs, verbose=verbose, batch_size=batch_size)

        # Set check point for 

        if save_model:
            self.combined_path="./" + self.domain + "/combined-weight-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
            checkpoint = ModelCheckpoint(self.combined_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            self.callbacks_list = [checkpoint]
        else:
            self.callbacks_list = None

        self.combined_history = self.combined.fit([self.X_train, self.y_train_agreement], self.y_train, 
                                                    validation_split=0.33, shuffle=True,
                                                    epochs=epochs, verbose=verbose, batch_size=batch_size,
                                                    callbacks=self.callbacks_list)

        if show_graph:
            self.history_plot(self.base_history, 'base')
            self.history_plot(self.combined_history, 'combined')

    def test(self, reject=False):
        #Evaluate.

        if reject:
             # define model which get the input from combined model
             # output the value after relu
            self.human_terms_relu_model = Model(inputs=self.combined.input,
                                            outputs=self.combined.get_layer('concatenate').output)
            predict_relu = self.human_terms_relu_model.predict([self.X_test, self.y_test_agreement])
            accept_indices = np.where(np.sum(predict_relu, axis=1)!=0)
            accept_indices = accept_indices[0]
            total_reject = self.X_test.shape[0] - len(accept_indices)
            rejection_rate = total_reject/self.X_test.shape[0]

            test_eval = self.combined.evaluate([self.X_test[accept_indices], self.y_test_agreement[accept_indices]], self.y_test[accept_indices])
        else:
            # Test as usual
            rejection_rate = 0
            test_eval = self.combined.evaluate([self.X_test, self.y_test_agreement], self.y_test)
        
        return test_eval, rejection_rate

    def history_plot(self, history, model_name):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        
        title = model_name + 'accuracy'
        plt.title(title)
        plt.xlabel('epoch')
        plt.legend(['tr_acc', 'te_acc'], loc='upper left')
        plt.show()
        plt.clf()

        plt.plot(history.history['loss'], 'm--')
        plt.plot(history.history['val_loss'], 'y--')

        title = model_name + 'loss'
        plt.title(title)
        plt.xlabel('epoch')
        plt.legend(['tr_loss', 'te_loss'], loc='upper left')
        plt.show()
        plt.clf()