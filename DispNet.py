import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Deconvolution2D
from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import UpSampling2D
from keras.layers import Input, Concatenate, Conv2D
from keras.models import Model, model_from_json
from keras.optimizers import *
import numpy as np


class DispNet:
    def __init__(self):
        super().__init__()
        self.model = None

    # Initialize the model
    def init_model(self):
        Input_2 = Input(shape=(128, 256, 2), name='Input_2')  # 6,768,384

        Convolution2D_1 = Conv2D(name='Convolution2D_1', nb_col=7, nb_filter=64, border_mode='same',
                                 activation='relu', nb_row=7)(Input_2)

        MaxPooling2D_1 = MaxPooling2D(name='MaxPooling2D_1')(Convolution2D_1)
        Convolution2D_3 = Convolution2D(name='Convolution2D_3', nb_col=5, nb_filter=128, border_mode='same',
                                        activation='relu', nb_row=5)(MaxPooling2D_1)
        MaxPooling2D_3 = MaxPooling2D(name='MaxPooling2D_3')(Convolution2D_3)
        Convolution2D_17 = Convolution2D(name='Convolution2D_17', nb_col=5, nb_filter=256, border_mode='same',
                                         activation='relu', nb_row=5)(MaxPooling2D_3)
        MaxPooling2D_8 = MaxPooling2D(name='MaxPooling2D_8')(Convolution2D_17)
        Convolution2D_4 = Convolution2D(name='Convolution2D_4', nb_col=3, nb_filter=256, border_mode='same',
                                        activation='relu', nb_row=3)(MaxPooling2D_8)
        Convolution2D_9 = Convolution2D(name='Convolution2D_9', nb_col=3, nb_filter=512, border_mode='same',
                                        activation='relu', nb_row=3)(Convolution2D_4)
        MaxPooling2D_5 = MaxPooling2D(name='MaxPooling2D_5')(Convolution2D_9)
        Convolution2D_10 = Convolution2D(name='Convolution2D_10', nb_col=3, nb_filter=512, border_mode='same',
                                         activation='relu', nb_row=3)(MaxPooling2D_5)
        Convolution2D_11 = Convolution2D(name='Convolution2D_11', nb_col=3, nb_filter=512, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_10)
        MaxPooling2D_6 = MaxPooling2D(name='MaxPooling2D_6')(Convolution2D_11)
        Convolution2D_12 = Convolution2D(name='Convolution2D_12', nb_col=3, nb_filter=512, border_mode='same',
                                         activation='relu', nb_row=3)(MaxPooling2D_6)
        Convolution2D_13 = Convolution2D(name='Convolution2D_13', nb_col=3, nb_filter=1024, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_12)
        MaxPooling2D_7 = MaxPooling2D(name='MaxPooling2D_7')(Convolution2D_13)
        Convolution2D_14 = Convolution2D(name='Convolution2D_14', nb_col=3, nb_filter=1024, border_mode='same',
                                         activation='relu', nb_row=3)(MaxPooling2D_7)
        Convolution2D_18 = Convolution2D(name='Convolution2D_18', nb_col=3, nb_filter=1, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_14)
        UpSampling2D_1 = UpSampling2D(name='UpSampling2D_1')(Convolution2D_18)
        Deconvolution2D_4 = Deconvolution2D(name='Deconvolution2D_4', nb_col=4, border_mode='same', strides=(2, 2),
                                            nb_filter=512, activation='relu', nb_row=4)(
            Convolution2D_14)  # output_shape=(None, 512, 24, 12),
        BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1')(Deconvolution2D_4)
        merge_2 = Concatenate(axis=3)([Convolution2D_12, BatchNormalization_1, UpSampling2D_1, ])
        Convolution2D_19 = Convolution2D(name='Convolution2D_19', nb_col=3, nb_filter=512, border_mode='same',
                                         activation='relu', nb_row=3)(merge_2)

        Convolution2D_20 = Convolution2D(name='Convolution2D_20', nb_col=3, nb_filter=1, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_19)
        UpSampling2D_2 = UpSampling2D(name='UpSampling2D_2')(Convolution2D_20)
        Deconvolution2D_5 = Deconvolution2D(name='Deconvolution2D_5', nb_col=4, nb_row=4, strides=(2, 2),
                                            padding='same',
                                            activation='relu', nb_filter=256)(Convolution2D_19)
        BatchNormalization_2 = BatchNormalization(name='BatchNormalization_2')(Deconvolution2D_5)
        merge_3 = Concatenate(axis=3)([Convolution2D_10, BatchNormalization_2, UpSampling2D_2])
        Convolution2D_21 = Convolution2D(name='Convolution2D_21', nb_col=3, nb_filter=256, border_mode='same',
                                         activation='relu', nb_row=3)(merge_3)
        Deconvolution2D_24 = Deconvolution2D(name='Deconvolution2D_24', nb_col=4, border_mode='same',
                                             strides=(2, 2), nb_filter=128, activation='relu', nb_row=4)(
            Convolution2D_21)  # output_shape=(None, 128, 96, 48)
        BatchNormalization_3 = BatchNormalization(name='BatchNormalization_3')(Deconvolution2D_24)
        Convolution2D_22 = Convolution2D(name='Convolution2D_22', nb_col=3, nb_filter=1, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_21)
        UpSampling2D_3 = UpSampling2D(name='UpSampling2D_3')(Convolution2D_22)
        merge_4 = Concatenate(axis=3)([Convolution2D_4, BatchNormalization_3, UpSampling2D_3, ])
        Convolution2D_23 = Convolution2D(name='Convolution2D_23', nb_col=3, nb_filter=128, border_mode='same',
                                         activation='relu', nb_row=3)(merge_4)
        Convolution2D_24 = Convolution2D(name='Convolution2D_24', nb_col=3, nb_filter=1, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_23)
        UpSampling2D_4 = UpSampling2D(name='UpSampling2D_4')(Convolution2D_24)
        Deconvolution2D_7 = Deconvolution2D(name='Deconvolution2D_7', nb_col=4, border_mode='same',
                                            strides=(2, 2), nb_filter=64, activation='relu', nb_row=4)(
            Convolution2D_23)  # output_shape=(None, 64, 192, 96)
        BatchNormalization_4 = BatchNormalization(name='BatchNormalization_4')(Deconvolution2D_7)
        merge_5 = Concatenate(axis=3)([MaxPooling2D_3, BatchNormalization_4, UpSampling2D_4])
        Convolution2D_25 = Convolution2D(name='Convolution2D_25', nb_col=3, nb_filter=64, border_mode='same',
                                         activation='relu', nb_row=3)(merge_5)
        Convolution2D_26 = Convolution2D(name='Convolution2D_26', nb_col=3, nb_filter=1, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_25)
        UpSampling2D_5 = UpSampling2D(name='UpSampling2D_5')(Convolution2D_26)
        Deconvolution2D_8 = Deconvolution2D(name='Deconvolution2D_8', nb_col=4, nb_row=4, strides=(2, 2),
                                            padding='same',
                                            activation='relu', nb_filter=32)(
            Convolution2D_25)  # output_shape=(None, 64, 384, 192)
        BatchNormalization_5 = BatchNormalization(name='BatchNormalization_5')(Deconvolution2D_8)
        merge_6 = Concatenate(axis=3)([MaxPooling2D_1, BatchNormalization_5, UpSampling2D_5])
        Convolution2D_27 = Convolution2D(name='Convolution2D_27', nb_col=3, nb_filter=32, border_mode='same',
                                         activation='relu', nb_row=3)(merge_6)
        Convolution2D_28 = Convolution2D(name='Convolution2D_28', nb_col=3, nb_filter=1, border_mode='same',
                                         activation='relu', nb_row=3)(Convolution2D_27)

        model = Model([Input_2], [Convolution2D_28])

        print(model.summary())
        self.model = model
        return model

    # Return the model optimizer
    def get_optimizer(self):
        return Adam()

    # Return the model Loss function
    def get_loss_function(self):
        return 'mean_squared_error'

    # Return the Batch size
    def get_batch_size(self):
        return 32

    # Return the default number of epochs
    def get_num_epoch(self):
        return 100

    # Load model and weights from disk
    def load_model_and_weight(self, model_name):
        # load model
        json_file = open(model_name + '.json', 'r')
        model = json_file.read()
        json_file.close()
        model = model_from_json(model)
        # load weights into model
        model.load_weights(model_name + ".h5")
        print("Loaded model from disk")
        self.model = model

    # Save model and weights into model directory
    def save_model_and_weight(self, model_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_name + '.h5')
        print("Saved model to disk")

    # Compile the model
    def compile(self):
        self.model.compile(optimizer=self.get_optimizer(), loss=self.get_loss_function(), metrics=['accuracy'])
        return self.model

    # Train the model
    def train(self, x, y, n_epoch=20, batch_size=1):
        self.model.fit([x], y, epochs=n_epoch, batch_size=batch_size, verbose=1)

    # Check the error rate on its input test data (x_test & y_test) and print the result in consule
    def get_error_rate(self, x_ts, y_ts):
        p = self.model.predict(x_ts, batch_size=self.get_batch_size(), verbose=0)
        mse = np.mean(np.square(y_ts - p))
        print("Error rate is " + str(mse))
        return mse
