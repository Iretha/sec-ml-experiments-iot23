import logging
import time

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from keras.layers import Input, Dense
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split


# https://machinelearningmastery.com/autoencoder-for-classification/
# Multilayer Perceptron (MLP) autoencoder model.
def create_autoenc_feature_selection_model(X, y, epochs=20, batch_size=1000, model_dir="../models/", model_name="_Encoder.h5"):
    logging.info('------ Creating encoder model... ----')
    start_time = time.time()

    # number of input columns
    n_inputs = X.shape[1]
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    # define encoder
    visible = Input(shape=(n_inputs,))

    # encoder level 1
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    # bottleneck
    n_bottleneck = round(float(n_inputs) / 2.0)
    bottleneck = Dense(n_bottleneck)(e)

    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    # decoder level 2
    d = Dense(n_inputs * 2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')

    # if plot_model:  # plot the autoencoder
    # plot_model(model, model_dir + model_name + '_autoenc_compress.png', show_shapes=True)

    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=2,
                        validation_data=(X_test, X_test))

    # if plot_model:  # plot loss
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)
    # plot_model(encoder, model, model_dir + model_name + '_encoder_compress.png', show_shapes=True)

    # save the encoder to file
    model_path = model_dir + model_name
    encoder.save(model_path)

    logging.info('------ Feature Selection Model saved in: ' + model_path)
    exec_time_seconds = (time.time() - start_time)
    exec_time_minutes = exec_time_seconds / 60

    print("---> END in %s seconds = %s minutes ---" % (exec_time_seconds, exec_time_minutes))


def encode_data_with_autoencoder(X_train, y_train, X_test, y_test, encoder_path):
    logging.info('------ Encoding data with autoencoder... ----')

    # load encoder
    encoder = load_model(encoder_path)

    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    # X_test = t.transform(X_test)

    # encode the train data
    X_train_encode = encoder.predict(X_train)
    # encode the test data
    X_test_encode = encoder.predict(X_test)

    return X_train_encode, y_train, X_test_encode, y_test
