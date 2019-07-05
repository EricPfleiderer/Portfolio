"""
Autheur: Eric Pfleiderer (20048976)
Remise: 04/30/19
Cours: PHY3075-Numerical modelling - Modélisation numérique en physique
"""

import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense, Dropout
from FFNN.higgs_boson.dataprocessing import load_data_sets

# Data
x_train, y_train, x_test, y_test = load_data_sets()


def gen_and_save_models(quantity=250):
    """
    This function generates and saves models according to get_model architecture
    :param quantity: number of models to generate and save
    :return: void
    """

    for x in range(quantity):
        model = get_model()

        optimizer = tf.keras.optimizers.Adam(lr=5e-5, decay=5e-4)

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        callback = [tf.keras.callbacks.EarlyStopping(patience=5)]

        _ = model.fit(x_train, y_train, epochs=150, batch_size=50, validation_data=(x_test, y_test), callbacks=callback, verbose=2)
        results = model.evaluate(x_test, y_test)

        model.save_weights('models/weights'+str(results[1])+'.h5')

        print(results)


def get_model():
    """
    Builds and returns a feed forward neural net
    :return: keras sequential model
    """

    model = tf.keras.Sequential()
    model.add(Dense(250, input_dim=13, activation=tf.nn.relu))
    model.add(Dropout(0.4))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation=tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation=tf.nn.relu))
    model.add(Dense(1, activation=tf.nn.sigmoid))
    return model
