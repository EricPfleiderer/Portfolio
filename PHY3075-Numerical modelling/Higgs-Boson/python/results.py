"""
Autheur: Eric Pfleiderer (20048976)
Remise: 04/30/19
Cours: PHY3075-Numerical modelling - Modélisation numérique en physique
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from FFNN.higgs_boson.sequential import get_model
from FFNN.higgs_boson.dataprocessing import load_data_sets

# Data
x_train, y_train, x_test, y_test = load_data_sets()
optimizers = np.array(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
losses = np.empty(optimizers.size, dtype='object')
train_accs = np.empty(optimizers.size, dtype='object')
test_accs = np.empty(optimizers.size, dtype='object')


def print_model():
    """
    Build model and print its plot to file
    :return:void
    """
    model = get_model()
    plot_model(model, to_file='results/model_plot.png', show_shapes=True, show_layer_names=False)


def decayed_lr(lr=1e-5, decay_rate=0.95, global_steps=100, decay_steps=5):
    """
    Describes exponential growth of learning rate
    :param lr: initial learning rate
    :param decay_rate: rate of decay per epoch
    :param global_steps: current step
    :param decay_steps: attenuation factor
    :return: current learning rate
    """
    return lr*decay_rate**(global_steps/decay_steps)


def loss_lr(total_steps=100):
    """
    Print graph of loss vs learning rate for exponential learning rate
    :param total_steps: total epochs
    :return: void
    """

    learning_rates = np.zeros(total_steps)
    losses = np.zeros(total_steps)

    model = get_model()

    for x in range(total_steps):
        learning_rates[x] = decayed_lr(lr=1e-7, decay_rate=1.1, global_steps=x, decay_steps=1)

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rates[np.nonzero(learning_rates)][-1]),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        losses[x] = model.fit(x_train, y_train, epochs=1).history['loss'][0]

        print(learning_rates)

    plt.figure(figsize=(8, 6))
    plt.plot(learning_rates, losses)
    plt.xlabel('Rythme d\'apprentissage')
    plt.ylabel('Perte')
    plt.xscale('log')
    plt.axvline(x=1e-5, c='black', ls='--')
    plt.axvline(x=2*1e-4, c='black', ls='--')
    plt.savefig('results/acc_lr.pdf')


def compare_optimizers():
    """
    Draw graphs of acc vs epoch, loss vs epoch and final acc on testing set for all Keras optimizers
    :return: void
    """
    for idx, optimizer in enumerate(optimizers):
        # np.random.seed(1)
        model = get_model()

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.load_weights('models/model_seed.h5')

        initial_results = model.evaluate(x_test, y_test)

        history = model.fit(x_train, y_train, epochs=50, batch_size=64)
        losses[idx] = np.append(np.array([initial_results[0]]), np.array(history.history['loss']))
        train_accs[idx] = np.append(np.array([initial_results[1]]), np.array(history.history['acc']))

        final_results = model.evaluate(x_test, y_test)
        test_accs[idx] = final_results[1]

    xAxis = np.arange(0, len(losses[0]), 1)

    plt.figure()
    for idx, loss in enumerate(losses):
        plt.plot(xAxis, loss, label=optimizers[idx])
    plt.ylabel('Perte')
    plt.xlabel('Époque')
    plt.legend(loc='best')
    plt.savefig('results/compare_loss.pdf')

    plt.figure()
    for idx, acc in enumerate(train_accs):
        plt.plot(xAxis, acc, label=optimizers[idx])
    plt.ylabel('Précision')
    plt.xlabel('Époque')
    plt.legend(loc='best')
    plt.savefig('results/compare_train_acc.pdf')

    plt.figure()
    plt.bar(np.arange(0, test_accs.size, 1), test_accs)
    plt.xticks([x for x in range(optimizers.size)], optimizers, rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('Précision')
    plt.xlabel('Optimiseurs')
    plt.savefig('results/compare_test_acc.pdf')


def get_confusion_matrix(quantity=135):

    """
    Get the average confusion matrix on test set for multiple predictions
    :param quantity: number of models used for predictions
    :return: average confusion matrix
    """

    matrixes = np.empty(quantity, dtype=object)

    for x in range(quantity):
        model = get_model()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.load_weights('models/weights' + str(x) + '.h5')

        predictions = model.predict(x_test)

        print(x)

        matrixes[x] = confusion_matrix(y_test, np.round(predictions).astype(dtype=int))

    return np.sum(matrixes)/quantity


def get_signal_distribution(quantity=135):

    """
    Get average signal distribution for multiple predictions for true positives and true negatives
    :param quantity: number of models used for predictions
    :return: average signal distribution for 0 and 1 class
    """

    negative = np.empty(quantity, dtype=object)
    positive = np.empty(quantity, dtype=object)

    for x in range(quantity):

        print(x)
        model = get_model()
        model.load_weights('models/weights' + str(x) + '.h5')

        predictions = model.predict(x_test)

        mask_neg = np.where(y_test[:] == 0)
        mask_pos = np.where(y_test[:] == 1)

        negative[x] = predictions[mask_neg]
        positive[x] = predictions[mask_pos]

    negative = np.average(negative)
    positive = np.average(positive)

    return np.histogram(positive), np.histogram(negative)


def graph_distribution(quantity=135):

    """
    Graph signal distribution for multiple predictions for true positives and true negatives
    :param quantity: number of models used for predictions
    :return: void
    """

    pos, neg = get_signal_distribution(quantity=quantity)

    print(pos[0], neg[0])
    print(np.arange(0, 10, 1))

    plt.figure()
    plt.bar(np.arange(0, 10, 1), pos[0])
    plt.xticks([x-0.5 for x in range(10)], [str(x/10) for x in range(0, 10, 1)], rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.axvline(x=4.5, ls='--', c='black')
    plt.ylabel('Décompte')
    plt.xlabel('Signal de sortie')
    plt.savefig('results/pos_distri.pdf')

    plt.figure()
    plt.bar(np.arange(0, 10, 1), neg[0])
    plt.xticks([x-0.5 for x in range(10)], [str(x/10) for x in range(0, 10, 1)], rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.axvline(x=4.5, ls='--', c='black')
    plt.ylabel('Décompte')
    plt.xlabel('Signal de sortie')
    plt.savefig('results/neg_distri.pdf')


def print_classification():

    """
    Classify class_set using multiple saved models. Save results to txt file.
    :return: void
    """
    class_data = preprocessing.scale(pd.read_csv('data/class_set.csv', sep=r'\s+').values)

    model = get_model()

    total_models = 135

    for x in range(total_models):
        model.load_weights('models/weights'+str(x)+'.h5')

        predictions = np.squeeze(np.int32(np.round(model.predict(class_data))))

        np.savetxt('results/predictions'+str(x)+'.txt', predictions, fmt='%i')

    data = np.empty(total_models, dtype=object)

    for x in range(data.size):
        data[x] = np.loadtxt('results/predictions'+str(x)+'.txt')

    print(np.sum(data))

    data = np.sum(data)/data.size

    data = np.round(data)

    np.savetxt('results/predictions_final.txt', data, fmt='%i')
