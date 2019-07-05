import matplotlib.pyplot as plt
import numpy as np

from PHY3075.PROJET5.python.GenOpti import GenOpti

def draw_radial_velocity(iterations=30):

    optimizer = GenOpti()
    optimizer.simulate_generations(iterations)

    #solution_vector = np.array([494.2, 14299, 5.7397, 0.2626, 8.3836, 1.0026]).reshape(1, 6)
    solution_vector = optimizer.population[0, :].reshape(1, 6)

    radial_velocity_pred = np.empty(0)
    time_disc_pred = np.append(np.arange(13000, 19000, 10), np.arange(27000, 33000, 10))
    mask_pred1 = np.where(np.logical_and(13000 <= time_disc_pred, time_disc_pred <= 19000))
    mask_pred2 = np.where(np.logical_and(27000 <= time_disc_pred, time_disc_pred <= 33000))

    for time in time_disc_pred:
        radial_velocity_pred = np.append(radial_velocity_pred, optimizer.evaluate_solution(solution_vector, time))

    time_disc_exp = optimizer.data.iloc[:, 0].values
    radial_velocity_exp = optimizer.data.iloc[:, 1].values
    mask_exp1 = np.where(np.logical_and(13000 <= time_disc_exp, time_disc_exp <= 19000))
    mask_exp2 = np.where(np.logical_and(27000 <= time_disc_exp, time_disc_exp <= 33000))

    round_solution_vector = np.round(solution_vector, 2)

    label='\n$P=$' + str(round_solution_vector[0][0]) + '\n$\\tau=$' + str(round_solution_vector[0][1]) + \
          '\n$\omega=$' + str(round_solution_vector[0][2]) + '\n$e=$' + str(round_solution_vector[0][3]) + \
          '\n$K=$' + str(round_solution_vector[0][4]) + '\n$V_0=$' + str(round_solution_vector[0][5])

    plt.figure(figsize=(8, 4))
    plt.plot(time_disc_pred[mask_pred1], radial_velocity_pred[mask_pred1], label=label)
    plt.scatter(time_disc_exp[mask_exp1], radial_velocity_exp[mask_exp1], label='mesures')
    plt.xlim([12500, 22000])
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig('results/data_fitting1.pdf')

    plt.figure(figsize=(8, 4))
    plt.plot(time_disc_pred[mask_pred2], radial_velocity_pred[mask_pred2], label=label)
    plt.scatter(time_disc_exp[mask_exp2], radial_velocity_exp[mask_exp2], label='mesures')
    plt.xlim([26500, 37000])
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig('results/data_fitting2.pdf')


draw_radial_velocity()
