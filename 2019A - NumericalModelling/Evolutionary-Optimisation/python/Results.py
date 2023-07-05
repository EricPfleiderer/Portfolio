"""
Eric Pfleiderer (20048976)
PHY3075-Numerical modelling: Modelisation numerique en phyique
Projet 5: Mutation adaptive et orbite eta_Bootis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GenOpti import GenOpti


def best_of(number_attemps=25, number_kept=5, generations=500, mutation_type='standard'):

    results = np.zeros(0)

    for attempt in range(number_attemps):
        print(attempt)
        optimizer = GenOpti(random=True)
        optimizer.simulate_generations(generations, selective_pressure=1 / 4, p_c=0.8, p_m=0.1, mutation_type=mutation_type, max_generations=generations)
        winner = optimizer.population[0]
        results = np.append(results, 1/optimizer.evaluate_objective(winner.reshape(1, 6)))

    rank = np.argsort(results)

    return results[rank[:number_kept]]


def compete(number_attemps=25, number_kept=5, generations=500):

    mutation_types = ['standard', 'objective', 'distance', 'non_uniform']

    results = np.empty(len(mutation_types), dtype='object')

    for idx, mtype in enumerate(mutation_types):
        results[idx] = best_of(number_attemps, number_kept, generations, mtype)

    return results


def write_competition(path='results/', number_attemps=25, number_kept=25, generations=1000):

    results = compete(number_attemps, number_kept, generations)

    data = {
            'standard': results[0],
            'objective': results[1],
            'distance': results[2],
            'non_uniform:': results[3],
    }

    frame = pd.DataFrame(data=data)
    frame.to_csv(path+'competition_results.csv', index=False)


def read_competition(n=5, path='results/'):
    results = pd.read_csv(path+'competition_results.csv')

    for idx, key in enumerate(results.keys()):
        print(key)
        print('Total:', results[key][:n].sum())
        print('Average:', results[key][:n].sum()/n)
        print('Standard Err:', np.std(results[key][:n].to_numpy()))
        print('\n')


def draw_compare_adaptive(iterations=5, measure_density=25, path='', title=''):
    optimizer1 = GenOpti(random=True)
    optimizer2 = GenOpti(random=True)
    optimizer3 = GenOpti(random=True)
    error1 = np.zeros(0)
    error2 = np.zeros(0)
    error3 = np.zeros(0)
    error1 = np.append(error1, 1 / optimizer1.evaluate_objective(optimizer1.population[0].reshape(1, 6))[0])
    error2 = np.append(error2, 1 / optimizer2.evaluate_objective(optimizer2.population[0].reshape(1, 6))[0])
    error3 = np.append(error3, 1 / optimizer2.evaluate_objective(optimizer3.population[0].reshape(1, 6))[0])

    for iteration in range(iterations):
        print('Iterating:', iteration)
        optimizer1.simulate_generations(measure_density, selective_pressure=1 / 4, p_c=0.8, p_m=0.1,
                                        mutation_type='standard', max_generations=iterations*measure_density)
        optimizer2.simulate_generations(measure_density, selective_pressure=1 / 4, p_c=0.8, p_m=0.1,
                                        mutation_type='distance', max_generations=iterations*measure_density)
        optimizer3.simulate_generations(measure_density, selective_pressure=1 / 4, p_c=0.8, p_m=0.1,
                                        mutation_type='distance', max_generations=iterations*measure_density)
        error1 = np.append(error1, 1/optimizer1.evaluate_objective(optimizer1.population[0].reshape(1, 6))[0])
        error2 = np.append(error2, 1/optimizer2.evaluate_objective(optimizer2.population[0].reshape(1, 6))[0])
        error3 = np.append(error3, 1/optimizer3.evaluate_objective(optimizer3.population[0].reshape(1, 6))[0])

    x_axis = np.arange(0, measure_density*(iterations+1), measure_density)

    plt.figure()
    plt.plot(x_axis, error1, linestyle='--', marker='o', markersize=4, c='red', label=r'$mutations\ non\ adaptives$')
    plt.plot(x_axis, error2, linestyle='--', marker='o', markersize=4, c='green', label=r'$mutations\ adaptives$')
    plt.plot(x_axis, error3, linestyle='--', marker='o', markersize=4, c='green', label=r'$mutations\ adaptives$')
    plt.ylabel(r'$\chi^2$')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.subplots_adjust(left=0.1)
    plt.legend(loc='best')
    plt.savefig(path+title+'error_compare_adaptive'+str(iterations*measure_density) + '.pdf')


def draw_triple_error(iterations=5, measure_density=25, path=''):

    optimizer1 = GenOpti(random=False, mutation_size=1/4)
    optimizer2 = GenOpti(random=False, mutation_size=1/4)
    optimizer3 = GenOpti(random=False, mutation_size=1/4)

    error1 = np.zeros(0)
    error2 = np.zeros(0)
    error3 = np.zeros(0)
    error1 = np.append(error1, 1 / optimizer1.evaluate_objective(optimizer1.population[0].reshape(1, 6))[0])
    error2 = np.append(error2, 1 / optimizer2.evaluate_objective(optimizer2.population[0].reshape(1, 6))[0])
    error3 = np.append(error3, 1 / optimizer3.evaluate_objective(optimizer3.population[0].reshape(1, 6))[0])

    for iteration in range(iterations):
        print('Iterating:', iteration)
        optimizer1.simulate_generations(measure_density, selective_pressure=1 / 4, p_c=0.25, p_m=0.1, mutation_type='standard')
        optimizer2.simulate_generations(measure_density, selective_pressure=1 / 4, p_c=0.5, p_m=0.1, mutation_type='standard')
        optimizer3.simulate_generations(measure_density, selective_pressure=1 / 4, p_c=0.75, p_m=0.1, mutation_type='standard')
        error1 = np.append(error1, 1/optimizer1.evaluate_objective(optimizer1.population[0].reshape(1, 6))[0])
        error2 = np.append(error2, 1/optimizer2.evaluate_objective(optimizer2.population[0].reshape(1, 6))[0])
        error3 = np.append(error3, 1/optimizer3.evaluate_objective(optimizer3.population[0].reshape(1, 6))[0])

    x_axis = np.arange(0, measure_density*(iterations+1), measure_density)

    plt.figure()
    plt.plot(x_axis, error1, linestyle='--', marker='o', markersize=4, c='red', label=r'$P_c=\frac{1}{4}$')
    plt.plot(x_axis, error2, linestyle='--', marker='o', markersize=4, c='green', label=r'$P_c=\frac{1}{2}$')
    plt.plot(x_axis, error3, linestyle='--', marker='o', markersize=4, c='blue', label=r'$P_c=\frac{3}{4}$')
    plt.ylabel(r'$\chi^2$')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(path+'error_p_c'+str(iterations*measure_density) + '.pdf')


def draw_radial_velocity_Bootis(iterations=10, path='', mutation_type='standard'):

    optimizer = GenOpti(mutation_size=0.25)
    optimizer.simulate_generations(iterations, selective_pressure=1/4, p_c=0.8, p_m=0.1, mutation_type=mutation_type, max_generations=iterations)

    solution_vector = optimizer.population[0, :].reshape(1, 6)

    objective_value = round(optimizer.evaluate_objective(solution_vector)[0], 6)

    radial_velocity_pred = np.empty(0)
    time_disc_pred = np.append(np.arange(13000, 19000, 10), np.arange(27000, 33000, 10))
    mask_pred1 = np.where(np.logical_and(13000 <= time_disc_pred, time_disc_pred <= 19000))
    mask_pred2 = np.where(np.logical_and(27000 <= time_disc_pred, time_disc_pred <= 33000))

    for time in time_disc_pred:
        radial_velocity_pred = np.append(radial_velocity_pred, optimizer.evaluate_solution(solution_vector, time))

    time_disc_exp = optimizer.data.iloc[:, 0].values
    radial_velocity_exp = optimizer.data.iloc[:, 1].values
    std_error_exp = 1/optimizer.data.iloc[:, 2].values
    mask_exp1 = np.where(np.logical_and(13000 <= time_disc_exp, time_disc_exp <= 19000))
    mask_exp2 = np.where(np.logical_and(27000 <= time_disc_exp, time_disc_exp <= 33000))

    round_solution_vector = np.round(solution_vector, 3)

    label = '\n$P=$' + str(round_solution_vector[0][0]) + '\n$\\tau=$' + str(round_solution_vector[0][1]) + \
            r'\n$\omega=$' + str(round_solution_vector[0][2]) + '\n$e=$' + str(round_solution_vector[0][3]) + \
            '\n$K=$' + str(round_solution_vector[0][4]) + '\n$V_0=$' + str(round_solution_vector[0][5])

    plt.figure(figsize=(8, 4))
    plt.plot(time_disc_pred[mask_pred1], radial_velocity_pred[mask_pred1], label=label, linestyle='--', c='gray')
    plt.errorbar(time_disc_exp[mask_exp1], radial_velocity_exp[mask_exp1], yerr=std_error_exp[mask_exp1], linestyle='none',  fmt='o', c='black')
    plt.xlim([12500, 22000])
    plt.ylabel(r'$V_r\ [km\ s^{-1}]$')
    plt.xlabel(r'$t\ [\ J.D.]$')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(path+'nu_opti' + str(objective_value) + '.pdf')

    plt.figure(figsize=(8, 4))
    plt.plot(time_disc_pred[mask_pred2], radial_velocity_pred[mask_pred2], label=label, linestyle='--', c='gray')
    plt.errorbar(time_disc_exp[mask_exp2], radial_velocity_exp[mask_exp2], yerr=std_error_exp[mask_exp2], linestyle='none', fmt='o', c='black')
    plt.xlim([26500, 36000])
    plt.ylabel(r'$V_r\ [km\ s^{-1}]$')
    plt.xlabel(r'$t\ [\ J.D.]$')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(path+'nu_opti' + str(objective_value) + '.pdf')


def draw_radial_velocity_CBr(iterations=10, path='', mutation_type='standard'):

    optimizer = GenOpti(N=16, mutation_size=0.35, path='data/rho_Corona_Borealis.csv')
    optimizer.simulate_generations(iterations, selective_pressure=1/4, p_c=0.8, p_m=0.1, mutation_type=mutation_type, max_generations=iterations)

    solution_vector = optimizer.population[0, :].reshape(1, 6)

    objective_value = round(optimizer.evaluate_objective(solution_vector)[0], 6)

    radial_velocity_pred = np.empty(0)
    time_disc_pred = np.arange(0, 600, 0.1)

    for time in time_disc_pred:
        radial_velocity_pred = np.append(radial_velocity_pred, optimizer.evaluate_solution(solution_vector, time))

    time_disc_exp = optimizer.data.iloc[:, 0].values
    radial_velocity_exp = optimizer.data.iloc[:, 1].values
    std_error_exp = optimizer.data.iloc[:, 2].values

    round_solution_vector = np.round(solution_vector, 3)

    label = '\n$P=$' + str(round_solution_vector[0][0]) + '\n$\\tau=$' + str(round_solution_vector[0][1]) + \
            r'\n$\omega=$' + str(round_solution_vector[0][2]) + '\n$e=$' + str(round_solution_vector[0][3]) + \
            '\n$K=$' + str(round_solution_vector[0][4]) + '\n$V_0=$' + str(round_solution_vector[0][5])

    plt.figure(figsize=(8, 4))
    plt.plot(time_disc_pred, radial_velocity_pred, label=label, linestyle='--', c='gray')
    plt.errorbar(time_disc_exp, radial_velocity_exp, yerr=std_error_exp, linestyle='none',  fmt='o', c='black')
    plt.xlim([0, 800])
    plt.ylabel(r'$V_r\ [km\ s^{-1}]$')
    plt.xlabel(r'$t\ [\ J.D.]$')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(path+'opti' + str(objective_value) + '.pdf')


if __name__ == '__main__':
    for x in range(1):
        print('Simulation #', str(x))
        draw_radial_velocity_Bootis(iterations=100, path='results/std_', mutation_type='standard')
        draw_radial_velocity_Bootis(iterations=250, path='results/std_', mutation_type='standard')
