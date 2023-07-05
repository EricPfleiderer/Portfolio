import pandas as pd
import numpy as np
from scipy.optimize import bisect

class GenOpti:

    #N must be even
    def __init__(self, N=10, path='data/eta_Bootis.csv'):
        self.N = N
        self.data = pd.read_csv(path)
        self.population = self.initialize_random() #P, tau, omega, e, K, V0


    #Initialize population of N solution vectors according to specific intervals obtained from data analysis
    def initialize_random(self):

        population = np.zeros((self.N, 6))

        for n in range(self.N):
            population[n][0] = np.random.uniform(200, 800) #P
            population[n][1] = np.random.uniform(self.data.iloc[0, 0], self.data.iloc[0, 0] + population[n][0]) #tau
            population[n][2] = np.random.uniform(0, 2*np.pi) #omega
            population[n][3] = np.random.uniform(0, 1) #e
            population[n][4] = np.random.uniform(0, self.data['radial_velocity'].max() - self.data['radial_velocity'].min()) #K
            population[n][5] = np.random.uniform(self.data['radial_velocity'].min(), self.data['radial_velocity'].max()) #V0

        return population

    def simulate_generations(self, generations=1, selective_pressure=1/4, p_c=0.8, p_m=0.1, mutation_amp=50):
        pass
        #for generation in range(generations):


    #TO CHECK FOR BUGS
    def generate_kids(self, parents_idx, p_c=0.8):
        pass


    #Returns index of chosen parents from rank
    def generate_parents(self, rank, selective_pressure=1/5):
        # Choose best parent randomly according to selective pressure
        best_index = rank[np.random.randint(int((1 - selective_pressure) * self.N), self.N)]

        # Choose second parent
        second_index = best_index
        while second_index == best_index:
            second_index = np.random.randint(0, self.N - 1)

        return best_index, second_index



    #Evaluate objective function for each individual (solution matrix must be n by 6 size or 6 length list)
    def evaluate_objective(self, solution_matrix):
        chi = 0
        for idx, row in self.data.iterrows():
            chi += (row[1] - self.evaluate_solution(solution_matrix, row[0])/row[2])**2
        chi *= 1/(self.data.shape[0])

        return 1/chi

    #Evaluate solution at specific time
    def evaluate_solution(self, solution_matrix, t):

        E = np.empty(0)

        if len(solution_matrix.shape) is 1:
            P = solution_matrix[:, 0]
            tau = solution_matrix[:, 1]
            omega = solution_matrix[:, 2]
            e = solution_matrix[:, 3]
            K = solution_matrix[:, 4]
            V0 = solution_matrix[:, 5]

        else:
            P = solution_matrix[:, 0]
            tau = solution_matrix[:, 1]
            omega = solution_matrix[:, 2]
            e = solution_matrix[:, 3]
            K = solution_matrix[:, 4]
            V0 = solution_matrix[:, 5]

        #Solve keplers equation for the root
        for n in range(P.shape[0]):
            E = np.append(E, bisect(self.evaluate_kepler, -1000, 1000, args=(t, P[n], tau[n], e[n])))

        #Solve for projected velocity v
        orbital_velocity = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2)) #CHECK FOR ATAN2 ERROR ? ONLY 1 INPUT AVAILABLE

        #Solve for radial velocity
        radial_velocity = V0 + K * (np.cos(omega + orbital_velocity) + e*np.cos(omega))

        return radial_velocity

    #Keplers equation
    def evaluate_kepler(self, E, t, P, tau, e):
        return E - e * np.sin(E) - (2*np.pi/P) * (t-tau)


test = GenOpti()
print(test.population)
print(test.evaluate_objective(test.population))
print(test.evaluate_objective(np.array([[494.2, 14299, 5.7397, 0.2626, 8.3836, 1.0026]])))
