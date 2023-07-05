import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import bisect

'''
BUGS:
    -NaNs everywhere when e<-1 or e>1, must artificially contain e to [0,1]??
'''


'''
TO DO:
    -Implement adaptive mutation step

'''

class GeneticOptimisation:

    #N must be even
    def __init__(self, N=10, path='data/eta_Bootis.csv'):
        self.N = N
        self.data = pd.read_csv(path)
        self.population = self.initialize_random() #6 by N shape

    def simulate_generations(self, generations=1, selective_pressure=1/4, p_c=0.8, p_m=0.1, mutation_amp=50):
        #print(self.population)

        for generation in range(generations):
            P = self.population[0]
            tau = self.population[1]
            omega = self.population[2]
            e = self.population[3]
            K = self.population[4]
            V0 = self.population[5]

            #Evaluate current solutions
            current_solution = self.evaluate_objective(P, tau, omega, e, K, V0)

            #Rank solutions
            rank = np.argsort(current_solution) #Contains index of current solution elements (from smallest to lowest)

            #Select N new solutions, repetitions allowed (parents)
            parents = np.empty(0)

            for n in range(self.N//2):
                best_parent, second_parent = self.generate_parents(rank, selective_pressure)
                parents = np.append(parents, best_parent)
                parents = np.append(parents, second_parent)
                parents = np.int32(parents)

            #Generate new children
            children = self.generate_kids(parents, p_c)

            #Adjust mutation

            '''
            print(self.population[5][rank[self.N-1]])

            best_sol = self.evaluate_chi_squared(self.population[0][rank[self.N-1]], self.population[1][rank[self.N-1]],
                                                 self.population[2][rank[self.N-1]], self.population[3][rank[self.N-1]],
                                                 self.population[4][rank[self.N-1]], self.population[5][rank[self.N-1]])
            median_sol = self.evaluate_chi_squared(self.population[0][rank[self.N//2]], self.population[1][rank[self.N//2]],
                                                   self.population[2][rank[self.N//2]], self.population[3][rank[self.N//2]],
                                                   self.population[4][rank[self.N//2]], self.population[5][rank[self.N//2]])

            difference = best_sol - median_sol

            if abs(difference) <= best_sol*0.1:
                mutation_amp *=2

            '''

            #Mutate children if need be
            for n in range(self.N):
                for component in range(6):
                    if np.random.uniform(0, 1) <= p_m:
                        children[component][n] += np.random.normal(0, mutation_amp)

            # Preserve best solution (elitism) and update current population
            top_sol = self.population[:, rank[self.N-1]]
            self.population = children
            self.population[:, 0] = top_sol

    #TO CHECK FOR BUGS
    def generate_kids(self, parents_idx, p_c=0.8):

        r = np.empty(0)

        #Generate a ratio for every vector component
        for n in range(6):
            r = np.append(r, np.random.uniform(0, 1))

        children = np.empty((6, self.N))

        #For every couple of parents
        for n in range(self.N//2):

            #Choose two parent vectors
            parent1_sol = self.population[:, parents_idx[2*n]]
            parent2_sol = self.population[:, parents_idx[2*n+1]]

            #Generate 2 new solutions vectors
            for component in range(parent1_sol.size):
                if np.random.uniform(0, 1) <= p_c:
                    children[component][2*n] = r[component] * parent1_sol[component] + (1-r[component]) * parent2_sol[component]
                    children[component][2*n+1] = r[component] * parent1_sol[component] + (1-r[component]) * parent2_sol[component]
                else:
                    children[component][2*n] = parent1_sol[component]
                    children[component][2*n+1] = parent2_sol[component]

        return children


    #Returns index of chosen parents from rank
    def generate_parents(self, rank, selective_pressure=1/5):
        # Choose best parent randomly according to selective pressure
        best_index = rank[np.random.randint(int((1 - selective_pressure) * self.N), self.N)]

        # Choose second parent
        second_index = best_index
        while second_index == best_index:
            second_index = np.random.randint(0, self.N - 1)

        return best_index, second_index

    #Initialize population of N solution vectors according to specific intervals obtained from data analysis
    def initialize_random(self):

        population = np.zeros((6, self.N))

        for n in range(self.N):
            population[0][n] = np.random.uniform(200, 800) #P
            population[1][n] = np.random.uniform(self.data.iloc[0, 0], self.data.iloc[0, 0] + population[0][n]) #tau
            population[2][n] = np.random.uniform(0, 2*np.pi) #omega
            population[3][n] = np.random.uniform(0, 1) #e
            population[4][n] = np.random.uniform(0, self.data['radial_velocity'].max() - self.data['radial_velocity'].min()) #K
            population[5][n] = np.random.uniform(self.data['radial_velocity'].min(), self.data['radial_velocity'].max()) #V0

        return population

    #Evaluate objective function for each individual
    def evaluate_objective(self, P, tau, omega, e, K, V0):
        return 1/self.evaluate_chi_squared(P, tau, omega, e, K, V0)

    #Sum squared mean difference of all evaluations (all recorded timestamps)
    def evaluate_chi_squared(self, P, tau, omega, e, K, V0):

        sum = 0

        for idx, row in self.data.iterrows():
            sum += (row[1] - self.evaluate_solution(P, tau, omega, e, K, V0, row[0])/row[2])**2

        return 1/(self.data.shape[0]-6) * sum

    #Evaluate solution at specific time
    def evaluate_solution(self, P, tau, omega, e, K, V0, t):

        E = np.empty(self.N)

        #Solve keplers equation for the root
        for n in range(self.N):
            E[n] = bisect(self.kepler_equation, -300, 300, args=(t, P[n], tau[n], e[n]))


        #Solve for projected velocity v
        v = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2)) #CHECK FOR ATAN2 ERROR ? ONLY 1 INPUT AVAILABLE

        print(v)

        #Solve for radial velocity
        V = V0 + K *(np.cos(omega + v) + e*np.cos(omega))

        return V

    #Keplers equation
    def kepler_equation(self, E, t, P, tau, e):
        return E - e * np.sin(E) - (2*np.pi/P) * (t-tau)


gen_evolve = GeneticOptimisation()
gen_evolve.simulate_generations(10)
print(gen_evolve.population)
