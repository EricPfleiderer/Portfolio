import pandas as pd
import numpy as np
from scipy.optimize import bisect

'''
BUG:
    -All data becomes NaN if e>1 or e<-1, must impose bounderies?
    -No error caused by omega, but same idea? impose bounderies?
'''

class GenOpti:

    #N must be even
    def __init__(self, N=10, vector_size=6, path='data/eta_Bootis.csv'):
        self.N = N
        self.vector_size = vector_size
        self.data = pd.read_csv(path)
        self.population = self.initialize_random() #P, tau, omega, e, K, V0
        self.mutation_vector = self.initialize_mutation(size=0.1)

    #Initialize population of N solution vectors according to specific intervals obtained from data analysis
    def initialize_random(self):

        population = np.zeros((self.N, self.vector_size))

        for n in range(self.N):
            population[n][0] = np.random.uniform(200, 800) #P
            population[n][1] = np.random.uniform(self.data.iloc[0, 0], self.data.iloc[0, 0] + population[n][0]) #tau
            population[n][2] = np.random.uniform(0, 2*np.pi) #omega
            population[n][3] = np.random.uniform(0, 1) #e
            population[n][4] = np.random.uniform(0, self.data['radial_velocity'].max() - self.data['radial_velocity'].min()) #K
            population[n][5] = np.random.uniform(self.data['radial_velocity'].min(), self.data['radial_velocity'].max()) #V0

        return population

    def simulate_generations(self, generations=20, selective_pressure=1/5, p_c=0.8, p_m=0.1):

        for generation in range(generations):

            #Evaluate current solutions
            current_solution = self.evaluate_objective(self.population)

            #Rank solutions
            rank = np.argsort(current_solution) #Contains index of current solution elements (from smallest to lowest)

            #Select N new solutions, repetitions allowed (parents)
            parents_idx = np.empty(0)

            for n in range(self.N//2):
                best_parent, second_parent = self.generate_parents(rank, selective_pressure)
                parents_idx = np.append(parents_idx, best_parent)
                parents_idx = np.append(parents_idx, second_parent)
                parents_idx = np.int32(parents_idx)

            #print(current_solution)
            #print(rank)
            #print(parents)

            children = self.generate_kids(parents_idx, p_c)

            #print(children)

            #Mutate children if need be
            for n in range(self.N):
                for component in range(self.vector_size):
                    if np.random.uniform(0, 1) <= p_m:
                        children[n][component] += np.random.normal(0, abs(self.mutation_vector[component]))

            # Preserve best solution (elitism) and update current population
            top_sol = self.population[rank[self.N-1], :]
            self.population = children
            self.population[0, :] = top_sol

    #TO CHECK FOR BUGS, SUSPECTED
    def generate_kids(self, parents_idx, p_c=0.8):

        r = np.empty(self.vector_size)

        #Generate a ratio for every vector component
        for n in range(r.size):
            r[n] = np.random.uniform(0, 1)

        children = np.empty((self.N, self.vector_size))

        #For every couple of parents
        for n in range(self.N//2):

            #Solution vectors for current pair of reproducing parents
            parent1_sol = self.population[parents_idx[2*n], :]
            parent2_sol = self.population[parents_idx[2*n+1], :]

            #Generate 2 new solutions vectors from parent vectors
            for component in range(parent1_sol.size):
                if np.random.uniform(0, 1) <= p_c:
                    children[2*n][component] = r[component] * parent1_sol[component] + (1-r[component]) * parent2_sol[component]
                    children[2*n+1][component] = (1-r[component]) * parent1_sol[component] + r[component] * parent2_sol[component]
                else:
                    children[2*n][component] = parent1_sol[component]
                    children[2*n+1][component] = parent2_sol[component]

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

    #Evaluate objective function for each individual (solution matrix must be n by 6 2darray)
    def evaluate_objective(self, solution_matrix):
        chi = 0
        for idx, row in self.data.iterrows():
            chi += (row[1] - self.evaluate_solution(solution_matrix, row[0])/row[2])**2
        chi *= 1/(self.data.shape[0])

        return 1/chi

    #Evaluate solution at specific time
    def evaluate_solution(self, solution_matrix, t):

        E = np.empty(0)

        P = solution_matrix[:, 0]
        tau = solution_matrix[:, 1]
        omega = solution_matrix[:, 2]
        e = solution_matrix[:, 3]
        K = solution_matrix[:, 4]
        V0 = solution_matrix[:, 5]


        #Solve keplers equation for the root
        for n in range(P.shape[0]):
            E = np.append(E, bisect(self.evaluate_kepler, -10000, 10000, args=(t, P[n], tau[n], e[n])))

        #Solve for projected velocity v
        orbital_velocity = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2)) #CHECK FOR ATAN2 ERROR ? ONLY 1 INPUT AVAILABLE

        #Solve for radial velocity
        radial_velocity = V0 + K * (np.cos(omega + orbital_velocity) + e*np.cos(omega))

        return radial_velocity

    #Keplers equation
    def evaluate_kepler(self, E, t, P, tau, e):
        return E - e * np.sin(E) - (2*np.pi/P) * (t-tau)

    #To check if needed for different order of parameters
    def initialize_mutation(self, size=0.2):

        mutation_vector = np.zeros(self.vector_size)

        for vector_idx in range(self.vector_size):
            mutation_vector[vector_idx] = size * (1/self.vector_size)*np.sum(self.population[:, vector_idx])

        return mutation_vector

#test = GenOpti()
#test.simulate_generations(30)
#print(test.population)
#print(test.mutation_vector)
#print(test.evaluate_objective(test.population))
#print(test.evaluate_objective(np.array([[494.2, 14299, 5.7397, 0.2626, 8.3836, 1.0026]])))
