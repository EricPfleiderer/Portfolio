import numpy as np
import matplotlib.pyplot as plt
import math


'''
Do not over abstract!
do not overoptimize!
'''
class EvolutionOpti:

    #N must be even
    def __init__(self, N=10):

        self.N = N #Population size
        self.population = self.initialize_random() #Initialize solutions to random (2xN structure)


    def simulate_generations(self, generations=100, selective_pressure=1/5, p_c=0.8, p_m=0.1, mutation_amp = 0.25):

        for generation in range(generations):

            #Evaluate current solutions
            current_solution = self.evaluate_quality(self.population)

            #Rank solutions
            rank = np.argsort(current_solution) #Contains index of current solution elements (from smallest to lowest)

            #Select N new solutions, repetitions allowed (parents)
            parents = np.empty(0)

            for n in range(self.N//2):
                best_parent, second_parent = self.generate_parents(rank, selective_pressure)
                parents = np.append(parents, best_parent)
                parents = np.append(parents, second_parent)
                parents = np.int32(parents)

            #Generate new children through procreation of parents
            children = np.array([np.empty(self.N), np.empty(self.N)])

            for n in range(0, parents.size, 2):
                x1, y1, x2, y2 = self.generate_kids(parents[n], parents[n+1], p_c)

                children[0][n] = x1
                children[1][n] = y1
                children[0][n+1] = x2
                children[1][n+1] = y2

            #Mutate children if need be
            for n in range(children[0].size):
                if np.random.uniform(0, 1) <= p_m:
                    children[0][n] += np.random.normal(0, mutation_amp)
                if np.random.uniform(0, 1) <= p_m:
                    children[1][n] += np.random.normal(0, mutation_amp)


            #Preserve best solution (elitism) and update current population
            xtop, ytop = self.population[0][rank[self.N-1]], self.population[1][rank[self.N-1]]
            self.population = children
            self.population[0][0], self.population[1][0] = xtop, ytop

    #Returns two kids according to parents
    def generate_kids(self, parent1_idx, parent2_idx, p_c=0.8):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)

        # Parent solutions
        x_p1 = self.population[0][parent1_idx]
        x_p2 = self.population[0][parent2_idx]
        y_p1 = self.population[1][parent1_idx]
        y_p2 = self.population[1][parent2_idx]

        if np.random.uniform(0, 1) <= p_c:
            x1 = r1 * x_p1 + (1 - r1) * x_p2
            x2 = (1 - r1) * x_p1 + r1 * x_p2

        else:
            x1 = x_p1
            x2 = x_p2

        if np.random.uniform(0, 1) <= p_c:
            y1 = r2 * y_p1 + (1 - r2) * y_p2
            y2 = (1 - r2) * y_p2 + r2 * y_p2

        else:
            y1 = y_p1
            y2 = y_p2

        return x1, y1, x2, y2

    #Returns index of chosen parents from rank
    def generate_parents(self, rank, selective_pressure=1/5):
        # Choose best parent randomly according to selective pressure
        best_index = rank[np.random.randint(int((1 - selective_pressure) * self.N), self.N)]

        # Choose second parent
        second_index = best_index
        while second_index == best_index:
            second_index = np.random.randint(0, self.N - 1)

        return best_index, second_index


    def initialize_random(self, L=3*np.pi):
        population = np.array([np.zeros(self.N), np.zeros(self.N)])

        for n in range(self.N):
            population[0][n] = np.random.uniform(-L, L)
            population[1][n] = np.random.uniform(-L, L)

        return population

    def evaluate_quality(self, solution_matrix):
        return (np.sin(solution_matrix[0]) * np.sin(solution_matrix[1]) / (solution_matrix[0]*solution_matrix[1]))**2


evolve10 = EvolutionOpti(N=10)
evolve20 = EvolutionOpti(N=20)
evolve10.simulate_generations(0)
evolve20.simulate_generations(0)

delta = 0.025
x = np.arange(-3*np.pi, 3*np.pi, delta)
y = np.arange(-3*np.pi, 3*np.pi, delta)
X, Y = np.meshgrid(x, y)
Z = (np.sin(X) * np.sin(Y) / (X*Y))**2

step_size = 1
total_gens = 1000

bestx10, besty10 = np.empty(0), np.empty(0)
bestx20, besty20 = np.empty(0), np.empty(0)

for x in range(0, int(total_gens/step_size)):
    evolve10.simulate_generations(step_size)
    evolve20.simulate_generations(step_size)
    bestx10, besty10 = np.append(bestx10, evolve10.population[0][0]), np.append(besty10, evolve10.population[1][0])
    bestx20, besty20 = np.append(bestx20, evolve20.population[0][0]), np.append(besty20, evolve20.population[1][0])

    '''
    plt.figure()
    plt.scatter(evolve.population[0][1:], evolve.population[1][1:], c='red')
    plt.scatter(evolve.population[0][0], evolve.population[1][0], c='blue')
    plt.contour(X, Y, Z, levels=np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]))
    plt.plot(bestx, besty, c='green')
    plt.xlim(-3*np.pi, 3*np.pi)
    plt.ylim(-3*np.pi, 3*np.pi)
    plt.savefig('results/evolve'+str(x*step_size)+'.png')
    '''

error10 = np.abs(1 - evolve10.evaluate_quality(np.array([bestx10, besty10])))
error20 = np.abs(1 - evolve20.evaluate_quality(np.array([bestx20, besty20])))
xAxis = np.arange(0, error10.size)

plt.figure()
plt.plot(xAxis, error10, label='10')
plt.plot(xAxis, error20, label='20')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('$\epsilon$')
plt.legend(loc='best')
plt.savefig('results/error.png')


