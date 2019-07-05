from Model import Model
import numpy as np


class Leapfrog:

    """
    gridSize: Scope of spatial dimensions
    iterations: Number of steps to be solved
    tStart, tEnd: Interval of active source (pollutant or otherwise)
    deltas: Parameters for numerical algorithms used to solve the simulation. Must meet Courant criteria to avoid divergence of numerical solution.
    """

    # Solver constructor
    def __init__(self, iterations, S=2, x=0.8*2*np.pi, y=0.6*2*np.pi, tStart=0.01, tEnd=1.75, Pe=1e3, deltaS=0.02*np.pi, deltaT=1e-3):

        # Generalizing scope
        self.gridSize = int(round(2*np.pi/deltaS, 1)) + 1
        self.iterations = iterations
        self.tStart = tStart
        self.tEnd = tEnd
        self.Pe = Pe
        self.deltaS = deltaS
        self.deltaT = deltaT

        # Initializing
        self.discretX = np.arange(0, self.gridSize*deltaS, deltaS)
        self.discretY = np.arange(0, self.gridSize*deltaS, deltaS)
        self.discretX, self.discretY = np.meshgrid(self.discretX, self.discretY)
        self.discretT = np.arange(0, iterations*deltaT, deltaT)
        self.time = 0
        self.model = Model(S, x, y)
        self.grid = np.zeros([iterations, self.gridSize+2, self.gridSize+2])

        # Simulation
        for x in range(iterations-1):
            self.LeapfrogStep(x)
            self.time += deltaT

    # Impose periodicity over the studied domain.
    def setPeriodicity(self, spatialArray):

        N = self.gridSize

        # Horizontal
        spatialArray[1:N+1, 0] = spatialArray[1:N+1, N]
        spatialArray[1:N+1, N+1] = spatialArray[1:N+1, 1]

        # Vertical
        spatialArray[0, 1:N+1] = spatialArray[N, 1:N+1]
        spatialArray[N+1, 1:N+1] = spatialArray[1, 1:N+1]

        # Corners
        spatialArray[0, 0] = spatialArray[N, N]
        spatialArray[N+1, N+1] = spatialArray[1, 1]
        spatialArray[0, N+1] = spatialArray[N, 1]
        spatialArray[N+1, 0] = spatialArray[1, N]

    # Returns two lists to be plotted, time and concentration according to position entered.
    # X and Y between [0, 2Pi]
    def c_pos(self, x, y):
        return self.discretT, [t[int(y*self.gridSize/(2*np.pi))+1][int(x*self.gridSize/(2*np.pi))+1] for t in self.grid]

    # Leapfrog with FCTS as first step to avoid out of bounds error.
    def LeapfrogStep(self, stepNumber):

        # First step with FCTS
        if stepNumber is 0:
            self.grid[stepNumber+1, 1:-1, 1:-1] = 0.25*(self.grid[stepNumber, 1:-1, 2:] + self.grid[stepNumber, 1:-1, 0:-2] + self.grid[stepNumber, 2:, 1:-1] + self.grid[stepNumber, 0:-2, 1:-1]) \
                                                - ((self.model.advection.getXAdv(self.discretY, self.time)*self.deltaT/(2*self.deltaS)) * (self.grid[stepNumber, 1:-1, 2:] - self.grid[stepNumber, 1:-1, 0:-2])) \
                                                - ((self.model.advection.getYAdv(self.discretX, self.time)*self.deltaT/(2*self.deltaS)) * (self.grid[stepNumber, 2:, 1:-1] - self.grid[stepNumber, 0:-2, 1:-1])) \
                                                + (((self.deltaT)/(self.deltaS**2 * self.Pe)) * (self.grid[stepNumber, 1:-1, 2:] - 2 * self.grid[stepNumber, 1:-1, 1:-1] + self.grid[stepNumber, 1:-1, 0:-2])) \
                                                + (((self.deltaT)/(self.deltaS**2 * self.Pe)) * (self.grid[stepNumber, 2:, 1:-1] - 2 * self.grid[stepNumber, 1:-1, 1:-1] + self.grid[stepNumber, 0:-2, 1:-1]))

            # Source
            if self.tStart <= self.time <= self.tEnd:
                self.grid[stepNumber + 1, 1:-1, 1:-1] += self.deltaT * self.model.source.getSource(self.discretX, self.discretY)

        # Rest of the steps with Leapfrog
        else:

            self.grid[stepNumber+1, 1:-1, 1:-1] = self.grid[stepNumber-1, 1:-1, 1:-1] \
                                                - ((self.model.advection.getXAdv(self.discretY, self.time)*self.deltaT/self.deltaS) * (self.grid[stepNumber, 1:-1, 2:] - self.grid[stepNumber, 1:-1, 0:-2])) \
                                                - ((self.model.advection.getYAdv(self.discretX, self.time)*self.deltaT/self.deltaS) * (self.grid[stepNumber, 2:, 1:-1] - self.grid[stepNumber, 0:-2, 1:-1])) \
                                                + (((2*self.deltaT)/(self.deltaS**2 * self.Pe)) * (self.grid[stepNumber, 1:-1, 2:] - 2 * self.grid[stepNumber, 1:-1, 1:-1] + self.grid[stepNumber, 1:-1, 0:-2])) \
                                                + (((2*self.deltaT)/(self.deltaS**2 * self.Pe)) * (self.grid[stepNumber, 2:, 1:-1] - 2 * self.grid[stepNumber, 1:-1, 1:-1] + self.grid[stepNumber, 0:-2, 1:-1]))

            # Source
            if self.tStart <= self.time <= self.tEnd:
                self.grid[stepNumber + 1, 1:-1, 1:-1] += 2 * self.deltaT * self.model.source.getSource(self.discretX, self.discretY)

        # Impose periodicity over the domain once the solution is obtained.
        self.setPeriodicity(self.grid[stepNumber+1])


