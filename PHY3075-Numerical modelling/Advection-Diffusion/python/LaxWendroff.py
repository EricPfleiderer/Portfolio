from Model import Model
import numpy as np


class LaxWendroff:

    """
    gridSize: Scope of spatial dimensions
    iterations: Number of steps to be solved
    tStart, tEnd: Interval of active source (pollutant or otherwise)
    deltas: Parameters for numerical algorithms used to solve the simulation. Must meet Courant criteria to avoid divergence of numerical solution.
    """

    # Solver constructor
    def __init__(self, iterations, S=2., x=0.8*2*np.pi, y=0.6*2*np.pi, tStart=0.01, tEnd=1.75, Pe=1e3, deltaS=5e-3*np.pi, deltaT=1e-3):

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

        self.sourceValues = self.model.source.getSource(self.discretX, self.discretY)

        # Simulation
        for x in range(iterations-1):
            self.LaxWendroffStep(x)
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
    def LaxWendroffStep(self, stepNumber):

        if stepNumber % 1000 is 0:
            print("LaxWendroff working... step: ", stepNumber)

        # Calcul des solutions a demis pas dans le temps et dans l'espace
        solDemiPas = np.zeros((self.gridSize+2, self.gridSize+2))

        solDemiPas[1:-1, 1:-1] = 0.25 * (self.grid[stepNumber, 1:-1, 2:] + self.grid[stepNumber, 1:-1, 1:-1] + self.grid[stepNumber, 2:, 2:] + self.grid[stepNumber, 2:, 1:-1]) \
                                 - (self.deltaT/(2.*self.deltaS)) * self.model.advection.getXAdv(self.discretY + self.deltaS/2., self.time+self.deltaT/2.) * (self.grid[stepNumber, 1:-1, 2:] - self.grid[stepNumber, 1:-1, 1:-1]) \
                                 - (self.deltaT/(2.*self.deltaS)) * self.model.advection.getYAdv(self.discretX + self.deltaS/2., self.time+self.deltaT/2.) * (self.grid[stepNumber, 2:, 1:-1] - self.grid[stepNumber, 1:-1, 1:-1])

        self.setPeriodicity(solDemiPas)

        # Calcul de Flux a demis pas dans le temps et dans l'espace
        fluxXDemiPas = np.zeros((self.gridSize+2, self.gridSize+2))
        fluxYDemiPas = np.zeros((self.gridSize+2, self.gridSize+2))

        diffX = self.grid[stepNumber, 1:-1, 2:] + self.grid[stepNumber, 2:, 2:] - self.grid[stepNumber, 1:-1, 1:-1] - self.grid[stepNumber, 2:, 1:-1]
        diffY = self.grid[stepNumber, 2:, 1:-1] + self.grid[stepNumber, 2:, 2:] - self.grid[stepNumber, 1:-1, 1:-1] - self.grid[stepNumber, 1:-1, 2:]
        fluxXDemiPas[1:-1, 1:-1] = self.model.advection.getXAdv(self.discretY + self.deltaS/2, self.time+self.deltaT/2) * solDemiPas[1:-1, 1:-1] - (1/(2*self.deltaS*self.Pe)) * diffX
        fluxYDemiPas[1:-1, 1:-1] = self.model.advection.getYAdv(self.discretX + self.deltaS/2, self.time+self.deltaT/2) * solDemiPas[1:-1, 1:-1] - (1/(2*self.deltaS*self.Pe)) * diffY
        self.setPeriodicity(fluxXDemiPas)
        self.setPeriodicity(fluxYDemiPas)

        # Calcul des nouvelles solutions a plein pas
        self.grid[stepNumber+1, 1:-1, 1:-1] = self.grid[stepNumber, 1:-1, 1:-1] - \
                                (self.deltaT/(2*self.deltaS)) * (fluxXDemiPas[0:-2, 1:-1] - fluxXDemiPas[0:-2, 0:-2] + fluxXDemiPas[1:-1, 1:-1] - fluxXDemiPas[1:-1, 0:-2]) - \
                                (self.deltaT/(2*self.deltaS)) * (fluxYDemiPas[1:-1, 1:-1] - fluxYDemiPas[0:-2, 1:-1] + fluxYDemiPas[1:-1, 0:-2] - fluxYDemiPas[0:-2, 0:-2]) \

        # Impose periodicity over the domain once the solution is obtained.
        self.setPeriodicity(self.grid[stepNumber+1])

        if self.tStart <= self.time <= self.tEnd:
            # self.grid[stepNumber + 1, 1:-1, 1:-1] += self.deltaT * self.model.source.getSource(self.discretX, self.discretY)
            self.grid[stepNumber + 1, 1:-1, 1:-1] += self.deltaT * self.sourceValues

