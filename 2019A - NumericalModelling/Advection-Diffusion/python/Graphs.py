import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from Leapfrog import Leapfrog
from LaxWendroff import LaxWendroff
from Analysis import SD


# Prints results of simulations and analysis
class Print:

    def __init__(self):
        self.fig2_21()
        self.fig2_22LF()
        self.fig2_22LW()
        self.fig2_23()
        self.printData()
        self.printGuess()
        self.residualsGraph()

    @staticmethod
    def fig2_21():
        LF = Leapfrog(int(3 / 10 ** -3))

        plt.figure()
        plt.imshow(LF.grid[int(-2.5/1e-3), 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=0.5$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LFDM0_5s.pdf', bbox_inches='tight')

        plt.figure()
        plt.imshow(LF.grid[int(-2/1e-3), 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=1.0$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LFDM1s.pdf', bbox_inches='tight')

        plt.figure()
        plt.imshow(LF.grid[int(-1/(1e-3)), 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=2.0$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LFDM2s.pdf', bbox_inches='tight')

        plt.figure()
        plt.imshow(LF.grid[-1, 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=3.0$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LFDM3s.pdf', bbox_inches='tight')

    @staticmethod
    def fig2_22LF():

        LF = Leapfrog(int(3 / 10 ** -3))
        time, c = LF.c_pos(0.25*2*np.pi, 0.75*2*np.pi)
        time2, c2 = LF.c_pos(0.5*2*np.pi, 0.5*2*np.pi)
        time3, c3 = LF.c_pos(0.75*2*np.pi, 0.25*2*np.pi)

        plt.figure()
        plt.xlabel('$t$')
        plt.ylabel('$Concentration$')
        plt.plot(time, c, color='purple', label=r'$(0.25 \times 2 \pi, 0.75 \times 2 \pi)$')
        plt.plot(time2, c2, color='blue', label=r'$(0.5 \times 2 \pi, 0.5 \times 2 \pi)$')
        plt.plot(time3, c3, color='green', label=r'$(0.75 \times 2 \pi, 0.25 \times 2 \pi)$')
        plt.legend(loc='best')
        plt.savefig('img/LeapfrogGraph.pdf', bbox_inches='tight')

    @staticmethod
    def fig2_22LW():

        LW = LaxWendroff(int(3 / 10 ** -3))
        time, c = LW.c_pos(0.25*2*np.pi, 0.75*2*np.pi)
        time2, c2 = LW.c_pos(0.5*2*np.pi, 0.5*2*np.pi)
        time3, c3 = LW.c_pos(0.75*2*np.pi, 0.25*2*np.pi)

        plt.figure()
        plt.xlabel('$t$')
        plt.ylabel('$Concentration$')
        plt.plot(time, c, color='purple', label=r'$(0.25 \times 2 \pi, 0.75 \times 2 \pi)$')
        plt.plot(time2, c2, color='blue', label=r'$(0.5 \times 2 \pi, 0.5 \times 2 \pi)$')
        plt.plot(time3, c3, color='green', label=r'$(0.75 \times 2 \pi, 0.25 \times 2 \pi)$')
        plt.legend(loc='best')

        plt.savefig('img/LaxWendroffGraph.pdf', bbox_inches='tight')

    def fig2_23(self):
        LW = LaxWendroff(int(3 / 10 ** -3))

        plt.figure()
        plt.imshow( LW.grid[int(-2.5/1e-3), 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=0.5$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LWDM0_5s.pdf', bbox_inches='tight')

        plt.figure()
        plt.imshow( LW.grid[int(-2/(1e-3)), 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=1.0$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LWDM1s.pdf', bbox_inches='tight')

        plt.figure()
        plt.imshow( LW.grid[int(-1/(1e-3)), 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=2.0$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LWDM2s.pdf', bbox_inches='tight')

        plt.figure()
        plt.imshow( LW.grid[-1, 1:-1, 1:-1], cmap=cm.YlGnBu, origin='lower', extent=(0, 2*np.pi, 0, 2*np.pi), vmin=-0.05, vmax=1)
        plt.colorbar()
        plt.title('$t=3.0$')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(0.25*2*np.pi, 0.75*2*np.pi, 'o', color='purple')
        plt.plot(0.5*2*np.pi, 0.5*2*np.pi, 'o', color='blue')
        plt.plot(0.75*2*np.pi, 0.25*2*np.pi, 'o', color='green')
        plt.savefig('img/LWDM3s.pdf', bbox_inches='tight')

    @staticmethod
    def printData():
        stochDesc = SD()
        plt.figure()
        plt.xlabel('$t$')
        plt.ylabel('$Concentration$')
        plt.plot(stochDesc.data['t'], stochDesc.data['1'], color='green', label=r'$(0.75 \times 2 \pi, 0.25 \times 2 \pi)$')
        plt.plot(stochDesc.data['t'], stochDesc.data['2'], color='blue', label=r'$(0.5 \times 2 \pi, 0.5 \times 2 \pi)$')
        plt.plot(stochDesc.data['t'], stochDesc.data['3'], color='purple', label=r'$(0.25 \times 2 \pi, 0.75 \times 2 \pi)$')
        plt.legend(loc="upper left")
        plt.savefig('img/printData.pdf', bbox_inches='tight')

    @staticmethod
    def printGuess():

        stochDesc = SD()

        # Initial solution
        S = 1.8
        x = 4
        y = 2.5
        ti = 0.1

        # Close to solution
        LW = LaxWendroff(int(4 / 10 ** -3), S=S, x=x, y=y, tStart=ti, tEnd=2.5)

        time, c = LW.c_pos(0.25*2*np.pi, 0.75*2*np.pi)
        time2, c2 = LW.c_pos(0.5*2*np.pi, 0.5*2*np.pi)
        time3, c3 = LW.c_pos(0.75*2*np.pi, 0.25*2*np.pi)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('$t$')
        ax1.set_ylabel('$Concentration$')

        exp1, = ax1.plot(time, c, color='purple', label=r'$(0.25 \times 2 \pi, 0.75 \times 2 \pi)$')
        exp2, = ax1.plot(time2, c2, color='blue', label=r'$(0.5 \times 2 \pi, 0.5 \times 2 \pi)$')
        exp3, = ax1.plot(time3, c3, color='green', label=r'$(0.75 \times 2 \pi, 0.25 \times 2 \pi)$')
        exp1.set_dashes([2, 2, 10, 2])
        exp2.set_dashes([2, 2, 2, 2])
        exp3.set_dashes([5, 2, 5, 2])

        _, = ax1.plot(stochDesc.data['t'], stochDesc.data['1'], color='green')
        _, = ax1.plot(stochDesc.data['t'], stochDesc.data['2'], color='blue')
        _, = ax1.plot(stochDesc.data['t'], stochDesc.data['3'], color='purple')
        plt.legend(loc="upper left")
        plt.savefig('img/GuessGraph.pdf', bbox_inches='tight')

    @staticmethod
    def residualsGraph():

        StochDesc = SD()

        residuals = np.array(StochDesc.optimize())
        xAxis = [x for x in range(len(residuals))]

        plt.figure()
        plt.xlabel('$Itérations$')
        plt.ylabel('$Résiduels$')
        plt.ylim(0, np.amax(residuals))
        plt.scatter(xAxis, residuals)
        plt.savefig('img/residuels.pdf', bbox_inches='tight')
