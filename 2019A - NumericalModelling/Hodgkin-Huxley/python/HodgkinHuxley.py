import math
from abc import ABC, abstractmethod
import numpy as np


# Implementation of the Hudgkin-Huxley model, including Runge Katta solver.
class HodgkinHuxley:

    # Internal classes
    class Channel(ABC):

        @abstractmethod
        def alpha(self, V):
            pass

        @abstractmethod
        def beta(self, V):
            pass

        def getDiff(self, V, prob):
            return self.alpha(V)*(1-prob) - self.beta(V)*prob

        def equil(self, V):
            return self.alpha(V)/(self.alpha(V)+self.beta(V))

        def tau(self, V):
            return 1/(self.alpha(V) + self.beta(V))

    class N(Channel):
        def alpha(self, V):
            return (0.1 - 0.01 * V)/(math.e ** (1 - 0.1 * V) - 1)

        def beta(self, V):
            return 0.125 * math.e ** (-V / 80)

    class M(Channel):
        def alpha(self, V):
            return (2.5 - 0.1 * V) / (math.e ** (2.5 - 0.1 * V) - 1)

        def beta(self, V):
            return 4 * math.e ** (-V / 18)

    class H(Channel):
        def alpha(self, V):
            return 0.07 * math.e ** (-V / 20)

        def beta(self, V):
            return 1 / (math.e ** (3 - 0.1 * V) + 1)

    class Voltage:
        @staticmethod
        def getDiff(V, n, m, h, Ia=0, gL=0.3, Cm=1):
            return (1/Cm)*(Ia - 36*n**4*(V-(-12)) - 120*m**3*h*(V-115) - gL*(V-10.6))

        @staticmethod
        def equil(Ia, n, m, h):
            return (Ia + 36*(-12)*n**4 + 115*120*m**3*h + 10.6*0.3) / (36**n**4 + 120*m**3*h + 0.3)

    def __init__(self):
        self.n = self.N()
        self.m = self.M()
        self.h = self.H()
        self.V = self.Voltage()

    # Evaluate and return diffential equations
    def evaluate(self, t0, vector, amperage, gL=0.3, Cm=1):
        V, n, m, h = vector[0], vector[1], vector[2], vector[3]

        return np.array([self.V.getDiff(V, n, m, h, amperage, gL, Cm),
                         self.n.getDiff(V, n),
                         self.m.getDiff(V, m),
                         self.h.getDiff(V, h)])

    # Return next step
    def step(self, h, t0, vector, amperage, gL=0.3, Cm=1):

        sol1 = self.evaluate(t0, vector, amperage, gL, Cm)
        sol2 = self.evaluate(t0+h/2., vector+h*sol1/2, amperage, gL, Cm)
        sol3 = self.evaluate(t0+h/2., vector+h*sol2/2, amperage, gL, Cm)
        sol4 = self.evaluate(t0+h, vector+h*sol3, amperage, gL, Cm)
        return vector + h/6.*(sol1+2.*sol2+2.*sol3+sol4)

    '''
    Concerning this method:
        -The system must be at rest before starting to push current
        -Amplitude and duration of current can be modified
        -Multiple pulses of current can be applied with constant interval of time in between
    '''
    def solve(self, pulseAmperage, pulseLength, nbPulses, pauseLength, tend, Vini, gL = 0.3, Cm = 1):

        # print("Starting Runge Kutta...")
        nMax = 100000  # Lots of room to breath...
        eps = 1.e-12  # Tolerance
        tend = tend  # Integration length (ms)
        t = np.zeros(nMax)  # Time
        u = np.zeros([nMax, 4])  # Solutions

        # Define intervals during which the pulses will be active
        activePulse = [[(pulseLength+pauseLength)*x, (pulseLength+pauseLength)*x+pulseLength] for x in range(nbPulses)]

        # Rest conditions for initial conditions
        u[0, :] = np.array([Vini, self.n.equil(Vini), self.m.equil(Vini), self.h.equil(Vini)])

        nn = 0  # Iteration count
        h = 0.01  # Step size (ms)

        # Data points retained
        data = {"t": [],
                "V": [],
                "n": [],
                "m": [],
                "h": [],
                "Ir": []}

        # Initialize amperage according to parameter
        currAmperage = pulseAmperage

        # Iterative solution
        while (t[nn] < tend) and (nn < nMax-1):

            # Random current
            if pulseAmperage is "random":
                if nn is 0:
                    currAmperage = np.random.normal(0, 1)

                else:
                    hit = False
                    for x in range(len(activePulse)):
                        # Active current
                        if activePulse[x][0] <= t[nn] < activePulse[x][1]:
                            hit = True

                            # Current ampplitude needs to be changed
                            if not (activePulse[x][0] <= t[nn - 1] < activePulse[x][1]):
                                currAmperage = np.random.normal(0, 1)

                            # Else... active current keeps same amplitude

                    # Inactive current
                    if not hit:
                        currAmperage = 0

            # Constant current
            else:

                currAmperage = 0

                for x in activePulse:
                    if x[0] <= t[nn] < x[1]:
                        currAmperage = pulseAmperage

            # Full step solution
            u1 = self.step(h, t[nn], u[nn, :], currAmperage, gL, Cm)

            # Half steps solution
            u2a = self.step(h/2., t[nn], u[nn, :], currAmperage, gL, Cm)
            u2 = self.step(h/2., t[nn], u2a[:], currAmperage, gL, Cm)

            # Delta
            delta = max(abs(u2-u1)/abs(u2))  # Erreurs relative

            # Reject the step
            if delta > eps:
                h /= 1.5

            # Accept and apply the step
            else:
                nn += 1
                t[nn] = t[nn-1]+h
                u[nn, :] = u2[:]
                if delta <= eps/2.:
                    h *= 1.5
                data["t"].append(t[nn])
                data["V"].append(u[nn, 0])
                data["n"].append(u[nn, 1])
                data["m"].append(u[nn, 2])
                data["h"].append(u[nn, 3])
                data["Ir"].append(currAmperage)

        return data
