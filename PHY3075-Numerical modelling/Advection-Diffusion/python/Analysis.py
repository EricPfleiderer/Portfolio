from LaxWendroff import LaxWendroff
from Leapfrog import Leapfrog
from random import gauss
import numpy as np
import matplotlib.pyplot as plt


class SD:

    def __init__(self, epsilon=1.e-10, maxIter=20):
        self.epsilon = epsilon
        self.maxIter = maxIter
        self.data = {"t": [],
                     "1": [],
                     "2": [],
                     "3": []
                     }
        self.readData()
        self.printData()

    def readData(self):
        file = open("polluants.txt", "r")
        lines = file.readlines()

        for x in range(1, len(lines)):
            fragments = lines[x].split()
            self.data["t"].append(float(fragments[0]))
            self.data["1"].append(float(fragments[1]))
            self.data["2"].append(float(fragments[2]))
            self.data["3"].append(float(fragments[3]))

    def printData(self):
        plt.figure()
        plt.plot(self.data["t"], self.data["1"], color="green")
        plt.plot(self.data["t"], self.data["2"], color="blue")
        plt.plot(self.data["t"], self.data["3"], color="purple")
        plt.show()

    def optimize(self):

        # Connu
        tf = 2.5

        # Solution obtenu par leapfrog
        S = 2
        x = 0.7*2*np.pi
        y = 0.4*2*np.pi
        ti = 0.25

        tin = ti
        Sn = S
        xn = x
        yn = y

        residuals = []

        # A minimiser, stock la plus petite valeur trouvée jusqu'à présent
        f = self.squaredResiduals(S, x, y, ti, tf)
        residuals.append(f)

        # Essaies Monte Carlo
        fn = f*2

        # Step size
        h = 0.1

        delta = 2*self.epsilon

        # Monte carlo max tries to find better solution
        maxTries = 20

        k = 0
        while(delta > self.epsilon) and (k < self.maxIter):
            j = 0

            print("Iteration k:", k, "Current solution (S, x, y, td): ", round(S, 6), round(x, 6), round(y, 6), round(ti, 6),
                  ", delta: " + str(round(delta, 6)), " h step: " + str(round(h, 6)))

            # Monte carlo
            while (fn >= f) and (j < maxTries):
                if j % 5 is 0 and j > 0:
                    print("Monte Carlo trial j:", j)

                # Descente stochastique
                xn = x + gauss(0, h)
                yn = y + gauss(0, h)
                Sn = S + gauss(0, h)
                tin = ti + gauss(0, h)

                # On sait que 0 <= td <= 0.5
                if tin < 0:
                    tin = 0
                if tin > 0.5:
                    tin = 0.5

                fn = self.squaredResiduals(Sn, xn, yn, tin, tf)
                j += 1

            # On trouve une meilleur solution. On continue à avancer.
            if fn < f:
                S, x, y, ti = Sn, xn, yn, tin
                delta = abs(fn-f)
                f = fn
                print(f)
                residuals.append(f)

            # On ne trouve pas de meilleur solution après une séquence de Monte Carlo. On réduit le pas et on converge vers le minimum local.
            else:
                h /= 2
            k += 1

        return residuals

    # Returns sum or residual squared errors for 3 data positions
    def squaredResiduals(self, S, x, y, ti, tf):

        # LW = Leapfrog(int(4/1e-3), S, x, y, ti, tf)
        LW = LaxWendroff(int(4/1e-3), S, x, y, ti, tf)
        time1, c1 = LW.c_pos(0.75*2*np.pi, 0.25*2*np.pi)
        time2, c2 = LW.c_pos(0.5*2*np.pi, 0.5*2*np.pi)
        time3, c3 = LW.c_pos(0.25*2*np.pi, 0.75*2*np.pi)

        res1, res2, res3 = 0, 0, 0

        for step in range(len(c1)):
            res1 += (self.data["1"][step] - c1[step]) ** 2
            res2 += (self.data["2"][step] - c2[step]) ** 2
            res3 += (self.data["3"][step] - c3[step]) ** 2

        return 1/len(c1) * (res1 + res2 + res3)
