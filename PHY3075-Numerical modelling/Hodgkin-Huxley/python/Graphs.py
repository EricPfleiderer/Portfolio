import matplotlib.pyplot as plt
import numpy as np

from HodgkinHuxley import HodgkinHuxley
from Analysis import Analysis


class Graphs:

    """
    Prints results of various Hodgkin-Huxley model simulations.
    """

    def __init__(self):

        # Initialize the model and solver
        self.model = HodgkinHuxley()

        # Initialize analysis tool
        self.analysis = Analysis()

        # # Q1 Reproduce results to show that the solution is valid
        # self.graphVarEquil()
        # self.graphTau()
        # self.graphReprodSingleImpulse()
        # self.graphPotentialContinuousCurrent()
        #
        # # Q2 Frequency and Amplitude in relation to Amperage
        # self.graphsFrequencyAmplitude()
        #
        # # Q3 Delta min
        # self.graphMinDeltaAmperage()
        #
        # # Q4 Random current application
        # self.graphRandomCurrent()
        #
        # # Q5 Multiple sclerosis (gL vs pulseAmperage, gL vs pulseDuration)
        # self.graphMingLAmperage()
        # self.graphMingLLength()

        # Q6 Cm
        # self.graphCm()
        self.graphMaxCm()
        # self.graphRandomCm()

    def graphCm(self):

        data1 = self.model.solve(7, 1, 1, 20, 40, 0, Cm=2)
        data2 = self.model.solve(14, 1, 1, 20, 40, 0, Cm=2)

        # Plotting
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r"$t\ \ [ms]$")
        ax1.set_ylabel(r"$\phi\ \ [mV]$")
        lines1, = ax1.plot(data1["t"], data1["V"], label=r"$I = 7 \mu A$")
        lines2, = ax1.plot(data2["t"], data2["V"], label=r"$I = 14 \mu A$")
        lines1.set_dashes([2, 2, 10, 2])
        lines2.set_dashes([2, 2, 2, 2])
        plt.legend(loc="best")
        plt.savefig("img/triple_Cm.pdf")

    def graphMaxCm(self):

        xAxis = np.arange(3, 20, 0.5)
        yAxis = [self.analysis.getMaxCm([x, 1, 1, 0, 20, 0]) for x in xAxis]

        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r"$c_m\ \ [\mu F]$")
        ax1.set_xlabel(r"$I_a\ \ $" + r"$[\mu A]$")
        ax1.plot(xAxis, yAxis)
        plt.savefig("img/MaxCm.pdf")

    def graphVarEquil(self):

        # Setting up data
        xAxis = np.arange(-100, 100, 0.01)  # mV
        nEquil = np.array([self.model.n.equil(x) for x in xAxis])
        mEquil = np.array([self.model.m.equil(x) for x in xAxis])
        hEquil = np.array([self.model.h.equil(x) for x in xAxis])

        # Plotting
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r"$\phi\ \ [mV]$")
        ax1.set_ylabel(r"$x_{Eq}\ \ [1]$")
        lines1, = ax1.plot(xAxis, nEquil, label="$n_{Eq}$")
        lines2, = ax1.plot(xAxis, mEquil, label="$m_{Eq}$")
        lines3, = ax1.plot(xAxis, hEquil, label="$h_{Eq}$")
        lines1.set_dashes([2, 2, 10, 2])
        lines2.set_dashes([2, 2, 2, 2])
        lines3.set_dashes([5, 2, 5, 2])
        ax1.legend(loc="best")
        plt.savefig("img/VarEquil.pdf")

    def graphTau(self):

        # Setting up data
        xAxis = np.arange(-100, 100, 0.01)  # mV
        nTau = np.array([self.model.n.tau(x) for x in xAxis])
        mTau = np.array([self.model.m.tau(x) for x in xAxis])
        hTau = np.array([self.model.h.tau(x) for x in xAxis])

        # Plotting
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r"$\phi\ \ [mV]$")
        ax1.set_ylabel(r"$\tau_x\ \ [ms]$")
        lines1, = ax1.plot(xAxis, nTau, label="$\\tau_n$")
        lines2, = ax1.plot(xAxis, mTau, label="$\\tau_m$")
        lines3, = ax1.plot(xAxis, hTau, label="$\\tau_h$")
        lines1.set_dashes([2, 2, 10, 2])
        lines2.set_dashes([2, 2, 2, 2])
        lines3.set_dashes([5, 2, 5, 2])
        ax1.legend(loc="best")
        plt.savefig("img/Tau.pdf")

    # Dessine une seule impulsion. Supporte le courant continue (pulseL)
    def graphPotentialContinuousCurrent(self):

        data1 = self.model.solve(7, 80, 1, 0, 80, 0)
        data2 = self.model.solve(10, 80, 1, 0, 80, 0)

        #Plotting
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r"$t\ \ [ms]$")
        ax1.set_ylabel(r"$\phi\ \ [mV]$")
        lines1, = ax1.plot(data1["t"], data1["V"], label="$I_a = 7$" + r"$\mu$"+"A")
        lines1.set_dashes([2, 2, 10, 2])
        lines2, = ax1.plot(data2["t"], data2["V"], label="$I_a = 10$" + r"$\mu$"+"A")
        lines2.set_dashes([2, 2, 2, 2])
        ax1.legend(loc="best")
        plt.grid(b=True)
        plt.savefig("img/Potentiel_courant_continu.pdf")

    # Replicates figure 1.15 of the course outline
    def graphReprodSingleImpulse(self):

        # 6.92 is the break point for impulsion for chosen parameters
        data = self.model.solve(7, 1, 1, 0, 25, 0)

        fig, ax1 = plt.subplots()

        ax1.set_xlabel(r"$t\ \ [ms]$")
        ax1.set_ylabel(r"$\phi\ \ [mV]$")
        ax1.plot(data["t"], data["V"], label=r"$\phi$")
        ax1.legend(loc="upper left")
        plt.grid(b=True)

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$n, m, h\ \ [1]$")
        lines1, =  ax2.plot(data["t"], data["n"], label="n")
        lines1.set_dashes([2, 2, 10, 2])
        lines2, = ax2.plot(data["t"], data["m"], label="m")
        lines2.set_dashes([2, 2, 2, 2])
        lines3, = ax2.plot(data["t"], data["h"], label="h")
        lines3.set_dashes([5, 2, 5, 2])
        ax2.legend(loc="upper right")
        plt.savefig("img/Vnmh.pdf")

    def graphRandomCurrent(self):

        data = self.model.solve("random", 2, 50, 0, 100, 0)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r"$t\ \ [ms]$")
        ax1.set_ylabel(r"$\phi\ \ [mV]$" + ",    " + r"$I_a\ \ $" + "[" + r"$\mu$"+"A]")
        lines1, = ax1.plot(data["t"], data["V"], label=r"$\phi$")
        lines1.set_dashes([2, 2, 10, 2])
        lines2, = ax1.plot(data["t"], data["Ir"], label="$I_a$")
        lines2.set_dashes([2, 2, 2, 2])
        ax1.legend(loc="best")
        plt.grid(b=True)
        plt.savefig("img/randomCurr.pdf")

    def graphRandomCm(self):
        data = self.model.solve("random", 2, 50, 0, 100, 0, Cm=1/2)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r"$t\ \ [ms]$")
        ax1.set_ylabel(r"$\phi\ \ [mV]$" + ",    " + r"$I_a\ \ $" + "[" + r"$\mu$" + "A]")
        lines1, = ax1.plot(data["t"], data["V"], label=r"$\phi$")
        lines1.set_dashes([2, 2, 10, 2])
        lines2, = ax1.plot(data["t"], data["Ir"], label="$I_a$")
        lines2.set_dashes([2, 2, 2, 2])
        ax1.legend(loc="best")
        plt.grid(b=True)
        plt.savefig("img/randomCm.pdf")

    def graphsFrequencyAmplitude(self):

        frequency, amplitude = [], []

        # Minimum starting value must be 7 due to implementation. (Smoothness required)
        xAxis = np.arange(6.3, 12, 0.15)

        for x in xAxis:
            temp1, temp2 = self.analysis.getFrequence_Amplitude(x)
            frequency.append(temp1), amplitude.append(temp2)

        fig, ax1 = plt.subplots()

        ax1.set_ylabel(r"$f\ \ [kHz]$")
        ax1.set_xlabel(r"$I_a\ \ $" + "[" + "$\mu$"+"A]")
        ax1.ticklabel_format(style='sci', axis='y')
        ax1.plot(xAxis, frequency, label="V")
        plt.savefig("img/Frequency.pdf")

        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r"$\phi_{max}\ \ [mV]$")
        ax1.set_xlabel(r"$I_a\ \ $" + "[" + r"$\mu$" + "A]")
        ax1.plot(xAxis, amplitude, label="V")
        plt.savefig("img/Amplitude.pdf")

    def graphMinDeltaAmperage(self):

        xAxis = np.arange(7, 25, 0.5)
        yAxis = [self.analysis.getMinDeltaDoubleAction(x, 0.01) for x in xAxis]

        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r"$\Delta_{min}\ \ [ms]$")
        ax1.set_xlabel(r"$I_a\ \ $" + "[" + r"$\mu$" + "A]")
        ax1.plot(xAxis, yAxis)
        plt.savefig("img/DeltaMinCurrent.pdf")

    def graphMingLAmperage(self):

        xAxis = np.arange(1, 10, 0.25)
        y1Axis = [self.analysis.getMingL([x, 0.5, 1, 0, 20, 0]) for x in xAxis]
        y2Axis = [self.analysis.getMingL([x, 1, 1, 0, 20, 0]) for x in xAxis]

        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r"$gL_{min}\ \ [ms\ cm^{-2}]$")
        ax1.set_xlabel(r"$I_a\ \ $" + "[" + r"$\mu$"+"A]")
        lines1, = ax1.plot(xAxis, y1Axis, label = "$I_a$" + ",  " + "$t_{I_a} = 0.5 ms$")
        lines1.set_dashes([2, 2, 10, 2])
        lines2, = ax1.plot(xAxis, y2Axis, label = "$I_a$" + ",  " + "$t_{I_a} = 1 ms$")
        lines2.set_dashes([2, 2, 2, 2])
        ax1.legend(loc="best")
        plt.savefig("img/MingLAmperage.pdf")

    def graphMingLLength(self):

        # Specially chosen interval to observe the behavior.
        # To observe longer intervals, reduce amperage in getMingL call
        xAxis = np.arange(0.1, 3.5, 0.2)
        y1Axis = [self.analysis.getMingL([4, x, 1, 0, 20, 0]) for x in xAxis]
        y2Axis = [self.analysis.getMingL([5, x, 1, 0, 20, 0]) for x in xAxis]

        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r"$gL_{min}\ \ [ms\ cm^{-2}]$")
        ax1.set_xlabel(r"$t_{I_a}\ \ $" + "[ms]")
        lines1, = ax1.plot(xAxis, y1Axis, label="$t_{I_a}$" + ",  " + "$I_a = 4$" + r"$\mu$"+"A")
        lines1.set_dashes([2, 2, 10, 2])
        lines2, = ax1.plot(xAxis, y2Axis, label="$t_{I_a}$" + ",  " + "$I_a = 5$" + r"$\mu$"+"A")
        lines2.set_dashes([2, 2, 2, 2])
        ax1.legend(loc="best")
        plt.savefig("img/MingLLength.pdf")


if __name__ == '__main__':
    graph = Graphs()