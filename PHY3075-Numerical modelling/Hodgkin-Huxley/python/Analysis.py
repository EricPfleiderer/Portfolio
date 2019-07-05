import numpy as np

from HodgkinHuxley import HodgkinHuxley


class Analysis:

    def __init__(self):
        self.model = HodgkinHuxley()

    # Returns polyfit of degree n for amplitude vs current intensity.
    def polyfitAmplitude(self, degree, full=True):

        xAxis = np.arange(6.3, 12, 0.15)
        yAxis = []

        for x in xAxis:
            freq, amp = self.getFrequence_Amplitude(x)
            yAxis.append(amp)

        return np.polyfit(xAxis, yAxis, degree, full=full)

    # Returns period of potential oscillations for continuously applied current
    # Can only return frequency and amplitude for a current that will generate an oscillating signal.
    # Is not valid for current under 7 microAmperes.
    def getFrequence_Amplitude(self, amperage):

        # Calculations based on a continuous current applied for 80 ms
        data = self.model.solve(amperage, 80, 1, 0, 80, 0)

        # Time at which function crosses zero
        zeros = self.getZeros(data)

        # Amplitude of each oscillation, ignoring the first
        maximas = self.getMaximas(data)[1:]
        # minimas = self.getMinimas(data)[1:]

        # Calculating period
        averagePeriod = 0
        counter = 0
        for x in range(len(zeros)-2):
            averagePeriod += zeros[x+2] - zeros[x]
            counter += 1
        if counter > 0:
            averagePeriod /= counter

        # Calculating amplitude
        averageAmplitude = 0
        for x in maximas:
            averageAmplitude += x
        if len(maximas) > 0:
            averageAmplitude /= len(maximas)

        # Return frequency and amplitude
        return 1/averagePeriod, averageAmplitude

    # Returns the times for which the potential was null. Assumes the smoothness of the function.
    @staticmethod
    def getZeros(data):

        zeroTimes = []
        for x in range(1, len(data["V"])):
            if data['V'][x-1] < 0 < data['V'][x] or data['V'][x-1] > 0 > data['V'][x]:
                zeroTimes.append((data['t'][x-1] + data['t'][x])/2)

        return zeroTimes

    # Assumes smoothness of the function. Ignores the first maximum.
    @staticmethod
    def getMaximas(data, breakPoint=50):

        maximaTimes = []
        for x in range(1, len(data["V"])-1):
            if data["V"][x-1] < data["V"][x] > data["V"][x+1] and data["V"][x] >= breakPoint:
                maximaTimes.append(data["V"][x])

        return maximaTimes

    # Assumes smoothness of the function. Ignores the first maximum.
    @staticmethod
    def getMinimas(data, breakPoint=-5):

        minimaTimes = []
        for x in range(1, len(data["V"])-1):
            if data["V"][x-1] > data["V"][x] < data["V"][x+1] and data["V"][x] <= breakPoint:
                minimaTimes.append(data["V"][x])

        return minimaTimes

    # Pass the break point value of desired "no impulsions"
    def containsAtLeastNImpulsions(self, data, n=1, breakPoint=50):
        peaks = self.getMaximas(data, breakPoint)
        return len(peaks) >= n

    # Method has better runtime and higher likelyhood of success for underestimations than overestimations on initial gL value.
    # High number of simulations for low epsilon.
    def getMinDeltaDoubleAction(self, amperage, epsilon=0.0025):

        # Search interval
        delta = 10
        step = 5

        # Loop parameters
        deltaFound = False
        escapeHatch = 0
        maxCount = 100

        while not deltaFound:
            print(delta, step)

            if escapeHatch >= maxCount:
                print("Failed to find delta after "+str(maxCount)+" iterations. Consider increasing tolerance epsilon.")
                return

            escapeHatch += 1

            data = self.model.solve(amperage, 1, 2, delta, 50, 0)

            if self.containsAtLeastNImpulsions(data, 2):

                # Accept the solution
                if step <= epsilon:
                    return delta

                # Keep getting closer to the solution
                else:
                    delta -= step
                    step /= 2

            else:
                delta += step

    # Method has better runtime and higher likelyhood of success for underestimations than overestimations on initial gL value.
    # High number of simulations for low epsilon.
    # Parameters is an ordered list of the all parameters but gL.
    def getMingL(self, parameters, epsilon=0.0005):

        # Search interval
        gL = 0
        step = 0.1

        # Loop parameters
        deltaFound = False
        escapeHatch = 0
        maxCount = 250

        while not deltaFound:
            print(gL, step)

            if escapeHatch >= maxCount:
                print("Failed to find delta after "+str(maxCount)+" iterations. Consider increasing tolerance epsilon.")
                return

            escapeHatch += 1

            data = self.model.solve(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], gL)

            # If an activation occurred
            if self.containsAtLeastNImpulsions(data, 1):

                # Accept the solution if the step if small enough
                if step <= epsilon:
                    return gL

                # If not, keep getting closer to the solution
                else:
                    gL -= step
                    step /= 2
            # Step
            else:
                gL += step

    def getMaxCm(self, parameters, epsilon=0.01):

        # Search interval
        Cm = 0.01
        step = 1

        # Loop parameters
        deltaFound = False
        escapeHatch = 0
        maxCount = 250

        while not deltaFound:
            print(Cm, step)

            if escapeHatch >= maxCount:
                print("Failed to find delta after "+str(maxCount)+" iterations. Consider increasing tolerance epsilon.")
                return

            escapeHatch += 1

            data = self.model.solve(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], Cm=Cm)

            # If an activation occurred
            if not self.containsAtLeastNImpulsions(data, 1):
                print("h")

                # Accept the solution if the step if small enough
                if step <= epsilon:
                    return Cm

                # If not, keep getting closer to the solution
                else:
                    Cm -= step
                    step /= 2
            # Step
            else:
                Cm += step

    def compareNbMaximas_Cm(self, nbRuns):

        nbMaximasHalfCm = 0
        nbMaximasFullCm = 0

        for x in range(nbRuns):
            nbMaximasHalfCm += len(self.getMaximas(self.model.solve("random", 2, 50, 0, 100, 0, Cm=1/2)))
            nbMaximasFullCm += len(self.getMaximas(self.model.solve("random", 2, 50, 0, 100, 0, Cm=1)))

        return nbMaximasHalfCm/nbRuns, nbMaximasFullCm/nbRuns
