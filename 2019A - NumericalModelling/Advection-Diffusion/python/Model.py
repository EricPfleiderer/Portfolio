import numpy as np


class Model:

    class Source:

        # Specifiying source location and amplitude
        def __init__(self, S=2, x=0.8*2*np.pi, y=0.6*2*np.pi, w=0.25):
            self.S = S
            self.w = w
            self.x = x
            self.y = y

        # Get polutant amplitude originating from point x,y with source at point self.x, self.y
        def getSource(self, x, y):
            return self.S * np.exp(-((x-self.x)**2 + (y-self.y)**2) / self.w**2)

    class Advection:

        def __init__(self, u=1, A=np.sqrt(6), B=np.sqrt(6), epsilon=1, w=5):
            self.u = u
            self.A = A
            self.B = B
            self.epsilon = epsilon
            self.w = w

        def getXAdv(self, y, t):
            return self.u * self.A * np.cos(y + self.epsilon * np.sin(self.w * t))

        def getYAdv(self, x, t):
            return self.u * self.B * np.sin(x + self.epsilon * np.cos(self.w * t))

    def __init__(self, S=2, x=0.8*2*np.pi, y=0.6*2*np.pi):
        self.source = self.Source(S, x, y)
        self.advection = self.Advection()
