# Effondrement
import numpy as np
import matplotlib.pyplot as plt
import math


# Reflect particles into observed space when they go out of bounds
def imposeBorderConditions():
    xiu = np.where(x > L)[0]
    xid = np.where(x < -L)[0]
    yiu = np.where(y > L)[0]
    yid = np.where(y < -L)[0]

    x[xiu] = 2 * L - x[xiu]
    x[xid] = -2 * L - x[xid]
    y[yiu] = 2 * L - y[yiu]
    y[yid] = -2 * L - y[yid]

    vx[xiu] = -vx[xiu]
    vx[xid] = -vx[xid]
    vy[yiu] = -vy[yiu]
    vy[yid] = -vy[yid]


def getDensity():
    i_grid_k = np.int32((x + L) / delta)
    i_grid_l = np.int32((y + L) / delta)
    H, xedges, yedges = np.histogram2d(i_grid_k, i_grid_l, bins=(np.arange(0, M + 1, dtype='int32'), np.arange(0, M + 1, dtype='int32')))

    return H * m / (delta ** 2)


def getPotential():
    density = getDensity()
    pot = np.zeros((M + 1, M + 1))

    for i in range(M + 1):
        for j in range(M + 1):
            d = np.sqrt((i - X - 0.5) ** 2 + (j - Y - 0.5) ** 2)
            pot[i, j] += np.sum(density / d)

    return pot * -(G * m * delta)


def getAcceleration():
    ix, iy = np.int32(np.trunc((x + L) / delta)), np.int32(np.trunc((y + L) / delta))
    forceX = -((potential[ix + 1, iy] - potential[ix, iy]) + (potential[ix + 1, iy + 1] - potential[ix, iy + 1])) / (2 * delta)
    forceY = -((potential[ix, iy + 1] - potential[ix, iy]) + (potential[ix + 1, iy + 1] - potential[ix + 1, iy])) / (2 * delta)

    return forceX / m, forceY / m


def initialize():
    N_a = 500
    deltaM = np.zeros(N_a)
    Mr = np.zeros(N_a)
    Omega = np.zeros(N_a)
    dr = 6*r_D/N_a

    rayon = np.sqrt(x**2+y**2)
    iann = np.int32(rayon/dr)

    for k in range(N):
        deltaM[iann[k]] += m

    Mr[0] = deltaM[0]
    for j in range(1, N_a):
        Mr[j] = Mr[j - 1] + deltaM[j]
        Omega[j] = math.sqrt(G*Mr[j]/(j*dr)**3) * (1-0.9*math.exp(-j*dr/2))
    Omega[0] = Omega[1]

    for k in range(N):
        vrot = Omega[iann[k]] * rayon[k]
        vx[k] = -Omega[iann[k]] * y[k] + np.random.normal(0., vrot * 0.05)
        vy[k] = Omega[iann[k]] * x[k] + np.random.normal(0., vrot * 0.05)


iterations = 5000
everyXIter = 50
G = 2.277e-7
N = 300000
L = 40
M = 65
m = 1.e6 / (N / 1.e5)
delta = 2 * L / M
dt = 0.0002
r_D = 4

x, y = np.random.normal(0., r_D, N), np.random.normal(0., r_D, N)
x, y = np.clip(x, -L + 1e-5, L - 1e-5), np.clip(y, -L + 1e-5, L - 1e-5)
vx, vy = np.zeros(N), np.zeros(N)
Y, X = np.meshgrid(np.arange(0, M), np.arange(0, M))
potential = getPotential()
ax, ay = getAcceleration()

initialize()

for iteration in range(iterations+1):

    print("Iteration:", iteration)

    x = x + vx * dt + 0.5 * ax * dt ** 2
    y = y + vy * dt + 0.5 * ay * dt ** 2

    imposeBorderConditions()

    if iteration % everyXIter is 0:
        potential = getPotential()

    ax_temp, ay_temp = getAcceleration()

    vx = vx + dt / 2 * (ax + ax_temp)
    vy = vy + dt / 2 * (ay + ay_temp)

    ax, ay = ax_temp, ay_temp

    if iteration % 50 is 0:
        plt.figure()
        plt.axis([-L, L, -L, L])
        plt.scatter(x, y, c="black", s=0.0001)
        plt.savefig("imgs/Rotate" + str(iteration) + ".png")
