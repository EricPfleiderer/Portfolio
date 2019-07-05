import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate


def elastique(x, y, vx, vy):
    xiu = np.where(x >= L)[0]
    xid = np.where(x <= -L)[0]
    yiu = np.where(y >= L)[0]
    yid = np.where(y <= -L)[0]

    x[xiu] = 2 * L - x[xiu]
    x[xid] = -2 * L - x[xid]
    y[yiu] = 2 * L - y[yiu]
    y[yid] = -2 * L - y[yid]

    vx[xiu] = -vx[xiu]
    vx[xid] = -vx[xid]
    vy[yiu] = -vy[yiu]
    vy[yid] = -vy[yid]


def new_sigma(x, y, M):

    sigma = np.zeros([M, M])
    i_grid_k = np.int32((x + L) / Delta)  # k ième boîte de chaque particule
    i_grid_l = np.int32((y + L) / Delta)  # l ième boîte de chaque particule
    H, xedges, yedges = np.histogram2d(i_grid_k, i_grid_l,
                                       bins=(np.arange(0, M + 1, dtype='int32'), np.arange(0, M + 1, dtype='int32')))
    sigma = H * metoile / (Delta ** 2)

    return sigma


def sigma_anneau(rayon) :
    return 2*math.pi*sigma_zero*rayon / (1+(rayon/r_H)**alpha)


def sombre_sigma():
    r = np.sqrt((Xs - int(M/2)) ** 2 + (Ys - int(M/2)) ** 2)
    sombre_sig = sigma_zero/(1+(r/r_H)**alpha)

    return sombre_sig


def potential(Xs, Ys, M):

    pot = np.zeros([M + 1, M + 1])  # re-initialisation du potentiel

    for i in range(0, M + 1):
        for j in range(0, M + 1):
            d = np.sqrt((i - Xs - 0.5) ** 2 + (j - Ys - 0.5) ** 2)  # pour une case
            pot[i, j] = np.sum(sigma / d) * (-Ggrav * metoile * Delta)

    return pot


def force(x, y, L, N):

    ix = np.int32(np.trunc((x + L) / Delta))  # coin de la cellule correspondante
    iy = np.int32(np.trunc((y + L) / Delta))

    # force sur chaque particule
    f_x = -((pot[ix + 1, iy] - pot[ix, iy]) + (pot[ix + 1, iy + 1] - pot[ix, iy + 1])) / (2 * Delta)
    f_y = -((pot[ix, iy + 1] - pot[ix, iy]) + (pot[ix + 1, iy + 1] - pot[ix + 1, iy])) / (2 * Delta)

    # accélération de chaque particule
    ax_temp = f_x / metoile
    ay_temp = f_y / metoile

    return ax_temp, ay_temp


def getVRot():

    rayon_temp = np.sqrt(x**2 + y**2)  # Distance from center for each star
    number_rings = 500
    ring_width = L/number_rings
    i_anneau_temp = np.int32(rayon_temp/ring_width)  # Associated ring to each star
    stars_per_ring = np.zeros(number_rings)  # Counts how many stars per ring

    theta = np.arctan2(y, x)  # Theta of each star
    Vrot_particle = -vx * np.sin(theta) + vy * np.cos(theta)  # VRots of each star

    Vrot_ring = np.zeros(number_rings)  # Will contain average VRot per ring

    for n in range(N):
        stars_per_ring[i_anneau_temp[n]] += 1
        Vrot_ring[i_anneau_temp[n]] += Vrot_particle[n]

    # Returns ready-to-be-plotted X and Y axis
    return np.linspace(0, int(number_rings*ring_width), number_rings), Vrot_ring / stars_per_ring


Ggrav = 2.277e-7  # G en kilo parsec **3/ masse du soleil / période solaire **2
N = int(3e5)  # nombre de particules-étoiles
r_D = 4  # largeur du disque gaussien, en kpc
metoile = 1.e6 / (N / 1.e5)  # masse d'une étoile (masse total conservée) ### nb d'étoile /
rayon = np.zeros(N)  # init__ tableau
N_anneau = 500  # nombre d'anneaux
i_number = np.arange(0, 500)  # numéro de chaque l'anneau
L = 40
M = 65
Delta = 2 * L / M
dr = 6 * r_D / N_anneau  # largeur radiale des anneaux soit 6 sigma
dr_j = i_number * dr  # distance radiale intérieure des anneaux
dr_jp1 = (i_number + 1) * dr  # distance radiale extérieure des anneaux

xs_array = np.arange(0, M, 1)
Ys, Xs = np.meshgrid(xs_array, xs_array)

# Temps
temps_total = 1.5  # NOMBRE DE TEMPS
dt = 0.0002  # TAILLE DU PAS DE TEMPS
temps = np.arange(0, temps_total + dt, dt)
par_u = 50  # calcul de potentiel par u

# HALO
sigma_zero = 1.5e8
r_H = 21.5
alpha = 10

x = np.random.normal(0., r_D, N)  # X position of each star
y = np.random.normal(0., r_D, N)  # Y position of each star
x = np.clip(x, -L + 1e-5, L - 1e-5)
y = np.clip(y, -L + 1e-5, L - 1e-5)
rayon2 = np.sqrt(x ** 2 + y ** 2)  # Distance from center of domain for each star
i_anneau = np.int32(rayon2 / dr)  # anneau de chaque particule
number2 = np.bincount(i_anneau, minlength=500)  # quantité de particule dans l'anneau
deltaM = number2 * metoile  # masse dans l'anneau

# Adding mass contribution of dark matter halo
for j in range(N_anneau):
    deltaM[j] += integrate.quad(sigma_anneau, j*dr, (j+1)*dr)[0]

Mr = np.cumsum(deltaM)  # + integrate.quad(sigma_sombre, 0dr, 1dr)[0]  # masse cumulative partant du centre

Omega = np.zeros(N_anneau)  # vitesse angulaire par anneau
Omega[1:] = np.sqrt(Ggrav * Mr[1:] / (dr_j[1:]) ** 3) * (1 - 0.9 * np.exp(-dr_j[1:] / 2))
Omega[0] = Omega[1]

vrot = Omega[i_anneau] * rayon
vx = -Omega[i_anneau] * y + np.random.normal(0., vrot * 0.05, N)
vy = Omega[i_anneau] * x + np.random.normal(0., vrot * 0.05, N)

# Initialisations
sigma = new_sigma(x, y, M)
sigma_sombre = sombre_sigma()
pot = potential(Xs, Ys, M)
ax, ay = force(x, y, L, N)

# Main loop
for itera in range(1, len(temps)):

    print(itera)

    x = x + vx * dt + 0.5 * ax * dt ** 2
    y = y + vy * dt + 0.5 * ay * dt ** 2

    elastique(x, y, vx, vy)

    if itera % 50 == 0:
        sigma = new_sigma(x, y, M) + sombre_sigma()
        pot = potential(Xs, Ys, M)

    ax_temp, ay_temp = force(x, y, L, N)

    vx = vx + 0.5 * (ax + ax_temp) * dt
    vy = vy + 0.5 * (ay + ay_temp) * dt

    ax = ax_temp
    ay = ay_temp

    if itera % 250 == 0:
        plt.figure(figsize=(5, 5))
        plt.plot(x, y, 'k.', markersize='0.3')
        plt.ylabel('Y', fontsize=16)
        plt.xlabel('X', fontsize=16)
        plt.axis((L, -L, L, -L))
        plt.savefig('imgs/galaxy'+str(itera)+".png", bbox_inches='tight')

        xAxis, Vrot = getVRot()
        plt.figure()
        plt.ylabel('$V_{rot}$', fontsize=16)
        plt.xlabel('r', fontsize=16)
        plt.plot(xAxis, Vrot, color='black')
        plt.savefig('imgs/VROT'+str(itera)+".png")
