'''
2023.09.25 修改中...
no ts0
'''
import numpy as np
import matplotlib.pyplot as plt

def calculate_E(k, a, ts0, t, matrix_size):
    eigenvalue_results = []
    for i in range(len(k)):
        M = np.zeros((matrix_size, matrix_size), dtype=complex)
        for column in range(matrix_size):
            for row in range(matrix_size):

                if column %2 == 0 and (row - column) == 1:
                    M[column, row] = -t * np.exp(-1j * k[i] * a) - t
                elif (column) %2 == 1 and (column - row) == 1:
                    M[column, row] = -t * np.exp(1j * k[i] * a) - t
                elif (row%4==1) and (column - row)==2 :
                    M[column, row] = -t
                elif (column%4==0) and column!=0 and (column - row)==2 :
                    M[column, row] = -t
                elif (column%4==1) and (row - column)==2 :
                    M[column, row] = -t
                elif (row%4==0) and row!=0 and (row - column)==2 :
                    M[column, row] = -t
                else:
                    M[column, row] = 0

        eigenvalue = np.linalg.eigvalsh(M)
        eigenvalue_results.append(eigenvalue)

    return eigenvalue_results


# Parameters
t = 1
ts0 = 0.1
a = 1
r = 0.01 
#r = 1e-4
k_points_number = 10000
k = np.linspace(-np.pi, np.pi, k_points_number)
dk = 2*np.pi/k_points_number

matrix_size = 10
dos_w = np.linspace(-4, 4, 3000)
jdos_w = np.linspace(0, 8, 5000)

# calculate bands
bands = np.array(calculate_E(k, a, ts0, t, matrix_size))

plt.figure(figsize=(10,6))
plt.plot(k, bands, 'r')
plt.xlabel('k(ev)')
plt.ylabel('E(k)')
plt.title("band")
plt.grid(True)
plt.show()


def dos_func(Ek, w):
    return (dk/(2*np.pi*np.pi)) * (r/ ((w-Ek)**2 + (r)**2))

vec_func = np.vectorize(dos_func)

def compute_DOS(bands, w_values):
    D_results = []
    for w_val in w_values:
        result = np.sum(vec_func(bands, w_val))
        D_results.append(result)
    return np.array(D_results)

# Compute DOS (所有能帶)
dos = compute_DOS(bands, dos_w)


# Plot DOS
plt.figure(figsize=(10,6))
plt.plot(dos_w, dos)
plt.title("Zigzag Graphene DOS")
plt.xlabel('w')
plt.ylabel('DOS(w)')
plt.grid(True)
plt.show()

# calculate Ec & Ev
bands_sorted = np.sort(bands, axis=1)
Ec = bands_sorted[:, matrix_size//2:]   # positive
Ev = bands_sorted[:, :matrix_size//2]   # negative

# JDOS function
def JDOS_func(Ec, Ev, w, r, dk):
    sum_jdos = 0
    for i in range(matrix_size//2):
        for j in range(matrix_size//2):
            sum_jdos += np.sum((dk/(2*np.pi*np.pi)) * (r/ ((w - (Ec[:, i] - Ev[:, j]))**2 + r**2)))
    return sum_jdos

# Computing JDOS 
jdos = [JDOS_func(Ec, Ev, w_val, r, dk) for w_val in jdos_w]

# Plot JDOS
plt.figure(figsize=(10,6))
plt.plot(jdos_w, jdos)
plt.title("Zigzag Graphene JDOS")
plt.xlabel('w')
plt.ylabel('JDOS(w)')
plt.grid(True)
plt.show()
