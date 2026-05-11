import matplotlib.pyplot as plt
import pandas as pd

HARTREE_TO_KJMOL = 2625.5

# Daten einlesen
data1 = pd.read_csv('plotfci_he2.dat', header=None, sep=r'\s+')
data2 = pd.read_csv('plotrmp2_he2.dat', header=None, sep=r'\s+')
data3 = pd.read_csv('plotrhf_he2.dat', header=None, sep=r'\s+')

# Shift & Umrechnung anwenden
ref_fci = data1[1].iloc[-1]
ref_mp2 = data2[1].iloc[-1]
ref_rhf = data3[1].iloc[-1]

data1[1] = (data1[1] - ref_fci) * HARTREE_TO_KJMOL
data2[1] = (data2[1] - ref_fci) * HARTREE_TO_KJMOL
data3[1] = (data3[1] - ref_fci) * HARTREE_TO_KJMOL

# Minima suchen
min1_idx = data1[1].idxmin()
min2_idx = data2[1].idxmin()
min3_idx = data3[1].idxmin()

min1_x, min1_y = data1[0][min1_idx], data1[1][min1_idx]
min2_x, min2_y = data2[0][min2_idx], data2[1][min2_idx]
min3_x, min3_y = data3[0][min3_idx], data3[1][min3_idx]

# Plotten
plt.plot(data1[0], data1[1], marker='o', label='FCI')
plt.plot(data2[0], data2[1], marker='s', label='MP2')
plt.plot(data3[0], data3[1], marker='^', label='RHF')

plt.plot(min1_x, min1_y, marker='*', color='blue', markersize=15, zorder=5)
plt.plot(min2_x, min2_y, marker='*', color='orange', markersize=15, zorder=5)
plt.plot(min3_x, min3_y, marker='*', color='green', markersize=15, zorder=5)

plt.annotate(f'{min1_y:.2f}', xy=(min1_x, min1_y), xytext=(10, -10), textcoords='offset points', color='blue')
plt.annotate(f'{min2_y:.2f}', xy=(min2_x, min2_y), xytext=(10, 10), textcoords='offset points', color='orange')
plt.annotate(f'{min3_y:.2f}', xy=(min3_x, min3_y), xytext=(10, -15), textcoords='offset points', color='green')

plt.legend()
plt.title('PES SCAN He2 (Relative Energies)')
plt.xlabel('Distance (Bohr)')
plt.ylabel('Relative Energy (kJ/mol)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()