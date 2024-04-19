import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

lengde = 100
n = 200  # Antall ledd, jo flere, jo bedre
tid = 1000
alpha = 1

def regning(x, n, lengde):
    return np.exp(-(x/20)**2) * np.sin(x * (n*np.pi/lengde))

A = np.zeros((n, 1)) #Null matrise
for i in range(1, n+1): #fyller inn
    integrert = lambda x: regning(x, i, lengde)
    A[i-1, 0] = 2/lengde * np.trapz(integrert(np.linspace(0, lengde, num=1000)), np.linspace(0, lengde, num=1000))

x = np.arange(0, lengde+1) 

# Matrise med nuller
u = np.zeros((lengde+1, tid+1))

#Oppdatere plottet
def plotoppdatere(frame, data, line, x):
    line.set_data(x, data[:, frame])
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

ax.set_xlim(0, lengde)
ax.set_ylim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Temperatur')
ax.set_title('Løsning av varmelikningen i sylinder')

for t in range(tid):
    for n in range(1, n+1):
        u[:, t+1] += A[n-1, 0] * np.exp(-alpha * (n*np.pi/lengde)**2 * t) * np.sin(x*n*np.pi/lengde)

ani = FuncAnimation(fig, plotoppdatere, frames=tid+1, fargs=(u, line, x), interval=50, blit=True)

plt.show()
