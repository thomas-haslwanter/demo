""" Example of how to detect maxima/minima in a noisy sine-wave """

# author: Thomas Haslwanter
# 2024-10-29

# Import the required packages
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy import signal


# Set the parameters
dt = 0.1
noise_amp = 0.1

# Generate noisy data
t = np.arange(0, 20, dt)
x = np.sin(t)
noise = np.random.randn(len(x)) * noise_amp
x_noisy = x + noise

# Calculate a smooth velocity. The window_length has to be found empirically
vel = signal.savgol_filter(x_noisy, window_length=41, polyorder=3, deriv=1,
                           delta=dt)

# Find th zero_crossings in the velocity
prod = vel[:-1] * vel[1:]
zero_crossings = np.where(prod<0)[0]

# To decide between maxima and minima, you can either use the second derivative ...
acc = signal.savgol_filter(x, 41, 3, deriv=2, delta=0.1)
maxima = zero_crossings[acc[zero_crossings]<0]
minima = zero_crossings[acc[zero_crossings]>0]

print(f'Maxima: {maxima}')
print(f'Minima: {minima}')

# ... or you can also say that maxima have to lie above zero, minima below:
maxima_2 = zero_crossings[x_noisy[zero_crossings]>0]
minima_2 = zero_crossings[x_noisy[zero_crossings]<0]

print(f'Maxima, 2nd algorithm: {maxima_2}')
print(f'Minima, 2nd_algorithm: {minima_2}')

# Plot the results
plt.plot(x_noisy)

for loc in maxima:
    plt.axvline(loc, ls='dashed', color='C1')

for loc in minima:
    plt.axvline(loc, ls='dashed', color='C2')

plt.show()

