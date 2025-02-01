import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read data from a csv file in to numpy arrays
file_name = 'SDS00004.csv'  # put your file name here
time, ch1, ch2 = np.loadtxt(file_name, delimiter=',', skiprows=12, unpack=True)
# set up figure with two axes
fig1 = plt.figure(num=1, figsize=(8, 4))
fig1.clf()
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)
fig1.text(0.4, 0.96, 'Demo: read, plot, fit')
# plot raw data on axis 1
ax1.plot(time, ch1, '.', color='C0', markersize=1, label='ch1')
ax1.plot(time, ch2, '.', color='C1', markersize=1, label='ch2')
ax1.set_xlabel('seconds')
ax1.legend()


# Perform dummy least-square fit of ch2 vs. time data to a straight line
# to get full documentation, type curve_fit? in to the ipython console
# define the function to fit to
def line(t, p1, p2):
    return p1 * t + p2


p_0 = np.array([1., 2.])  # array with initial values of parameters
p_opt, p_conv = curve_fit(line, time, ch2, p_0)  # perform the curve fit
print(p_opt)
# plot the raw data and the fit
ax2.plot(time, ch2, '.', markersize=1, label='ch2 data', color='C1')
ax2.plot(time, line(time, p_opt[0], p_opt[1]), label='fit', color='C2')
ax2.set_xlabel('seconds')
ax2.legend()
# fig1.canvas.draw()
fig1.show()
# fig1.savefig('demo_read_plot_fit.png', dpi=300)
