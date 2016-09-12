'''
    Generate data for nonlinear fit exercise.
    Coded by: Christian Alis
    Annotated by: May Lim
'''
    
import numpy as np
import matplotlib.pylab as plt


x = np.linspace(0, 10, 20)
y = np.exp(-x**2/2) + 3*np.exp(-(x-7)**2/(2*1.5**2)) + \
        np.random.normal(scale=0.05, size=(20,))
#Describe the nature of the twin peaks.

#In general, save your data first...
np.savetxt('twinpeak.dat', zip(x,y))

#... before plotting 
plt.plot(x, y, 'o')

#Exercise: put axes labels
plt.show()

