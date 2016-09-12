'''
    Generate data for nonlinear fit exercise.
    Coded by: May Lim
    Derived from code by: Christian Alis
'''
    
import numpy as np
import matplotlib.pylab as plt

num = np.arange(100)
for value in range(len(num)):
    out = 'dampedocs' + str(num[value]) + '.dat'
    t = np.linspace(0, 3*np.pi, 11)
    y = np.sin(t)*np.exp(-t/10.) + 0.4*np.random.normal(scale=0.3,size=11)
#Describe the nature of the twin peaks.

#In general, save your data first...
    np.savetxt(out, zip(t,y))
#np.savetxt('dampedosc1.dat', zip(t,y))
#... before plotting 
    plt.plot(t, y, 'o')

#Exercise: put axes labels
    plt.show()

