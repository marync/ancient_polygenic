import numpy as np

from PlotTheory import PlotTheory

# parameters
avals = np.array([1e-4,1e-3,1e-2]) # mutation
ns    = np.array([1e3,1e4,1e5]) # GWA study sizes
times = np.arange (0,1.,0.001) # times
vstar = 99.5

# plot for tau in [0,1]
obj = PlotTheory (avals=avals, ns=[1e3,1e4,1e5], times=times, outputdir='../figures/longtime')
obj = PlotTheory (avals=avals, ns=[1e3,1e4,1e5], times=times, outputdir='.')
obj.process (vstar=vstar)

# plot for tau in [0,0.2]
times = np.arange (0,.2,0.001)
obj   = PlotTheory (avals=avals, ns=[1e3,1e4,1e5], times=times, outputdir='../figures/shorttime')
obj.process (vstar=vstar)

