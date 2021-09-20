import numpy as np

from PlotTheory import PlotTheory

# parameters
avals = np.array([1e-4,1e-3,1e-2]) # mutation
#ns   = np.array([1e3,1e4,1e5]) # GWA study sizes
ns    = np.array([1e4,1e5,1e6]) # GWA study sizes
ds    = np.array([10,100,1000,10000,50000,100000])
times = np.arange (0,1.,0.001) # times
vstar = 99.5

# plot for tau in [0,1]
obj = PlotTheory (avals=avals, ns=ns, times=times, outputdir='../figures/longtime')
obj.process (range_ds=ds, vstar=vstar)

# plot for tau in [0,0.2]
times = np.arange (0,.2,0.001)
obj   = PlotTheory (avals=avals, ns=ns, times=times, outputdir='../figures/shorttime')
obj.process (range_ds=ds, vstar=vstar)

