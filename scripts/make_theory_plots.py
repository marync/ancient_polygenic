import os
import numpy as np

from PlotTheory import PlotTheory

if not os.path.exists ('../figures') :
    os.mkdir ('../figures')

# parameters
avals = np.array([1e-4,1e-3,1e-2]) # mutation
#ns   = np.array([1e3,1e4,1e5]) # GWA study sizes
ns    = np.array([1e4,1e5,1e6]) # GWA study sizes
ds    = np.array([10,100,1000,10000,50000,100000])

focaln = 1e5
times  = np.arange (0,1.,0.001) # times
vstar  = 99.5

# plot for tau in [0,1]
obj = PlotTheory (avals=avals, ns=ns, times=times, outputdir='../figures/longtime')
obj.process (focaln=focaln, focala=1e-3, range_ds=ds, vstar=vstar, showapprox=False)

# make supplementary heatmaps
print('making dense plots.')
obj.make_dense_plots (denseas=np.logspace (-4,0,50), densens=np.logspace (4,6,50),
                      lowbeta=0.1, highbeta=0.5)

# plot bias figures
print('making asymmetric plots.')
obj.plot_asymmetric_bias (avals=[1e-4,1e-3,1e-2,1], d1=1, threshold_as_n=True)
print('making pd plot.')
obj.plot_p3d (avals=[1e-3, 1], ns=np.logspace (4,6,50), ds=np.linspace (1,1e4+1,1000), cmap=['Blues', 'Greys'], log=[False, True])

# plot for tau in [0,0.2]
times = np.arange (0,.2,0.001)
obj   = PlotTheory (avals=avals, ns=ns, times=times, outputdir='../figures/shorttime')
obj.process (focaln=focaln, focala=1e-3, range_ds=ds, vstar=vstar, showapprox=True)

