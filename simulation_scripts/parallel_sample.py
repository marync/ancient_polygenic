import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

# my code
from Simulate import *
from SamplingLarge import *
from Parser import *

# start clock
start = time.time()

print(sys.argv)

# arguments
args = parse_user_arguments (sys.argv)
print(args)
print()


# output directory
if not os.path.exists(args.outputdir) :
    os.mkdir(args.outputdir)

np.random.seed (args.rseed)

# find the times
times = np.sort(np.array([int(t.split('_')[1].split('.')[0]) for t in os.listdir (args.inputdir) if 'ancient_' in t]))
print(times)

# generate random seeds and save them
seeds = np.random.randint (1, 123456789, len(times))
print(seeds)
np.savetxt (fname=os.path.join(args.outputdir, "seeds.csv"), fmt="%i", X=seeds, delimiter=",")


def sample_time_point (time, rseed) :
    """
    Gwas sampling at time points.
    """

    ancient = None
    contemp = None
    sample  = None

    ancient = np.loadtxt (os.path.join(args.inputdir, 'ancient_' + str(time) + '.csv'), delimiter=',')
    contemp = np.loadtxt (os.path.join(args.inputdir, 'contemp_' + str(time) + '.csv'), delimiter=',')

    print(ancient.shape)

    sample = SamplingLarge (N            = args.N,
                            gwasSize     = args.n,
                            epsilon      = args.threshold,
                            beta         = args.beta,
                            L            = args.L,
                            sigma        = args.sigmae,
                            tau          = time,
                            ploidy       = args.ploidy)

    #sample.simulate (nsim=args.nsim, ancient=ancient, contemp=contemp,
    #                 seed=rseed+10, outputdir=args.outputdir)

    data = sample.simulate (time=time, nsim=args.nsim, ancient=ancient, contemp=contemp,
                            seed=rseed+10, na=args.na, outputdir=args.outputdir)

    return data


start = time.time ()

# start multiprocessing
multiprocessing.set_start_method("fork")
apool = multiprocessing.Pool (int(args.ncores))
simulations = apool.starmap(sample_time_point, zip(times, seeds))


len(simulations)
allsims = np.row_stack (simulations)
np.savetxt (fname=os.path.join(args.outputdir, 'allsims.csv'), X=allsims, delimiter=',',
            header='time,bias,s_bias,mse,s_mse,eva,s_eva,va,s_va,rho_tau,r2,s_r2,rho2_trait', comments='')

for i in simulations :
    print(i)



end = time.time()

print ('\nYour code took this long:')
print(end - start)
print ()
