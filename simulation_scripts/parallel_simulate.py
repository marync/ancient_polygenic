import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

# my code
from Simulate import *
from Parser import *


# start clock
start = time.time()

# parse user arguments
args = parse_user_arguments (sys.argv)
print(args)


# output directory
if not os.path.exists(args.outputdir) :
    os.mkdir(args.outputdir)


def generate_seeds (n, rng) :  # you want to avoid a seed to be a power of 2
    """
    Sets random seed.
    """

    seeds = rng.integers (1, 123456789, n)

    return seeds


def simulate_time_point (time, rng) :
    """
    Simulate allele frequency evolution.
    """

    sim = None

    # generate seeds
    rseeds = generate_seeds (n=args.nsim, rng=rng)
    print('time: ' + str(time) + ', seed: ' + str(rseeds[:10]))
    print('time: ' + str(time) + ', seed: ' + str(rseeds[-1]))

    # create simulation object
    sim = Simulate(N            = args.N,
                   gwasSize     = args.n,
                   tau          = time,
                   effectSize   = args.beta,
                   mu           = args.mutation,
                   nu           = args.mutation,
                   L            = args.L,
                   sigma        = args.sigmae,
                   ploidy       = args.ploidy,
                   s            = args.selcoeff,
                   onset        = args.onset,
                   verbose      = args.verbose)

    sim.simulate (seeds=rseeds, outputdir=args.outputdir, nsim=args.nsim)


# start multiprocessing
multiprocessing.set_start_method("fork")
ss          = np.random.SeedSequence (args.rseed)
child_seeds = ss.spawn (len (args.times))
streams     = [np.random.default_rng (s) for s in child_seeds]

apool = multiprocessing.Pool (int(args.ncores))
apool.starmap(simulate_time_point, zip(args.times, streams))

end = time.time()

print ('\nYour code took this long:')
print(end - start)
print ()
