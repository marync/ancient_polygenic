import sys
import argparse
import numpy as np

def parse_user_arguments (arguments) :

    parser = argparse.ArgumentParser ()

    parser.add_argument ("-m", "--mutation", dest='mutation', type=float, default=None)
    parser.add_argument ("-s", "--selection", dest='selcoeff', type=float, default=None)
    parser.add_argument ("--ncores", dest="ncores", type=int, default=1)
    parser.add_argument ("-o", "--output", dest="outputdir", type=str, default='.')
    parser.add_argument ("-i", "--input", dest="inputdir", type=str, default=None)
    parser.add_argument ("-k", "--nsim", dest="nsim", type=int, default=1000)
    parser.add_argument ("-r", "--rseed", dest="rseed", type=int)
    parser.add_argument ("-N", "--populationsize", dest="N", type=float, default=1e3)
    parser.add_argument ("-n", "--gwasize", dest="n", type=float, default=1e4 )
    parser.add_argument ("-b", "--beta", dest="beta", type=float, default=1 )
    parser.add_argument ("-p", "--ploidy", dest="ploidy", type=int, default=2 )
    parser.add_argument ("-t", "--ntimepoints", dest="ntimes", type=int, default=None )
    parser.add_argument ("-ngen", "--ngenerations", dest="ngen", type=int, default=None )
    parser.add_argument ("-L", "--nloci", dest="L", type=int, default=1e3 )
    parser.add_argument ("-e", "--sigma", dest="sigmae", type=float, default=0 )
    parser.add_argument ("--onset", dest="onset", type=int, default=None )
    parser.add_argument ("-v", "--verbose", dest="verbose", type=bool, default=None )
    parser.add_argument ("-h2", "--heritability", dest="h2", type=float, default=None )
    parser.add_argument ("--threshold", dest="threshold", type=int, default=None )
    parser.add_argument ("-na", "--ancientsize", dest="na", type=int, default=None )

    args = parser.parse_args ()

    if args.mutation is not None :
        scale_rates (args)
        print(args.mutation)
        print(args.selcoeff)

    if args.selcoeff is not None :
        if args.onset is None :
            print ('If selection coefficient is specified, you need to provide a time of onset.')
            sys.exit ()

    if args.h2 is not None :
        set_sigmae (args)

    if args.ntimes is not None :
        args.times = set_times (args)

    return args



def set_sigmae (parserObject) :

    a  = 4.*parserObject.N*parserObject.mutation
    va = (a / (2.*a + 1.)) * parserObject.L * (parserObject.beta**2)
    print('va: ' + str(va))


    sigmae  = (1. - parserObject.h2) / parserObject.h2
    sigmae *= va 
    parserObject.sigmae = sigmae
    print ('Set the environmental variance to: ' + str(sigmae))

    # check
    print('Heritability should be: ' + str(parserObject.h2))
    print('And, it is: ' + str(va / (va + sigmae)))


def scale_rates (parserObject) :

    parserObject.mutation = 0.25*parserObject.mutation
    if parserObject.selcoeff is not None :
        parserObject.selcoeff = 0.25*parserObject.selcoeff



def set_times (parserObject) :

    interval = int( parserObject.ngen / parserObject.ntimes )
    times    = np.arange (0, parserObject.ngen + 1, interval)

    return times
