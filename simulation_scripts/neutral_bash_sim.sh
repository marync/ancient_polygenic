#!/bin/bash
#PBS -d .
#PBS -j oe
#PBS -S /bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=16gb
#PBS -l nodes=1:ppn=11

OUTPUTDIR=../rawdata

mkdir $OUTPUTDIR
mkdir $OUTPUTDIR/0.0
mkdir $OUTPUTDIR/0.0/h20.5

python parallel_simulate.py -m 1e-6 -k 5000 -r 351 -N 1e3 -n 1e4 -t 20 -ngen 2000 -L 5000 --ncores 11 -o $OUTPUTDIR/0.0

python parallel_sample.py -m 1e-6 -i $OUTPUTDIR/0.0 -o $OUTPUTDIR/0.0/d1000 -k 5000 -r 1421 -N 1e3 -n 1e5 -L 5000 -h2 1.0 --threshold 1000 -na 100 --ncores 11
python parallel_sample.py -m 1e-6 -i $OUTPUTDIR/0.0 -o $OUTPUTDIR/0.0/d10000 -k 5000 -r 413 -N 1e3 -n 1e5 -L 5000 -h2 1.0 --threshold 10000 -na 100 --ncores 11 

python parallel_sample.py -m 1e-6 -i $OUTPUTDIR/0.0 -o $OUTPUTDIR/0.0/h20.5/d1000 -k 5000 -r 51 -N 1e3 -n 1e5 -L 5000 -h2 0.5 --threshold 1000 -na 100 --ncores 11
python parallel_sample.py -m 1e-6 -i $OUTPUTDIR/0.0 -o $OUTPUTDIR/0.0/h20.5/d10000 -k 5000 -r 17 -N 1e3 -n 1e5 -L 5000 -h2 0.5 --threshold 10000 -na 100 --ncores 11 

