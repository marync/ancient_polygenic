# ancient_polygenic
Code used to create the figures and conduct the simulations in Carlson et al. "Polygenic score accuracy in ancient samples: quantifying the effects of allelic turnover."

# Contents

1. The **scripts** directory contains the following files:
    - *EqualTheory.py*: Computes all of the quantities derived in the text.
    - *PlotTheory.py*: Class called by the plotting scripts to produce plots containing theoretical results.
    - *make_selection_plots.py*: Takes input from the *data* directory to make Figures 3, 4
    and S7 in the main (first two) and supplementary (third) texts.
    - *make_theory_plots.py*: Makes all other figures, or those only containing theoretical results.
    - *ukbiobank.py*: Makes the two figures in Supplementary Information S1.8 on the UK
    Biobank summary statistics.
        - Requires download of 'body height' data from the Price lab's site: https://alkesgroup.broadinstitute.org/UKBB.
    - *plot_simulations_helper.py*: Has a few functions that are used in plotting.
2. The **data** directory contains the following sub-directories:
    - *heritability*: Contains the simulation results for two values of heritability and
    a threshold of 10,000, each in their own directories.
    - *selection*: Contains the simulation results for the different selection coefficients,
    for two thresholds, 1,000 and 10,000.
    - Each of these sub-directories has a *times.csv* file which records the simulated
    time points and is required for plotting.
3. The **simulation_scripts** directory contains the following files:
    - *Allele.py*, *Ancient.py*, *Simulate.py*: Generate the allele frequency trajectories.
    - *Gwas.py*, *SamplingLarge.py*: Generate the ancient sample and conduct the GWA study, and output 
    realizations of the statistics.
    - *Parser.py*: Self-explanatory.
    - *parallel_sample.py* and *parallel_simulate.py*: Parallelize the sampling and simulation
    procedures, respectively.
    - *\*\_bash\_sim\.sh*: Run the simulations on the Gardner Computing Cluster at University of Chicago.
