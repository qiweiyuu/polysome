# Reaction-diffusion model for nucleoid segregation in *Escherichia coli*

This repository contains the code used to simulate a minimal reaction-diffusion model for nucleoid segregation in *E. coli*, as described in the following paper
+ A. Papagiannakis, Q. Yu, S. K. Govers, W-H Lin, N. S. Wingreen, C. Jacobs-Wagner, **DNA/polysome phase
separation and cell width confinement couple nucleoid segregation to cell growth in Escherichia coli**, *bioRxiv* 2024.10.08.617237 (2024). [[bioRxiv link]](https://doi.org/10.1101/2024.10.08.617237) [[eLife link]](https://elifesciences.org/reviewed-preprints/104276)

The model takes into account two ingredients important for nucleoid segregation: effective repulsion between DNA and polysomes from steric effects (described by the Cahn-Hilliard theory) and the nonequilibrium processes of polysome synthesis and degradation (described by linear reaction kinetics).  Further details can be found in the paper. 

This repository contains the following:
+ `polysome_phase_field.py`: the main script that simulates the model.
+ `Fig4.ipynb`: a Jupyter notebook to reproduce the main results in Figure 4.
+ `mixed_polysome.ipynb`: a Jupyter notebook to reproduce Figure 4, supplement 4, which considers multiple polysome species.
+ 'ectopic.ipynb': a Jupyter notebook to reproduce Figure 7, supplement 2, which considers ectopic polysome production. 