# Kineccmap
A tool for calculating kinematic maps for eccentric discs

run velofield2.py name-of-dump --- Calculate the azimuthal eccentric structure.
NB: the name-of-dump is needed as the scripts produces from scratch the pixel files for density and velocity, however
the datasim.h5 file for the simulation must be available and calculated in the folder. The script automatically matches the
time of the dump with the reference time in the datasim.h5 analysis.
REFERENCE LIB: genvecc.py

run velovert.py  name-of-dump --- Calculate the vertical structure
Open vertical_structure.py check thar R_ref, H0 and l are correct for your sim (note that in some sim R_ref!=R_in)
NB: same considerations as above concerning the name-of-dump.
REFERENCE LIB: vertical_structure.py

run plot_ellipses.py --- Plots a set of nested ellipses with e(a) and varpi(a) read from the datasim.h5 in the folder

run generate_ellipses_xy.py --- similar to velofield2.py but using power-law for e(a) and linear varpi(a). To change it change
the parameters in the preamble of the file.

run harmonic_solve.py solves an implementation of the equation for the vertical motion in the paper.

run velo_vert_model.py name-of-dump --- produces a model with velocity, vertical structure, and inclines, and rescale it to obtain a precise mock model of the system
