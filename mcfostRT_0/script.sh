#!/bin/bash

#mcfost ref3.0_3D.para -phantom ../simEcc2A_00500 -keep_particles 0.99999 -delete_above_latitude 0.4 -tau=1_surface -mol -scale_length_units 15

#if pymcfost
#python plotMoment9.py

python convolve_data_cube.py
python quadratic_moments.py 
python plot_maps_0.py
python create_zoomin.py
