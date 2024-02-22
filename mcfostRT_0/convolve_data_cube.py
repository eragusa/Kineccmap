import matplotlib.pyplot as plt
import pymcfost

mol = pymcfost.line.Line("data_CO")
mol.plot_map(v=0.25, Tb=True)
pymcfost.pseudo_CASA_simdata(mol, iTrans=0, beam=0.03,subtract_cont=True,simu_name='convolved_CO_cube',\
                             Delta_v=0.05, rms=0.000)
