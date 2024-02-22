import bettermoments as bm
import matplotlib.pyplot as plt
import numpy as np

binarysemimaj=15.
distance=130
lorig=1. #current binary size

#rescale=binarysemimaj/lorig

#rescale_simx=rescale/distance #to rescale the sim, after having already being rescaled
rescale_x=1.#binarysemimaj/distance #to rescale things expressed in binary units to arcseconds
rescale_v=1.#29.6/np.sqrt(binarysemimaj) #km/s to rescale velocities to phys units compatible with scale lengths


path = './CASA/convolved_CO_cube.fits'
data, velax = bm.load_cube(path)
data[0]=data[0]*rescale_x
data[1]=data[1]*rescale_x
data[2]=data[2]*rescale_v

smoothed_data = bm.smooth_data(data=data, smooth=1, polyorder=0)
rms = bm.estimate_RMS(data=smoothed_data, N=5)
moments = bm.collapse_quadratic(velax=velax, data=smoothed_data,rms=rms)
bm.save_to_FITS(moments=moments, method='quadratic', path=path)

velmax=2.60
velmin=-velmax
lev=np.linspace(-velmax,velmax,19)

plt.pcolormesh(data[0],data[1],moments)
plt.colorbar()
plt.contour(xnew,ynew,residuals_simmodel,levels=lev2,linewidths=0.5,colors='k')
plt.axis('equal')

