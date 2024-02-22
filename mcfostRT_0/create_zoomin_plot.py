import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import sys
import pymcfost

#image_file3="./data_CO/lines.fits.gz"
folder='sim2A500'
image_file3="./CASA/convolved_CO_cube.fits"

hdu_list=fits.open(image_file3)
hdu_list.info()

NpixX=hdu_list[0].header['NAXIS1']
NpixY=hdu_list[0].header['NAXIS2']
NpixV=hdu_list[0].header['NAXIS3']
pixscaleX=-hdu_list[0].header['CDELT1']*3600 #in arcsec per pix
pixscaleY=hdu_list[0].header['CDELT2']*3600 #in arcsec per pix
Dv=hdu_list[0].header['CDELT3']
vmax=Dv*hdu_list[0].header['CRPIX3']
vvec=np.linspace(-vmax,vmax,NpixV)

xmax=NpixX*pixscaleX/2.
xmin=-xmax
ymax=NpixY*pixscaleY/2.
ymin=-ymax
xvec=np.linspace(xmin,xmax,NpixX)
yvec=np.linspace(ymin,ymax,NpixY)

data=hdu_list[0].data*1000.

fig, ax = plt.subplots()

xgr,ygr=np.meshgrid(xvec,yvec)

print('x of 300,238 is:',xgr[300,238])
print('y of 300,238 is:',ygr[300,238])

plt.plot(vvec,data[:,300,238])
plt.xlim([-1.5,1.5])
#plt.ylim([0,3.])
plt.xlabel('Velocity [km$\\cdot$s$^{-1}$]')
plt.ylabel('Flux [mJy$\\cdot$beam$^{-1}$]')
ax=plt.gca()
ax.xaxis.label.set_size(17)
ax.yaxis.label.set_size(17)
ax.tick_params(labelsize = 17)

plt.savefig('/Users/enricoragusa/Works/eccMap/'+folder+'/mcfost_RT/0deg/spectrum_pixel.png')
