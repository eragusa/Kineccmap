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

rescale_x=15./130.
aout=10.
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
R=np.sqrt(xgr**2+ygr**2)
mask=np.ones_like(R)
mask[R>aout*rescale_x]=np.nan

print('x of 300,238 is:',xgr[300,238])
print('y of 300,238 is:',ygr[300,238])

extent=1.153846153846154

plt.pcolormesh(xvec,yvec,data[680,:,:]*mask,cmap='inferno')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])
plt.xlabel('$\\Delta \\alpha$ [\'\']')
plt.ylabel('$\\Delta \\delta$ [\'\']')
ax=plt.gca()
ax.xaxis.label.set_size(17)
ax.yaxis.label.set_size(17)
ax.tick_params(labelsize = 17)
ax.set_aspect('equal')
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 17)
cb.set_label('Flux [mJy$\\cdot$beam$^{-1}$]', size = 17)
plt.title('$v=0.72$ km$\\cdot$s$^{-1}$')

plt.savefig('/Users/enricoragusa/Works/eccMap/'+folder+'/mcfost_RT/0deg/highvel_pixel.png')
