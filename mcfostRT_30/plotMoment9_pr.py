import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import sys
import pymcfost

plt.figure(1)
image_file3='./data_CO/lines.fits.gz'
bmin=0.05
bmaj=0.05
bpa=0.


CO = pymcfost.line.Line('./data_CO') # Delta_v = 0.02 km/s, DIANA opacity, vmax = 4 km/s

deltav = 0.05
fmax = 0.5e-19

hdu_list=fits.open(image_file3)
hdu_list.info()


NpixX=hdu_list[0].header['NAXIS1']
NpixY=hdu_list[0].header['NAXIS2']
pixscaleX=-hdu_list[0].header['CDELT1']*3600 #in arcsec per pix
pixscaleY=hdu_list[0].header['CDELT2']*3600 #in arcsec per pix

xmax=NpixX*pixscaleX/2.
xmin=-xmax
ymax=NpixY*pixscaleY/2.
ymin=-ymax
xvec=np.linspace(xmin,xmax,NpixX)
yvec=np.linspace(ymin,ymax,NpixY)


CO.plot_map(subtract_cont = True,iTrans = 0, #psf_FWHM = 0.03,  
                 colorbar = True, bmaj = bmaj, bmin = bmin, bpa = bpa, 
                 moment = 9, Delta_v = deltav)

imagedata=CO.last_image

plt.figure(3)
plt.pcolormesh(-xvec,yvec,imagedata,vmin=-2.,vmax=2.,cmap='RdBu_r')
plt.xlim([3.5,-3.5])
plt.ylim([-3,3])

plt.text(3.4, 2.7, r'Moment1 CO 3-2, Synthetic',color='black',fontsize=14)
plt.xlabel('$\\Delta \\alpha$ [\'\']')
plt.ylabel('$\\Delta \\delta$ [\'\']')
ax=plt.gca()
ax.xaxis.label.set_size(17)
ax.yaxis.label.set_size(17)
ax.tick_params(labelsize = 17)
ax.set_aspect('equal')
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 17)
cb.set_label("Velocity [km$\\cdot$s$^{-1}$]", size = 17)


dx,dy=(0.125, 0.125)
beam = Ellipse(
    ax.transLimits.inverted().transform((dx, dy)),
    width=bmin,
    height=bmaj,
    angle=-bpa,
    fill=True,
    color="grey")

ax.add_patch(beam)
plt.savefig('./COM1sim.png')


hdr = fits.Header()
hdr["EXTEND"] = True
hdr["OBJECT"] = "mcfost"
hdr["CTYPE1"] = "RA---TAN"
hdr["CRVAL1"] = 0.0
hdr["CDELT1"] = -pixscaleX / 3600.0
hdr["CUNIT1"] = "deg"

hdr["CTYPE2"] = "DEC--TAN"
hdr["CRVAL2"] = 0.0
hdr["CDELT2"] = pixscaleY / 3600.0
hdr["CUNIT2"] = "deg"

hdr["BMAJ"] = bmaj/3600.
hdr["BMIN"] = bmin/3600.
hdr["BPA"] = bpa

hdu = fits.PrimaryHDU(imagedata, header=hdr)
hdul = fits.HDUList(hdu)

hdul.writeto("M9_30" + ".fits", overwrite=True)

