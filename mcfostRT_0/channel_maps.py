import matplotlib.patches as ptch
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import sys
import pymcfost

image_file3="./CASA/convolved_CO_cube.fits"

hdu_list=fits.open(image_file3)
hdu_list.info()

NpixX=hdu_list[0].header['NAXIS1']
NpixY=hdu_list[0].header['NAXIS2']
NpixV=hdu_list[0].header['NAXIS3']
pixscaleX=-hdu_list[0].header['CDELT1']*3600 #in arcsec per pix
pixscaleY=hdu_list[0].header['CDELT2']*3600 #in arcsec per pix
pixscaleV=hdu_list[0].header['CDELT3']

xmax=NpixX*pixscaleX/2.
xmin=-xmax
ymax=NpixY*pixscaleY/2.
ymin=-ymax
xvec=np.linspace(xmin,xmax,NpixX)
yvec=np.linspace(ymin,ymax,NpixY)

vmax=NpixV*pixscaleV/2.
vmin=-NpixV*pixscaleV/2.
vvec=np.linspace(vmin,vmax,NpixV)
data=hdu_list[0].data
#velo=hdu_list[4].data

numlist=range(255,746,35)
vellist=vvec[numlist]

#making string list
vstring=[]
for v in vellist:
    if(v<0):
        vstring.append(str(v)[0:5]+' ${\\rm kms}^{-1}$') 
    else:
        vstring.append(str(v)[0:4]+' ${\\rm kms}^{-1}$')

fig, axs = plt.subplots(3,5,figsize=(17,9), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.01,hspace=0.01)
#plt.subplots_adjust()
i=0

fig.suptitle('RT channel maps, $i=0^\\circ$')

#generate beam
dx,dy=(0.125, 0.125)

for ax in axs.flat:
    im=ax.pcolormesh(xvec,yvec,data[numlist[i]],cmap='inferno')
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.text(-1.,0.95,vstring[i],color='white',\
                horizontalalignment='left')
    ax.set_aspect('equal')

    beam = ptch.Ellipse(
    ax.transLimits.inverted().transform((dx, dy)),
    width=0.05,
    height=0.05,
    angle=0.,
    fill=True,
    color="grey")

    ax.add_patch(beam)
    i=i+1

for i in range(len(axs[0,:])):
    axs[2,i].set_xlabel('$\\Delta \\alpha$ [\'\']')
for i in range(len(axs[:,0])):
    axs[i,0].set_ylabel('$\\Delta \\delta$ [\'\']')

#cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width,height]
#cbar = fig.colorbar(im, cax=cbar_ax)
#cbar.set_label('Flux [${\\rm mJy\\cdot beam}^{-1}$]',rotation=90) 
#cbar=fig.colorbar(im, ax=axs[2,4],anchor=(0.0, 0.0))
#cbar.set_label('Flux [${\\rm mJy}\,{\\rm beam}^{-1}$]', rotation=90)
plt.savefig('./CASA/channel_maps.png')
