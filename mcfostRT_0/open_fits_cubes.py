import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import sys
import pymcfost

image_file3=sys.argv[1]

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

data=hdu_list[0].data[0][0][0]
#velo=hdu_list[4].data


mutable_object = {} 
fig = plt.figure()
def onclick(event):
    print('you pressed', event.key, event.xdata, event.ydata)
    X_coordinate = event.xdata
    Y_coordinate = event.ydata
    mutable_object['click'] = X_coordinate

cid = fig.canvas.mpl_connect('button_press_event', onclick)
lines = plt.pcolormesh(data[700,:,:])
plt.show()
X_coordinate = mutable_object['click']
print(X_coordinate)
