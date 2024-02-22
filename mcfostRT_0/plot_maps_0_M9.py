import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import scipy
import numpy as np
from numpy import random
import sys 
import pickle
import interpolation as itp
from astropy.io import fits


#filename='./12CO/30deg/quadratic/12co_30deg_v0.fits'
filename='./M9_0.fits'

binarysemimaj=15.
distance=130
lorig=1. #current binary size

rescale=binarysemimaj/lorig

rescale_simx=rescale/distance #to rescale the sim, after having already being rescaled
rescale_x=binarysemimaj/distance #to rescale things expressed in binary units 
rescale_v=1./np.sqrt(binarysemimaj) #to rescale velocities

msun = 1.99e33
gcgs = 6.67e-8
au = 1.5e13
h = 6.6261e-27
kb = 1.38e-16
mjup = 1.898E30
c = 3e10
plt.rc('xtick', labelsize=16.5) 
plt.rc('ytick', labelsize=16.5) 
plt.rcParams.update({'font.size': 14.75})
fontw = {'family' : 'serif','sans-serif': 'Serif'}
Jy =  1E-23
pc = 3.086e+18
sigmasb = 5.67e-5
lsun = 3.9e33
mp = 1.66e-24
from astropy.io import fits

# example with gaussian map of 12co , 30deg
hdu_list=fits.open(filename)
vfield = 1.*hdu_list[0].data #/ 1000 # m/s -> km/s

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


acav_u=2.2 #cavity in binary semi-maj axis units
aout=10.7
incldisc=0.*3.14/6.
velomax=0.25#*30.*np.sqrt(1./acav_u/binarysemimaj)*np.sin(incldisc) #30 km/s rescaled sqrt(1/R)
# read header to fix the extent of the image
# by changing this value you're basically deciding the size of the cavity (fixing the distance)
extent = aout*rescale_x #fits.open(filename)[0].header['CDELT2'] * 1024 * 3600

#PLOT
#fig, (ax) = plt.subplots(1,1,figsize=(8,7))

#we need to include a rescaling factor for the velocities due to the 

plt.pcolormesh(xvec,yvec,vfield, cmap='RdBu_r',vmin = -velomax, vmax=velomax)
#plt.imshow(m1_12co_30deg_g, cmap='seismic',vmin = -6.0, vmax=6.0,origin='lower')
plt.colorbar(label=r'$v$ [km/s]')

lev=np.linspace(-velomax,velomax,19)

plt.contour(xvec,yvec,vfield,levels=lev, linewidths=0.5, colors='k')
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])

#generate cavity ellipse
incl=incldisc #inclination of the ellipse
acav=acav_u*rescale_x
eccx=0.25
varpi=2.5
theta=np.linspace(0,6.28,100)
r=acav*(1-eccx**2)/(1+eccx*np.cos(theta-varpi))
x=r*np.cos(theta)
y=r*np.sin(theta)*np.cos(incl)
plt.plot(x,y,color='orange')

#find ellipse centre
xmax_e=np.max(x)
xmin_e=np.min(x)
ymax_e=np.max(y)
ymin_e=np.min(y)
centre=(xmin_e+(xmax_e-xmin_e)/2.,ymin_e+(ymax_e-ymin_e)/2.)
#plt.scatter(0.1,0,marker='.', s=6000, c='white') 
# la maschera sistemala come preferisci, io ho messo dei valori a caso per x e y
cav=ptch.Ellipse(centre,height=(ymax_e-ymin_e),width=(xmax_e-xmin_e),angle=varpi,alpha=1.,fill=True,color='white',zorder=2)
#ax.add_patch(cav)

interpolate_RT=itp.interpolator_2D(xvec,yvec,vfield)

file_path = './interpolate_RT_0_M9.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(interpolate_RT, file)

#generate outer disc
a_u=10.
aout=a_u*rescale_x
eccx=0.
varpi=0.
theta=np.linspace(0,6.28,100)
r=aout*(1-eccx**2)/(1+eccx*np.cos(theta-varpi))
x=r*np.cos(theta)
y=r*np.sin(theta)*np.cos(incl)
plt.plot(x,y,color='orange')



#generate beam
dx,dy=(0.125, 0.125)
beam = ptch.Ellipse(
    ax.transLimits.inverted().transform((dx, dy)),
    width=0.1*rescale_simx,
    height=0.1*rescale_simx,
    angle=0.,
    fill=True,
    color="grey")
ax.add_patch(beam)

plt.ylabel(r'$\Delta \delta$ [$^{\prime \prime}$]')
plt.xlabel(r'$\Delta \alpha$ [$^{\prime \prime}$]')
plt.tight_layout()
plt.savefig("vz_mcfost12CO_30.pdf")

#plot vertical scale with surface of last emission
image_file4='./data_CO/tau=1_surface.fits.gz'

hdu_list=fits.open(image_file4)
hdu_list.info()
height=hdu_list[0].data[2][0][0]
xgr=hdu_list[0].data[0][0][0]
ygr=hdu_list[0].data[1][0][0]
R=np.sqrt(xgr**2+ygr**2)

interpolate_HR=itp.interpolator_2D(xvec,yvec,np.array(height/rescale,dtype=float))

file_path = './interpolate_H_0.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(interpolate_HR, file)

plt.figure(2)
plt.pcolormesh(xvec,yvec,height/R,vmin=0.0,vmax=0.5,cmap='inferno')
plt.colorbar()
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])

xgr2,ygr2=np.meshgrid(xvec,yvec)
plt.figure(3)
plt.pcolormesh(xvec,yvec,vfield-interpolate_RT((xgr2,ygr2)), 
                           cmap='RdBu_r',vmin = -velomax*1.1, vmax=velomax*1.1)
#plt.imshow(m1_12co_30deg_g, cmap='seismic',vmin = -6.0, vmax=6.0,origin='lower')
plt.colorbar(label=r'$v$ [km/s]')

lev=np.linspace(-velomax,velomax,19)

#plt.contour(xvec*rescale_simx,yvec*rescale_simx,vfield*rescale_v,levels=lev, linewidths=0.5, colors='k')
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])



plt.show()
