import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import discEccanalysis_pysplash as de
import genvecc as gv
import vertical_structure as vs
import useful_param as up
from rotations import rot_x,rot_y,rot_z
import pickle
import interpolation as itp

img=up.img
#folderres='./analysis_paper'
folderres=up.folderres
name=sys.argv[1]
i0=0./180*np.pi#up.i0
PA0=up.PA0
aout=up.aout

#Rescale for mock image
binarysemimaj=15.
distance=130
lorig=1. #current binary size

rescale=binarysemimaj/lorig

rescale_simx=rescale/distance #to rescale the sim, after having already being rescaled
rescale_x=binarysemimaj/distance #to rescale things expressed in binary units to arcseconds
rescale_v=30./np.sqrt(binarysemimaj) #km/s to rescale velocities to phys units compatible with scale lengths




os.system("splash -p nonlog "+name+" -o ascii -r 6 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 7 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 8 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 9 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 14 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 15 -dev /png")

#load quantities
density=np.loadtxt(name+'_columndensity_proj.pix')
vx=np.loadtxt(name+'_vx_proj.pix')
vy=np.loadtxt(name+'_vy_proj.pix')
z=np.sqrt(np.loadtxt(name+'_z2_proj.pix'))
vz=np.loadtxt(name+'_vz1_proj.pix')
#vz=-vz #for consistency with sign convention in observations

#vz=np.sqrt(np.loadtxt(name+'_vz2_proj.pix'))#-vzm**2)
#vz=np.loadtxt(name+'_vz1_proj.pix')
#vz=(np.loadtxt(name+'_vz4_proj.pix'))**(1./4.)
#create x-y axes
with open(name+'_columndensity_proj.pix') as f:
    for i in range(20):
        xstr=f.readline()
        if('x axis' in xstr):
            loc=xstr.find('min = ')
            xmin=float(xstr[loc+6:loc+6+14])
            loc=xstr.find('max = ')
            xmax=float(xstr[loc+6:loc+6+14])
        if('y axis' in xstr):
            loc=xstr.find('min = ')
            ymin=float(xstr[loc+6:loc+6+14])
            loc=xstr.find('max = ')
            ymax=float(xstr[loc+6:loc+6+14])
        if('time' in xstr):
            loc=xstr.find('time = ')
            time=float(xstr[loc+7:loc+7+14])


zmin=xmin
zmax=xmax
nx=density.shape[1]
ny=density.shape[0]
x=np.linspace(xmin,xmax,nx)
y=np.linspace(ymin,ymax,ny)

xgr,ygr=np.meshgrid(x,y)

res=de.loadHDF5('datasim.h5')
index,t=de.matchtime(res['time'],np.array([time]))

ecc=np.abs(res['evecA'][index[0],:])
phase=np.angle(res['evecA'][index[0],:])
radii=res['radProf'][:]

mass=[]
for i in range(res['discfracA'][index[0]].shape[0]):
    mass.append(sum(res['discfracA'][index[0],:i]*res['Mdisc'][index[0]]))

Ma=np.gradient(mass,radii)

sigma=res['sigmaA'][index[0],:]
wheremax=np.nonzero(sigma==np.max(sigma))

#phase=np.ones(len(ecc))*phase[wheremax[0]]
x0v,v0v,selectxya,a,e,cosvarpi,sinvarpi,deda,dvpda,sigma_a,Ma_a,dPda1rhoa_a=gv.generate_velocity_map(x,y,ecc,phase,sigma,Ma,radii,nprocs=20,aout=10.)

Rgr=np.sqrt(xgr**2+ygr**2)
thetagr=np.arctan2(ygr,xgr)
#defining value of varpi and e for all cells
varpi=np.arctan2(sinvarpi(a),cosvarpi(a))
eccentricity=e(a)

vxcirc,vycirc,vmod=gv.vcircular(Rgr,thetagr)

Rplan=Rgr.reshape(nx*ny)[selectxya]
vyplan=vy.reshape(nx*ny)[selectxya]
vxplan=vx.reshape(nx*ny)[selectxya]
vzplan=vz.reshape(nx*ny)[selectxya]
zplan=z.reshape(nx*ny)[selectxya]
densityplan=density.reshape(nx*ny)[selectxya]
xgrplan=xgr.reshape(nx*ny)[selectxya]
ygrplan=ygr.reshape(nx*ny)[selectxya]
vr,vphi=gv.vxvy2vrvphi(xgrplan,ygrplan,v0v[0,:],v0v[1,:])
vrsim,vphisim=gv.vxvy2vrvphi(xgrplan,ygrplan,vxplan,vyplan)
#calculate H and vz teor
H,vz=vs.calculate_vertical_structure(xgrplan,ygrplan,a,e,cosvarpi,sinvarpi,sigma)
#vz=-vz #for consistency with sign convention

#phi=np.linspace(0,6.28,len(H[0]))
#for i in range(len(H)):
#    plt.figure(1)
 #   plt.plot(phi,H[i])
 #   plt.figure(2)
 #   plt.plot(phi,vz[i])

xmin=xmin*rescale_x
ymin=ymin*rescale_x
zmin=zmin*rescale_x

xmax=xmax*rescale_x
ymax=ymax*rescale_x
zmax=zmax*rescale_x

plt.figure(1)

x0=xgrplan*1. #*1. to make a copy of xgrplan
y0=ygrplan*1.
z0=H*1.

corrector=3.5#corrector for matching the last emission surface in units of H

#Create arrays for applying rotations and mirror the disc also on the negativ z-axis
xv=np.array([x0,y0,z0*corrector])*rescale_x
xvbottom=np.array([x0,y0,-z0*corrector])*rescale_x
vv=np.array([v0v[0,:],v0v[1,:],vz*corrector])*rescale_v
vvbottom=np.array([v0v[0,:],v0v[1,:],vz*corrector])*rescale_v

#eccentric models with no vertical height and velocity
vvz0=np.array([v0v[0,:],v0v[1,:],0.*v0v[0,:]])*rescale_v


#xcirc
zcirc=vs.Hcirc(np.sqrt(x0**2+y0**2))
xvcirc=np.array([x0,y0,zcirc*corrector])*rescale_x
xvcircbottom=np.array([x0,y0,-zcirc*corrector])*rescale_x
xvcirc0=np.array([x0,y0,0.*x0])*rescale_x #no vertical displacement

#vcirc
vycircplan=vycirc.reshape(nx*ny)[selectxya]
vxcircplan=vxcirc.reshape(nx*ny)[selectxya]
vcircv=np.array([vxcircplan,vycircplan,0.*vxcircplan])*rescale_v

xvsim=np.array([xgrplan,ygrplan,zplan*corrector])*rescale_x
xvsimbottom=np.array([xgrplan,ygrplan,-zplan*corrector])*rescale_x
vvsim=np.array([vxplan,vyplan,vzplan*corrector])*rescale_v
vvsimbottom=np.array([vxplan,vyplan,-vzplan*corrector])*rescale_v

#pressure corrected velocities
vr,vphi=gv.vxvy2vrvphi(xgrplan,ygrplan,v0v[0,:],v0v[1,:])
vphi_press=gv.pressure_corrected_vphi(a,vphi,dPda1rhoa_a)
vx_press,vy_press=gv.vrvphi2vxvy(xgrplan,ygrplan,vr,vphi_press)

vvpress=np.array([vx_press,vy_press,vz*corrector])*rescale_v
vvpressbottom=np.array([vx_press,vy_press,vz*corrector])*rescale_v



#Rotate positions for inclination and PA
x01v=rot_x(xv,i0)
x01vbottom=rot_x(xvbottom,i0)
x1v=rot_z(x01v,PA0) 
x1vbottom=rot_z(x01vbottom,PA0) 


x01vcirc=rot_x(xvcirc,i0)
x01vcircbottom=rot_x(xvcircbottom,i0)
x1vcirc=rot_z(x01vcirc,PA0) 
x1vcircbottom=rot_z(x01vcircbottom,PA0) 

x01vcirc0=rot_x(xvcirc0,i0)
x1vcirc0=rot_z(x01vcirc0,PA0)

#Rotate velocities for inclination
#NB you do not need to rotate velocities along z for PA, rotations along z do not change v_z
v1v=rot_x(vv,i0)
v1vbottom=rot_x(vvbottom,i0)

v1vz0=rot_x(vvz0,i0)

#rotate pressure corrected velocities
v1vpress=rot_x(vvpress,i0)
v1vpressbottom=rot_x(vvpressbottom,i0)

#rotate circ vel
v1circv=rot_x(vcircv,i0)
v1circvbottom=rot_x(vcircv,i0)#same as v1circv but for similarity

#vsim
x01vsim=rot_x(xvsim,i0)
x01vsimbottom=rot_x(xvsimbottom,i0)
x1vsim=rot_z(x01vsim,PA0) 
x1vsimbottom=rot_z(x01vsimbottom,PA0) 


v1vsim=rot_x(vvsim,i0)
v1vsimbottom=rot_x(vvsimbottom,i0)

velmax=0.3
velmin=-velmax
lev=np.linspace(-velmax,velmax,19)

extent = aout*rescale_x #fits.open(filename)[0].header['CDELT2'] * 1024 * 3600


matchsign=-1. #to match the sign convention in observations

#Watch the system as if it was originally created face-on (line of sight is z-axis)
#velmax=v1v[2,:].max()*1.3
#velmin=-velmax
plt.figure(1)
plt.scatter(x1v[0,:],x1v[1,:],c=v1v[2,:]*matchsign,cmap="RdBu_r",vmin=velmin*1.1,vmax=velmax*1.1)
#plt.scatter(x1vbottom[0,:],x1vbottom[1,:],c=v1vbottom[2,:],cmap="seismic",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.tricontour(x1v[0,:],x1v[1,:],v1v[2,:]*matchsign,levels=lev, linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])

#Watch the pressure corrected velocity as if it was originally created face-on (line of sight is z-axis)
#velmax=v1v[2,:].max()*1.3
#velmin=-velmax
plt.figure(2)
plt.scatter(x1v[0,:],x1v[1,:],c=v1vpress[2,:]*matchsign,cmap="RdBu_r",vmin=velmin*1.1,vmax=velmax*1.1)
#plt.scatter(x1vbottom[0,:],x1vbottom[1,:],c=v1vpressbottom[2,:],cmap="seismic",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.tricontour(x1v[0,:],x1v[1,:],v1vpress[2,:]*matchsign,levels=lev, linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])

#Watch the system in 3D
fig = plt.figure(3)
ax = fig.add_subplot(projection='3d')
ax.scatter(x1v[0,:],x1v[1,:],x1v[2,:],c=v1v[2,:]*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
ax.scatter(x1vbottom[0,:],x1vbottom[1,:],x1vbottom[2,:],c=v1vbottom[2,:]*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])
ax.set_zlim([zmin,zmax])

#watch the system from the face-on pov
plt.figure(4)
velmax=0.42
velmin=-velmax

plt.scatter(xv[0,:],xv[1,:],c=vv[2,:]*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.scatter(xvbottom[0,:],xvbottom[1,:],c=vvbottom[2,:]*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.axis('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])

#residuals from circ
velmax=0.5
velmin=-velmax
plt.figure(5)
plt.scatter(x1v[0,:],x1v[1,:],c=(v1vsim[2,:]-v1circv[2,:])*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.tricontour(x1v[0,:],x1v[1,:],(v1vsim[2,:]-v1circv[2,:])*matchsign,levels=lev, linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])

#residuals from eccentric
velmax=0.5
velmin=-velmax
plt.figure(6)
plt.scatter(x1v[0,:],x1v[1,:],c=(-v1v[2,:]+v1vsim[2,:])*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.tricontour(x1v[0,:],x1v[1,:],(-v1v[2,:]+v1vsim[2,:])*matchsign,levels=lev, linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])

#residuals from eccentric pressurecorrected
velmax=0.5
velmin=-velmax
plt.figure(7)
plt.scatter(x1v[0,:],x1v[1,:],c=(-0.8*v1vpress[2,:]+v1vsim[2,:])*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.tricontour(x1v[0,:],x1v[1,:],(-0.8*v1vpress[2,:]+v1vsim[2,:])*matchsign,levels=lev, linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])



npix=500
xnew=np.linspace(-extent,extent,npix)
ynew=np.linspace(-extent,extent,npix)

xnew_grid,ynew_grid=np.meshgrid(xnew, ynew)
#plotting RT model
file_path = './MCFOST/RT2A500/i_0_deg/interpolate_RT_0.pkl'

# Open the file in binary read mode and load the function using pickle
with open(file_path, 'rb') as file:
    func_RT = pickle.load(file)

vzpress_interpgrid=itp.interpolator_2D_nonregular_togrid(x1v[0,:],x1v[1,:],v1vpress[2,:]*matchsign,xnew_grid,ynew_grid)
vcirc_interpgrid=itp.interpolator_2D_nonregular_togrid(x1vcirc[0,:],x1vcirc[1,:],v1circv[2,:]*matchsign,xnew_grid,ynew_grid)
vcirc0_interpgrid=itp.interpolator_2D_nonregular_togrid(x1vcirc0[0,:],x1vcirc0[1,:],
                                                        v1circv[2,:]*matchsign,xnew_grid,ynew_grid)
vzpress0_interpgrid=itp.interpolator_2D_nonregular_togrid(x1vcirc0[0,:],x1vcirc0[1,:],
                                                          v1vz0[2,:]*matchsign,xnew_grid,ynew_grid)
vsim_interpgrid=itp.interpolator_2D_nonregular_togrid(x1vsim[0,:],x1vsim[1,:],v1vsim[2,:]*matchsign,xnew_grid,ynew_grid)
#vzpress_interpf=itp.interpolator_2D_spline(x1v[0,:],x1v[1,:],v1vpress[2,:]*matchsign)

residuals=func_RT((xnew_grid,ynew_grid))-vzpress_interpgrid
residuals_circ=func_RT((xnew_grid,ynew_grid))-vcirc_interpgrid
residuals_circ0=func_RT((xnew_grid,ynew_grid))-vcirc0_interpgrid
residuals_press0=func_RT((xnew_grid,ynew_grid))-vzpress0_interpgrid
residuals_sim=func_RT((xnew_grid,ynew_grid))-vsim_interpgrid
residuals_simmodel=vsim_interpgrid-vzpress_interpgrid

velomax=velmax

plt.figure(11)
plt.pcolormesh(xnew,ynew,residuals,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,residuals,levels=lev,linewidths=0.5,colors='k')

plt.figure(22)
plt.pcolormesh(xnew,ynew,vzpress_interpgrid,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,vzpress_interpgrid,levels=lev,linewidths=0.5,colors='k')

plt.figure(33)
plt.pcolormesh(xnew,ynew,func_RT((xnew_grid,ynew_grid)),cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,func_RT((xnew_grid,ynew_grid)),levels=lev,linewidths=0.5,colors='k')

#plt.figure(44)
#plt.pcolormesh(xnew,ynew,residuals_circ,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
#plt.colorbar()
#plt.contour(xnew,ynew,residuals_circ,levels=lev,linewidths=0.5,colors='k')
#
plt.figure(55)
plt.pcolormesh(xnew,ynew,residuals_sim,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,residuals_sim,levels=lev,linewidths=0.5,colors='k')

plt.figure(66)
plt.pcolormesh(xnew,ynew,residuals_circ0,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,residuals_circ0,levels=lev,linewidths=0.5,colors='k')

plt.figure(77)
plt.pcolormesh(xnew,ynew,residuals_press0,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,residuals_press0,levels=lev,linewidths=0.5,colors='k')

plt.figure(88)
plt.pcolormesh(xnew,ynew,residuals_simmodel,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,residuals_simmodel,levels=lev,linewidths=0.5,colors='k')



#nchannels=21
#xchan=[]
#ychan=[]
#vzchan=[]
#vzmin=np.min(v1v[2,:]*matchsign)
#vzmax=np.max(v1v[2,:]*matchsign)
#chanwidth=(vzmax-vzmin)/nchannels
#vbin=vzmin+np.array(range(nchannels))*chanwidth
##create channel maps
#for i in range(0,nchannels-1):
#    which=np.nonzero((v1v[2,:]*matchsign>vbin[i])*(v1v[2,:]*matchsign<vbin[i+1]))
#    xchan.append(x1v[0,which])
#    ychan.append(x1v[1,which])
#    vzchan.append(v1v[2,which]*matchsign)
#
#for i in range(0,nchannels-1,3):
#    plt.figure(10*i)
#    plt.scatter(xchan[i],ychan[i])
#    plt.xlim([xmin,xmax])
#    plt.ylim([ymin,ymax])
#
#plt.draw()
#plt.pause(1)
#input("<Hit enter to close the plots>")
#plt.close('all')
