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
from matplotlib.patches import Ellipse

img='0.png'#up.img
#folderres='./analysis_paper'
folderres=up.folderres+'/mcfost_RT/0deg/'
name=sys.argv[1]
i0=0./180*np.pi#up.i0
PA0=up.PA0
aout=up.aout

#Rescale for mock image
binarysemimaj=15.
distance=130
lorig=1. #current binary size

#corrector of vertical velocity and scale height
corrector=1.#3.5#corrector for matching the last emission surface in units of H
corrector2=corrector



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
vzsim=np.loadtxt(name+'_vz1_proj.pix')/(np.sqrt(2./np.pi)) #correction to match velocity at height H.
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

#####################################################
##### Calculating velocities and other profiles #####
#####################################################
x0v,v0v,selectxya,a,e,cosvarpi,sinvarpi,deda,dvpda,sigma_a,Ma_a,dPda1rhoa_a,a_full,ain,aout=\
                                                    gv.generate_velocity_map(x,y,ecc,phase,sigma,\
                                                                              Ma,radii,nprocs=20,aout=10.,ret_full_a=True)


#defining value of varpi and e for all selected pixels
varpi=np.arctan2(sinvarpi(a),cosvarpi(a))
eccentricity=e(a)

#define grid coordinates
Rgr=np.sqrt(xgr**2+ygr**2)
thetagr=np.arctan2(ygr,xgr)

#taking coordinates of the selection and making the plane vectors, we will revert the selection later
xgrplan=xgr.reshape(nx*ny)[selectxya]
ygrplan=ygr.reshape(nx*ny)[selectxya]
Rplan=Rgr.reshape(nx*ny)[selectxya]
thetaplan=thetagr.reshape(nx*ny)[selectxya]

#reshaping and selecting z from simulation
zplan=z.reshape(nx*ny)[selectxya]

############################################
#####Calculating the vertical structure#####
############################################
H,vz=vs.calculate_vertical_structure(xgrplan,ygrplan,a,e,cosvarpi,sinvarpi,sigma)

######### Adding other model velocities #############

#vr vphi of the model
vr,vphi=gv.vxvy2vrvphi(xgrplan,ygrplan,v0v[0,:],v0v[1,:])

#Calculating circular velocities for the model
vxcircplan,vycircplan,vmod=gv.vcircular(Rplan,thetaplan)

#pressure corrected velocities
vphi_press=gv.pressure_corrected_vphi(a,vphi,dPda1rhoa_a)
vx_press,vy_press=gv.vrvphi2vxvy(xgrplan,ygrplan,vr,vphi_press)

#transforming simulation coordinates to match the selected pixels, we will revert this later
vyplan=vy.reshape(nx*ny)[selectxya]
vxplan=vx.reshape(nx*ny)[selectxya]
vzplan=vzsim.reshape(nx*ny)[selectxya]
vrsim,vphisim=gv.vxvy2vrvphi(xgrplan,ygrplan,vxplan,vyplan)

#####################################################################################
#####   We revert the selection including also excluded region for velocities   #####
#####   this is achieved attributing v=0 in the cavity, v=nan outside.          #####
#####   mainly for plotting properly the cavity area when interpolating         #####
#####################################################################################

#reverting the selection to the full disc model
x0=xgr.reshape(nx*ny)[a_full<aout] #*1. to make a copy of xgrplan
y0=ygr.reshape(nx*ny)[a_full<aout] 
z0=gv.include_excluded_z(H,xgr,ygr,a_full,ain,aout)

#creating mask before reverting selection
#x_mask=np.array([x0,y0,z0])*rescale_x
mask=gv.create_mask(z0,xgr,ygr,a_full,ain,aout)

#sim
xgrplan=x0
ygrplan=y0
zplan=gv.include_excluded_z(zplan,xgr,ygr,a_full,ain,aout)
Rplan=Rgr.reshape(nx*ny)[a_full<aout] 
thetaplan=thetagr.reshape(nx*ny)[a_full<aout] 


#velocities model
vr=gv.include_excluded_velocity(vr,xgr,ygr,a_full,ain,aout)
vphi=gv.include_excluded_velocity(vphi,xgr,ygr,a_full,ain,aout)
vx0=gv.include_excluded_velocity(v0v[0,:],xgr,ygr,a_full,ain,aout)
vy0=gv.include_excluded_velocity(v0v[1,:],xgr,ygr,a_full,ain,aout)
vz=gv.include_excluded_velocity(vz,xgr,ygr,a_full,ain,aout)

v0v=np.array([vx0,vy0,vz])

#vcirc
vxcircplan=gv.include_excluded_velocity(vxcircplan,xgr,ygr,a_full,ain,aout)
vycircplan=gv.include_excluded_velocity(vycircplan,xgr,ygr,a_full,ain,aout)

#v_press
vx_press=gv.include_excluded_velocity(vx_press,xgr,ygr,a_full,ain,aout)
vy_press=gv.include_excluded_velocity(vy_press,xgr,ygr,a_full,ain,aout)

#vsim
vrsim=gv.include_excluded_velocity(vrsim,xgr,ygr,a_full,ain,aout)
vphisim=gv.include_excluded_velocity(vphisim,xgr,ygr,a_full,ain,aout)
vxplan=gv.include_excluded_velocity(vxplan,xgr,ygr,a_full,ain,aout)
vyplan=gv.include_excluded_velocity(vyplan,xgr,ygr,a_full,ain,aout)
vzplan=gv.include_excluded_velocity(vzplan,xgr,ygr,a_full,ain,aout)


#generate real height from radtransf of the faceon case
file_path = './MCFOST/RT2A500/i_0_deg/interpolate_H_0.pkl'

# Open the file in binary read mode and load the function using pickle
with open(file_path, 'rb') as file:
    func_H = pickle.load(file)

H_RT=func_H((x0*rescale_x,y0*rescale_x))#the interpolate is already with rescaled x,y


#Create arrays for applying rotations and mirror the disc also on the negativ z-axis
#xv=np.array([x0,y0,z0*corrector2])*rescale_x
#xvbottom=np.array([x0,y0,-z0*corrector2])*rescale_x
xv=np.array([x0,y0,H_RT*corrector2])*rescale_x
xvbottom=np.array([x0,y0,-H_RT*corrector2])*rescale_x
#vv=np.array([v0v[0,:],v0v[1,:],vz*corrector])*rescale_v
#vvbottom=np.array([v0v[0,:],v0v[1,:],-vz*corrector])*rescale_v

#accounting for vertical profile extracted from RT
#vv=np.array([v0v[0,:],v0v[1,:],np.nan_to_num(vz*corrector*H_RT/z0,posinf=0.,neginf=0.)])*rescale_v
#vvbottom=np.array([v0v[0,:],v0v[1,:],np.nan_to_num(-vz*corrector*H_RT/z0,posinf=0.,neginf=0.)])*rescale_v

#accounting for vertical profile extracted from RT + z**2+R**2 correction
rescale_v2=np.sqrt(Rplan)*(Rplan/(Rplan**2+H_RT**2)**(3./4.))
vv=np.array([v0v[0,:]*rescale_v2,v0v[1,:]*rescale_v2,np.nan_to_num(vz*corrector*H_RT/z0,posinf=0.,neginf=0.)])*rescale_v
vvbottom=np.array([v0v[0,:]*rescale_v2,v0v[1,:]*rescale_v2,np.nan_to_num(-vz*corrector*H_RT/z0,posinf=0.,neginf=0.)])*rescale_v



#eccentric models with no vertical height and velocity
vvz0=np.array([v0v[0,:],v0v[1,:],0.*v0v[0,:]])*rescale_v


#xcirc
correctorcirc=3.
zcirc=vs.Hcirc(np.sqrt(x0**2+y0**2))
xvcirc=np.array([x0,y0,zcirc*correctorcirc])*rescale_x
xvcircbottom=np.array([x0,y0,-zcirc*correctorcirc])*rescale_x
xvcirc0=np.array([x0,y0,0.*x0])*rescale_x #no vertical displacement

#vcirc
vcircv=np.array([vxcircplan,vycircplan,0.*vxcircplan])*rescale_v

#xvsim=np.array([xgrplan,ygrplan,zplan*corrector2])*rescale_x
#xvsimbottom=np.array([xgrplan,ygrplan,-zplan*corrector2])*rescale_x
#vvsim=np.array([vxplan,vyplan,vzplan*corrector])*rescale_v
#vvsimbottom=np.array([vxplan,vyplan,-vzplan*corrector])*rescale_v
xvsim=np.array([xgrplan,ygrplan,zplan*corrector2])*rescale_x
xvsimbottom=np.array([xgrplan,ygrplan,-zplan*corrector2])*rescale_x
#the 0.8 to account for factor sqrt(2/pi)
vvsim=np.array([vxplan,vyplan,np.nan_to_num(vzplan*corrector*H_RT/(zplan),posinf=0.,neginf=0.)])*rescale_v
vvsimbottom=np.array([vxplan,vyplan,np.nan_to_num(-vzplan*corrector*H_RT/(zplan),posinf=0.,neginf=0.)])*rescale_v

#the 0.8 to account for factor sqrt(2/pi)
vvpress=np.array([vx_press,vy_press,np.nan_to_num(vz*corrector*H_RT/(z0),posinf=0.,neginf=0.)])*rescale_v
vvpressbottom=np.array([vx_press,vy_press,np.nan_to_num(-vz*corrector*H_RT/(z0),posinf=0.,neginf=0.)])*rescale_v

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

#defining plot limits and rescaling
xmin=xmin*rescale_x
ymin=ymin*rescale_x
zmin=zmin*rescale_x

xmax=xmax*rescale_x
ymax=ymax*rescale_x
zmax=zmax*rescale_x

extent = aout*rescale_x #fits.open(filename)[0].header['CDELT2'] * 1024 * 3600
velmax=0.3
velmin=-velmax
lev=[-0.2,-0.1,0.,0.1,0.2]#np.linspace(-velmax,velmax,19)





matchsign=-1. #to match the sign convention in observations

#Watch the system as if it was originally created face-on (line of sight is z-axis)
#velmax=v1v[2,:].max()*1.3
#velmin=-velmax
plt.figure(1)
plt.scatter(x1v[0,:],x1v[1,:],c=v1v[2,:]*matchsign,cmap="RdBu_r",vmin=velmin*1.1,vmax=velmax*1.1)
#plt.scatter(x1vbottom[0,:],x1vbottom[1,:],c=v1vbottom[2,:],cmap="seismic",vmin=velmin,vmax=velmax)
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
#plt.scatter(xvbottom[0,:],xvbottom[1,:],c=vvbottom[2,:]*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
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
plt.scatter(x1v[0,:],x1v[1,:],c=(-v1vpress[2,:]+v1vsim[2,:])*matchsign,cmap="RdBu_r",vmin=velmin,vmax=velmax)
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
plt.tricontour(x1v[0,:],x1v[1,:],(-v1vpress[2,:]+v1vsim[2,:])*matchsign,levels=lev, linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([-extent,extent])
plt.ylim([-extent,extent])
plt.savefig(folderres+'res_scatter_RT_vpress_0.'+img)



npix=nx
xnew=np.linspace(-extent,extent,npix)
ynew=np.linspace(-extent,extent,npix)

xnew_grid,ynew_grid=np.meshgrid(xnew, ynew)
#plotting RT model
file_path = './MCFOST/RT2A500/i_0_deg/interpolate_RT_0.pkl'

# Open the file in binary read mode and load the function using pickle
with open(file_path, 'rb') as file:
    func_RT = pickle.load(file)


mask_interpgrid=itp.interpolator_2D_nonregular_togrid(x1v[0,:],x1v[1,:],mask,xnew_grid,ynew_grid)
H_RT_interpgrid=itp.interpolator_2D_nonregular_togrid(x1v[0,:],x1v[1,:],H_RT/zplan,xnew_grid,ynew_grid)
#H_RT_interpgrid=itp.interpolator_2D_nonregular_togrid(x1v[0,:],x1v[1,:],H_RT/vs.Hcirc(a_full[np.nonzero(a_full<aout)[0]]),xnew_grid,ynew_grid)
#H_RT_interpgrid=itp.interpolator_2D_nonregular_togrid(x1v[0,:],x1v[1,:],H_RT/z0,xnew_grid,ynew_grid)

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

#clean properly the cavity from spurious contours
residuals[(residuals>-0.0001)*(residuals<0.0001)]=0.
residuals_simmodel[(residuals_simmodel>-0.0001)*(residuals_simmodel<0.0001)]=0.
vzpress_interpgrid[(vzpress_interpgrid>-0.0001)*(vzpress_interpgrid<0.0001)]=0.

velomax=velmax

plt.figure(11)
plt.pcolormesh(xnew,ynew,residuals,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
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
plt.contour(xnew,ynew,residuals,levels=lev,linewidths=0.5,colors='k')
plt.savefig(folderres+'res_RT_vpress.'+img)

plt.figure(22)
plt.pcolormesh(xnew,ynew,vzpress_interpgrid,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
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
plt.contour(xnew,ynew,vzpress_interpgrid,levels=lev,linewidths=0.5,colors='k')
plt.savefig(folderres+'vpress_0.'+img)

plt.figure(33)
which=np.nonzero(((1.-mask_interpgrid)<0.0001)*((1.-mask_interpgrid)>-0.0001))
mask2=(1.-mask_interpgrid)
mask2[which]=0.
plt.pcolormesh(xnew,ynew,func_RT((xnew_grid,ynew_grid))*mask2,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
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
plt.contour(xnew,ynew,func_RT((xnew_grid,ynew_grid))*mask2,levels=lev,linewidths=0.5,colors='k')
dx,dy=(0.08, 0.08)
beam = Ellipse(
    ax.transLimits.inverted().transform((dx, dy)),
    width=0.05,
    height=0.05,
    angle=0.,
    fill=True,
    color="grey")

ax.add_patch(beam)
plt.savefig(folderres+'RT_0.'+img)


#plt.figure(44)
#plt.pcolormesh(xnew,ynew,residuals_circ,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
#plt.colorbar()
#plt.contour(xnew,ynew,residuals_circ,levels=lev,linewidths=0.5,colors='k')
#
plt.figure(55)
plt.pcolormesh(xnew,ynew,residuals_sim,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
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
plt.contour(xnew,ynew,residuals_sim,levels=lev,linewidths=0.5,colors='k')

plt.figure(66)
plt.pcolormesh(xnew,ynew,residuals_circ0,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
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
plt.contour(xnew,ynew,residuals_circ0,levels=lev,linewidths=0.5,colors='k')
plt.savefig(folderres+'res_circ.'+img)

plt.figure(77)
plt.pcolormesh(xnew,ynew,residuals_press0,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
plt.colorbar()
plt.contour(xnew,ynew,residuals_press0,levels=lev,linewidths=0.5,colors='k')

plt.figure(88)
plt.pcolormesh(xnew,ynew,residuals_simmodel,cmap='RdBu_r',vmin=-velomax,vmax=velomax)
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
plt.contour(xnew,ynew,residuals_simmodel,levels=lev,linewidths=0.5,colors='k')
plt.savefig(folderres+'res_sim_vpress.'+img)

plt.figure(99)
#do the same as above but place nans where 0 as viridis map does not have white zeros.
which=np.nonzero(((1.-mask_interpgrid)<0.0001)*((1.-mask_interpgrid)>-0.0001))
mask2=(1.-mask_interpgrid)
mask2[which]=np.nan
plt.pcolormesh(xnew,ynew,H_RT_interpgrid*mask2,cmap='viridis',vmin=0.,vmax=4)
plt.xlabel('$\\Delta \\alpha$ [\'\']')
plt.ylabel('$\\Delta \\delta$ [\'\']')
ax=plt.gca()
ax.xaxis.label.set_size(17)
ax.yaxis.label.set_size(17)
ax.tick_params(labelsize = 17)
ax.set_aspect('equal')
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 17)
cb.set_label("$H_{\\rm RT}/H_{\\rm sim}$", size = 17)
plt.title('$H_{\\rm RT}(\\tau=1)/H_{\\rm sim}$')
#plt.contour(xnew,ynew,residuals_simmodel,levels=lev,linewidths=0.5,colors='k')
plt.savefig(folderres+'H_RT.'+img)



def sumres(x):
    xx=x.reshape(nx*ny)[selectxya]
    return np.sqrt((xx**2).sum())

sumreslast=sumres(residuals)
print('Residuals RT-model:',sumreslast)
sumresmodel=sumres(residuals_simmodel)
print('Residualt sim-model:',sumresmodel)



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
