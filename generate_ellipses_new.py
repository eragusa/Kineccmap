import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import discEccanalysis_pysplash as de
import genvecc as gv
#import vertical_structure as vs

#size of ellipses
ain=2.
aout=10.
#eccentricity parameters
e0=0.27
qecc=0.5
#vertical properties
hor=0.1
hormin=0.1*hor #sets the values between which hor oscillate
hormax=hor
parh=1./e0 #rules strength of artificially prescribed h perturbations due to ecc
parvz=1./e0#same as above but for vz
flaring=0.25
#phase parameters 
varpi0=0.*np.pi/2.
orbitfrac=0.
###########
G=1.
M=1.
#######
xmin=-aout*(1.+e0)
xmax=aout*(1.+e0)
ymin=xmin
ymax=xmax
zmin=xmin
zmax=xmax
 
nx=200
ny=nx

x=np.linspace(xmin,xmax,nx)
y=np.linspace(ymin,ymax,ny)
 
xgr,ygr=np.meshgrid(x,y)

radii=np.linspace(ain,aout,200)

ecc=np.abs(res['evecA'][index[0],:])
phase=np.angle(res['evecA'][index[0],:])
mass=[]
for i in range(res['discfracA'][index[0]].shape[0]):
    mass.append(sum(res['discfracA'][index[0],:i]*res['Mdisc'][index[0]]))

Ma=np.gradient(mass,radii)
sigma=res['sigmaA'][index[0],:]
wheremax=np.nonzero(sigma==np.max(sigma))

#phase=np.ones(len(ecc))*phase[wheremax[0]]
x1v,v1v,selectxya,a,e,cosvarpi,sinvarpi,deda,dvpda,Ma_a,sigma_a,dPda1rhoa_a=gv.generate_velocity_map(x,y,ecc,phase,sigma,Ma,radii,nprocs=20)

Rgr=np.sqrt(xgr**2+ygr**2)
thetagr=np.arctan2(ygr,xgr)

vxcirc,vycirc,vmod=gv.vcircular(Rgr,thetagr)
vyplan=vy.reshape(nx*ny)[selectxya]
vxplan=vx.reshape(nx*ny)[selectxya]
vycircplan=vycirc.reshape(nx*ny)[selectxya]
vmodplan=vmod.reshape(nx*ny)[selectxya]
densityplan=density.reshape(nx*ny)[selectxya]

xgrplan=xgr.reshape(nx*ny)[selectxya]
ygrplan=ygr.reshape(nx*ny)[selectxya]

vr,vphi=gv.vxvy2vrvphi(xgrplan,ygrplan,v1v[0,:],v1v[1,:])
vphi_press=gv.pressure_corrected_vphi(a,vphi,dPda1rhoa_a)
vx_press,vy_press=gv.vrvphi2vxvy(xgrplan,ygrplan,vr,vphi_press)

vrsim,vphisim=gv.vxvy2vrvphi(xgrplan,ygrplan,vxplan,vyplan)
#H,vz=calculate_vertical_structure(xgrplan,ygrplan,a,e,cosvarpi,sinvarpi,sigma)

#whattoplot=selectxya.reshape()
velmax=v1v[1,:].max()*0.9
velmin=-velmax
plt.figure(1)
plt.scatter(xgrplan,ygrplan,c=(vyplan-v1v[1,:]),cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.pcolormesh(x,y,vy,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
#plt.contour(x,y,vy,levels=lev, linewidths=0.5, colors='k')
plt.tricontour(xgrplan,ygrplan,vyplan,levels=20, linewidths=0.5, colors='k')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{y,{\\rm sim}}-v_y$')
plt.axis('equal')
plt.savefig('./figures_for_poster/azimuth/Dvy.'+img,dpi=400)

velmax=v1v[1,:].max()*0.9
velmin=-velmax
plt.figure(11)
plt.scatter(xgrplan,ygrplan,c=(vyplan-vycircplan),cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.pcolormesh(x,y,vy,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
#plt.contour(x,y,vy,levels=lev, linewidths=0.5, colors='k')
plt.tricontour(xgrplan,ygrplan,vyplan,levels=20, linewidths=0.5, colors='k')
plt.title('$v_{y,{\\rm sim}}-v_{y,{\\rm circ}}$')
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('./figures_for_poster/azimuth/Dvy_circ.'+img,dpi=400)

plt.figure(2)
#norm=velmax/0.9
velmax=0.2#(vyplan-v1v[1,:]).max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(vyplan-v1v[1,:])/vphi,cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.pcolormesh(x,y,vy,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
#plt.contour(x,y,vy,levels=lev, linewidths=0.5, colors='k')
#plt.tricontour(xgrplan,ygrplan,(vyplan-v1v[1,:])/vphi,levels=20, linewidths=0.5, colors='k')
plt.title('$(v_{y,{\\rm sim}}-v_y)/v_{\\phi}$')
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('./figures_for_poster/azimuth/Dvy_vphi.'+img,dpi=400)

plt.figure(22)
#norm=velmax/0.9
velmax=0.2#(vyplan-v1v[1,:]).max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(vyplan-vycircplan)/vphi,cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.pcolormesh(x,y,vy,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
#plt.contour(x,y,vy,levels=lev, linewidths=0.5, colors='k')
#plt.tricontour(xgrplan,ygrplan,(vyplan-v1v[1,:])/vphi,levels=20, linewidths=0.5, colors='k')
plt.title('$(v_{y,{\\rm sim}}-v_{y,{\\rm circ}})/v_{\\phi}$')
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('./figures_for_poster/azimuth/Dvy_circ_vphi.'+img,dpi=400)

plt.figure(23)
#norm=velmax/0.9
velmax=0.2#(vyplan-v1v[1,:]).max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(vyplan-vy_press)/vphi,cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.pcolormesh(x,y,vy,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
#plt.contour(x,y,vy,levels=lev, linewidths=0.5, colors='k')
#plt.tricontour(xgrplan,ygrplan,(vyplan-vy_press)/vphi,levels=20, linewidths=0.5, colors='k')
plt.title('$(v_{y,{\\rm sim}}-v_{y,{\\rm press}})/v_{\\phi}$')
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('./figures_for_poster/azimuth/Dvy_press_vphi.'+img,dpi=400)

vphimax=1#vphi.max()
velmax=0.2#(vphi-vphisim).max()*0.9
velmin=-velmax
plt.figure(3)
plt.scatter(xgrplan,ygrplan,c=(vphisim-vphi)/vphi,cmap="RdBu_r",vmin=velmin/vphimax,vmax=velmax/vphimax)
plt.colorbar()
#plt.tricontour(xgrplan,ygrplan,(vphisim-vphi)/vphi,levels=20, linewidths=0.5, colors='k')
plt.title('$(v_{\\phi,{\\rm sim}}-v_\\phi)/v_{\\phi}$')
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('./figures_for_poster/azimuth/Dvphi_vphi.'+img,dpi=400)

vphimax=1#vphi.max()
velmax=0.2#(vphi-vphisim).max()*0.9
velmin=-velmax
plt.figure(32)
plt.scatter(xgrplan,ygrplan,c=(vphisim-vphi_press)/vphi,cmap="RdBu_r",vmin=velmin/vphimax,vmax=velmax/vphimax)
plt.colorbar()
#plt.tricontour(xgrplan,ygrplan,(vphisim-vphi_press)/vphi,levels=20, linewidths=0.5, colors='k')
plt.title('$(v_{\\phi,{\\rm sim}}-v_{\\phi,{\\rm press}})/v_{\\phi}$')
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('./figures_for_poster/azimuth/Dvphi_press_vphi.'+img,dpi=400)

vphimax=1#vphi.max()
velmax=0.2#(vphi-vphisim).max()*0.9
velmin=-velmax
plt.figure(33)
plt.scatter(xgrplan,ygrplan,c=(vphisim-vmodplan)/vphi,cmap="RdBu_r",vmin=velmin/vphimax,vmax=velmax/vphimax)
plt.colorbar()
#plt.tricontour(xgrplan,ygrplan,(vphisim-vphi_press)/vphi,levels=20, linewidths=0.5, colors='k')
plt.title('$(v_{\\phi,{\\rm sim}}-v_{\\rm circ})/v_{\\phi}$')
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('./figures_for_poster/azimuth/Dvphi_circ_vphi.'+img,dpi=400)



#vrmax=1.#vr.max()
#velmax=0.2#(vr-vrsim).max()*0.9
#velmin=-velmax
#plt.figure(4)
#plt.scatter(xgrplan,ygrplan,c=(vrsim-vr)/vphi,cmap="RdBu_r",vmin=velmin/vrmax,vmax=velmax/vrmax)
#plt.colorbar()
##plt.tricontour(xgrplan,ygrplan,(vr-vrsim)/vphi,levels=20, linewidths=0.5, colors='k')
#plt.title('$(v_{r,sim}-v_r)/v_{\\phi}$')
#plt.axis('equal')
#
#dmax=((vrsim-vr)*densityplan).max()  
#plt.figure(5)
#plt.scatter(xgrplan,ygrplan,c=densityplan*(vrsim-vr),cmap="RdBu_r") 
#plt.colorbar()
#plt.title('$\\Sigma(v_{r,sim}-v_r)$')
#plt.axis('equal')
#
#dmax=((vphisim-vphi)*densityplan).max()  
#plt.figure(6) 
#plt.scatter(xgrplan,ygrplan,c=densityplan*(vphisim-vphi),cmap="RdBu_r",vmin=-dmax,vmax=dmax) 
#plt.colorbar() 
#plt.title('$\\Sigma(v_{\\phi,sim}-v_\\phi)$')
#plt.axis('equal') 
#  
#
#plt.figure(7) 
#plt.scatter(xgrplan,ygrplan,c=densityplan,cmap="RdBu_r") 
#plt.colorbar() 
#plt.title('$\\Sigma$')
#plt.axis('equal') 
#
velmax=v1v[1,:].max()*0.9
velmin=-velmax
nlevel=20
lev=np.linspace(velmin,velmax,nlevel)
plt.figure(8)
plt.scatter(x1v[0,:],x1v[1,:],c=v1v[1,:],cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.tricontour(x1v[0,:],x1v[1,:],v1v[1,:],levels=lev, linewidths=0.5, colors='k')
plt.title('$v_{y,teor}$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.savefig('./figures_for_poster/azimuth/vyteor.'+img,dpi=400)

plt.figure(9)
plt.scatter(xgrplan,ygrplan,c=vyplan,cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.pcolormesh(x,y,vy,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
#plt.contour(x,y,vy,levels=lev, linewidths=0.5, colors='k')
plt.tricontour(xgrplan,ygrplan,vyplan,levels=20, linewidths=0.5, colors='k')
plt.title('$v_{y,sim}$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.savefig('./figures_for_poster/azimuth/vysim.'+img,dpi=400)


#plt.draw()
#plt.pause(1) 
#input("<Hit enter to close the plots>")
#plt.close('all')
