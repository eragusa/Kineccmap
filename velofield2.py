import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import discEccanalysis_pysplash as de
import genvecc as gv
import morphology as morph
#import vertical_structure as vs

img='png'
#folderres='./analysis_paper/azimuth'
#folderres='./figures_for_poster/azimuth'
folderres='./sim5A2000/azimuth'

name=sys.argv[1]

os.system("splash -p nonlog "+name+" -o ascii -r 6 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 7 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 8 -dev /png")

#load quantities
density=np.loadtxt(name+'_columndensity_proj.pix')
vx=np.loadtxt(name+'_vx_proj.pix')
vy=np.loadtxt(name+'_vy_proj.pix')

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
#sigma=res['sigmaA'][index[0],:]
#sigma is not actually the real sigma, but M_a which we calculate as follows
mass=[]
for i in range(res['discfracA'][index[0]].shape[0]):
    mass.append(sum(res['discfracA'][index[0],:i]*res['Mdisc'][index[0]]))

Ma=np.gradient(mass,radii)
sigma=res['sigmaA'][index[0],:]

wheremax=np.nonzero(sigma==np.max(sigma))

#phase=np.ones(len(ecc))*phase[wheremax[0]]
x1v,v1v,selectxya,a,e,cosvarpi,sinvarpi,deda,dvpda,sigma_a,Ma_a,dPda1rhoa_a=gv.generate_velocity_map(x,y,ecc,phase,sigma,Ma,radii,nprocs=20)

Rgr=np.sqrt(xgr**2+ygr**2)
thetagr=np.arctan2(ygr,xgr)
#defining value of varpi and e for all cells
varpi=np.arctan2(sinvarpi(a),cosvarpi(a))
eccentricity=e(a)

vxcirc,vycirc,vmod=gv.vcircular(Rgr,thetagr)
vyplan=vy.reshape(nx*ny)[selectxya]
vxplan=vx.reshape(nx*ny)[selectxya]
vycircplan=vycirc.reshape(nx*ny)[selectxya]
vmodplan=vmod.reshape(nx*ny)[selectxya]
densityplan=density.reshape(nx*ny)[selectxya]

xgrplan=xgr.reshape(nx*ny)[selectxya]
ygrplan=ygr.reshape(nx*ny)[selectxya]
thetaplan=thetagr.reshape(nx*ny)[selectxya]

vr,vphi=gv.vxvy2vrvphi(xgrplan,ygrplan,v1v[0,:],v1v[1,:])
vphi_press=gv.pressure_corrected_vphi(a,vphi,dPda1rhoa_a)
vx_press,vy_press=gv.vrvphi2vxvy(xgrplan,ygrplan,vr,vphi_press)

vrsim,vphisim=gv.vxvy2vrvphi(xgrplan,ygrplan,vxplan,vyplan)
#H,vz=calculate_vertical_structure(xgrplan,ygrplan,a,e,cosvarpi,sinvarpi,sigma)

#Morphology
J,alpha,q=morph.Jacobian_det(a,thetaplan,e,sinvarpi,cosvarpi,deda,dvpda)
SigmaEcc=Ma_a(a)*gv.Omega0(a)/(2*np.pi*J*gv.Omega_orb(a,thetaplan,eccentricity,varpi))

#whattoplot=selectxya.reshape()
velmax=v1v[1,:].max()*0.1
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
plt.savefig(folderres+'/Dvy.'+img,dpi=400)

velmax=v1v[1,:].max()*0.1
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
plt.savefig(folderres+'/Dvy_circ.'+img,dpi=400)

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
plt.savefig(folderres+'/Dvy_vphi.'+img,dpi=400)

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
plt.savefig(folderres+'/Dvy_circ_vphi.'+img,dpi=400)

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
plt.savefig(folderres+'/Dvy_press_vphi.'+img,dpi=400)

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
plt.savefig(folderres+'/Dvphi_vphi.'+img,dpi=400)

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
plt.savefig(folderres+'/Dvphi_press_vphi.'+img,dpi=400)

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
plt.savefig(folderres+'/Dvphi_circ_vphi.'+img,dpi=400)



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
plt.savefig(folderres+'/vyteor.'+img,dpi=400)

plt.figure(4)
plt.scatter(xgrplan,ygrplan,c=vyplan,cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.pcolormesh(x,y,vy,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
#plt.contour(x,y,vy,levels=lev, linewidths=0.5, colors='k')
plt.tricontour(xgrplan,ygrplan,vyplan,levels=20, linewidths=0.5, colors='k')
plt.title('$v_{y,sim}$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.savefig(folderres+'/vysim.'+img,dpi=400)

plt.figure(51)
vmax=np.max(Ma/6.28)
vmin=np.min(Ma/6.28)
plt.scatter(xgrplan,ygrplan,c=SigmaEcc,vmin=vmin,vmax=vmax,cmap="inferno")
plt.colorbar()
plt.title('$\Sigma_{\\rm teor}(a,\phi)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.savefig(folderres+'/Sigma_teor.'+img,dpi=400)

plt.figure(52)
plt.scatter(xgrplan,ygrplan,c=densityplan,vmin=vmin,vmax=vmax,cmap="inferno")
plt.colorbar()
plt.title('$\Sigma_{\\rm sim}(a,\phi)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.savefig(folderres+'/Sigma_sim.'+img,dpi=400)

plt.figure(53)
vmax=np.max(q)
vmin=0
plt.scatter(xgrplan,ygrplan,c=q,vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title('$q_{\\rm teor}(a,\phi)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.savefig(folderres+'/q_teor.'+img,dpi=400)

plt.figure(54)
vmax=3.14
vmin=-3.14
plt.scatter(xgrplan,ygrplan,c=alpha+varpi,vmin=vmin,vmax=vmax,cmap='hsv')
plt.colorbar()
plt.title('$\\alpha_{\\rm teor}(a,\phi)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.savefig(folderres+'/alpha_teor.'+img,dpi=400)

plt.figure(61)
plt.plot(radii,deda(radii),label='model')
plt.xlabel('$a$')
plt.ylabel('$deda$')
plt.xlim([2,13])
plt.ylim([-0.5,0.5])
plt.legend()
ax=plt.gca()
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(folderres+'/deda.'+img,dpi=400)

plt.figure(62)
plt.plot(radii,dvpda(radii),label='model')
plt.xlabel('$a$')
plt.ylabel('$dvpda$')
plt.xlim([2,13])
plt.ylim([-1.5,1.5])
plt.legend()
ax=plt.gca()
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(folderres+'/dvpda.'+img,dpi=400)

plt.figure(63)
plt.plot(radii,phase,label='simulation')
plt.plot(radii,np.arctan2(sinvarpi(radii),cosvarpi(radii)),label='model')
plt.xlabel('$a$')
plt.ylabel('$\\varpi$')
plt.xlim([2,13])
plt.legend()
plt.ylim([-3.20,3.20])
ax=plt.gca()
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(folderres+'/phasea.'+img,dpi=400)

plt.figure(64)
plt.plot(radii,ecc,label='simulation')
plt.plot(radii,e(radii),label='model')
plt.xlabel('$a$')
plt.ylabel('$e$')
plt.xlim([2,13])
plt.legend()
plt.ylim([0,0.5])
ax=plt.gca()
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(folderres+'/ea.'+img,dpi=400)

#plt.draw()
#plt.pause(1)
#input("<Hit enter to close the plots>")
#plt.close('all')
