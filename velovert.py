import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import discEccanalysis_pysplash as de
import genvecc as gv
import vertical_structure as vs

img='png'
folderres='./analysis_paper'
name=sys.argv[1]

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
sigma=res['sigmaA'][index[0],:]
wheremax=np.nonzero(sigma==np.max(sigma))

#phase=np.ones(len(ecc))*phase[wheremax[0]]
x1v,v1v,selectxya,a,e,cosvarpi,sinvarpi,sigma_a,dPda1rhoa_a=gv.generate_velocity_map(x,y,ecc,phase,sigma,radii,nprocs=20)

vyplan=vy.reshape(nx*ny)[selectxya]
vxplan=vx.reshape(nx*ny)[selectxya]
vzplan=vz.reshape(nx*ny)[selectxya]
zplan=z.reshape(nx*ny)[selectxya]
densityplan=density.reshape(nx*ny)[selectxya]
xgrplan=xgr.reshape(nx*ny)[selectxya]
ygrplan=ygr.reshape(nx*ny)[selectxya]
vr,vphi=gv.vxvy2vrvphi(xgrplan,ygrplan,v1v[0,:],v1v[1,:])
vrsim,vphisim=gv.vxvy2vrvphi(xgrplan,ygrplan,vxplan,vyplan)
#calculate H and vz teor
H,vz=vs.calculate_vertical_structure(xgrplan,ygrplan,a,e,cosvarpi,sinvarpi,sigma)


#phi=np.linspace(0,6.28,len(H[0]))
#for i in range(len(H)):
#    plt.figure(1)
 #   plt.plot(phi,H[i])
 #   plt.figure(2)
 #   plt.plot(phi,vz[i])

plt.figure(1)
zmax=H.max()*0.9
zmin=0
plt.scatter(xgrplan,ygrplan,c=H,cmap="inferno",vmin=zmin,vmax=zmax)
plt.colorbar()
plt.title('$H_{\\rm teor}$')
plt.savefig(folderres+'/Hteor.'+img,dpi=400)

plt.figure(12)
zmax=(H/a).max()*0.9
zmin=0
plt.scatter(xgrplan,ygrplan,c=H/a,cmap="inferno",vmin=zmin,vmax=zmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$H_{\\rm teor}/a$')
plt.savefig(folderres+'/HRteor.'+img,dpi=400)

plt.figure(2)
velmax=vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=vz,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{z,{\\rm teor}}$')
plt.savefig(folderres+'/vzteor.'+img,dpi=400)

plt.figure(3)
zmax=H.max()*0.9
zmin=0
plt.scatter(xgrplan,ygrplan,c=zplan,cmap="inferno",vmin=zmin,vmax=zmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$H_{\\rm sim}$')
plt.savefig(folderres+'/Hsim.'+img,dpi=400)

plt.figure(32)
zmax=(H/a).max()*0.9
zmin=0
plt.scatter(xgrplan,ygrplan,c=zplan/a,cmap="inferno",vmin=zmin,vmax=zmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$H_{\\rm sim}/a$')
plt.savefig(folderres+'/HRsim.'+img,dpi=400)

plt.figure(4)
velmax=vz.max()*0.5
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=vzplan,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$|v_{z,{\\rm sim}}|$')
plt.savefig(folderres+'/vzsim.'+img,dpi=400)

plt.figure(5)
zmax=1.#H.max()*0.9
zmin=0
plt.scatter(xgrplan,ygrplan,c=(zplan-H)/H,cmap="inferno",vmin=zmin,vmax=zmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(H_{\\rm sim}-H_{\\rm teor})/H_{\\rm teor}$')
plt.savefig(folderres+'/DH_H.'+img,dpi=400)

plt.figure(6)
velmax=0.1#vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(np.abs(vzplan)-np.abs(vz))/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(|v_{z,{\\rm sim}}|-|v_{z,{\\rm teor}}|)/v_{\\phi}$')
plt.savefig(folderres+'/Dvz_vphi.'+img,dpi=400)

plt.figure(62)
velmax=0.1#vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(np.abs(vzplan)-np.abs(vz))/np.abs(vphi),cmap="RdBu_r")
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(|v_{z,{\\rm sim}}|-|v_{z,{\\rm teor}}|)/v_{\\phi}$')
plt.savefig(folderres+'/Dvz_vphi_highcontrast.'+img,dpi=400)


plt.figure(7)
velmax=0.05#vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=vz/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$|v_{z,{\\rm teor}}|/v_{\\phi}$')
plt.savefig(folderres+'/vzteor_vphi.'+img,dpi=400)

plt.figure(72)
velmax=0.05#vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(vzplan)/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$|v_{z,{\\rm sim}}|/v_{\\phi}$')
plt.savefig(folderres+'/vzsim_vphi.'+img,dpi=400)

## A few more plots
plt.figure(8)
plt.plot(radii,ecc,label='simulation')
plt.plot(radii,e(radii),label='model')
plt.xlabel('$a$')
plt.ylabel('$e$')
plt.xlim([2,13])
plt.legend()
plt.ylim([0,0.35])
ax=plt.gca()
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(folderres+'/ea.'+img,dpi=400)

plt.figure(81)
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

plt.figure(82)
plt.plot(radii,sigma,label='simulation')
plt.plot(radii,sigma_a(radii),label='model')
plt.xlabel('$a$')
plt.ylabel('$\\Sigma$')
plt.xlim([2,13])
plt.legend()
plt.ylim([-0.00001,sigma.max()+0.00001])
ax=plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(folderres+'/sigmaa.'+img,dpi=400)


#plt.draw()
#plt.pause(1)
#input("<Hit enter to close the plots>")
#plt.close('all')
