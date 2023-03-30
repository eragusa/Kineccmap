import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import discEccanalysis_pysplash as de
import genvecc as gv
import vertical_structure as vs

name=sys.argv[1]
 
os.system("splash -p nonlog "+name+" -o ascii -r 6 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 7 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 8 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 14 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 15 -dev /png")

#load quantities
density=np.loadtxt(name+'_columndensity_proj.pix')
vx=np.loadtxt(name+'_vx_proj.pix')
vy=np.loadtxt(name+'_vy_proj.pix')
z=np.sqrt(np.loadtxt(name+'_z2_proj.pix'))
vz=np.sqrt(np.loadtxt(name+'_vz2_proj.pix'))

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
x1v,v1v,selectxya,a,e,cosvarpi,sinvarpi,sigma_a=gv.generate_velocity_map(x,y,ecc,phase,sigma,radii,nprocs=20)

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

plt.figure(2)
velmax=vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=vz,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()

plt.figure(3)
zmax=H.max()*0.9
zmin=0
plt.scatter(xgrplan,ygrplan,c=zplan,cmap="inferno",vmin=zmin,vmax=zmax)
plt.colorbar()

plt.figure(4)
velmax=vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=vzplan,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()

plt.figure(5)
zmax=1.#H.max()*0.9
zmin=0
plt.scatter(xgrplan,ygrplan,c=(H-zplan)/H,cmap="inferno",vmin=zmin,vmax=zmax)
plt.colorbar()

plt.figure(6)
velmax=0.1#vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(np.abs(vz)-vzplan)/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()

plt.figure(7)
velmax=0.1#vz.max()*0.9
velmin=-velmax
plt.scatter(xgrplan,ygrplan,c=(np.abs(vz))/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.colorbar()

plt.draw()
plt.pause(1) 
input("<Hit enter to close the plots>")
plt.close('all')
