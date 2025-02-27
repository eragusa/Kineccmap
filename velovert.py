import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import discEccanalysis_pysplash as de
import genvecc as gv
import vertical_structure as vs
import useful_param as up
import interpolation as itp

img=up.img
#folderres='./analysis_paper'
folderres=up.folderres
name=sys.argv[1]

os.system("splash -p nonlog "+name+" -o ascii -r 6 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 7 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 8 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 9 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 16 -dev /png")
#os.system("splash -p nonlog "+name+" -o ascii -r 14 -dev /png")
os.system("splash -p nonlog "+name+" -o ascii -r 17 -dev /png")
#os.system("splash -p nonlog "+name+" -o ascii -r 15 -dev /png")

hor=up.hor
flaring=up.flaring
#load quantities
density=np.loadtxt(name+'_columndensity_proj.pix')
vx=np.loadtxt(name+'_vx_proj.pix')
vy=np.loadtxt(name+'_vy_proj.pix')
z=np.sqrt(np.loadtxt(name+'_z2_proj.pix'))
vz=np.loadtxt(name+'_vz1_proj.pix')/np.sqrt(2./np.pi)#(np.sqrt(2./np.pi))**2 #correction to match velocity at height H.
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

mass=[]
for i in range(res['discfracA'][index[0]].shape[0]):
    mass.append(sum(res['discfracA'][index[0],:i]*res['Mdisc'][index[0]]))

Ma=np.gradient(mass,radii)

sigma=res['sigmaA'][index[0],:]
wheremax=np.nonzero(sigma==np.max(sigma))

#phase=np.ones(len(ecc))*phase[wheremax[0]]
x1v,v1v,selectxya,a,e,cosvarpi,sinvarpi,deda,dvpda,sigma_a,Ma_a,dPda1rhoa_a=gv.generate_velocity_map(x,y,ecc,phase,sigma,Ma,radii,nprocs=20)

vyplan=vy.reshape(nx*ny)[selectxya]
vxplan=vx.reshape(nx*ny)[selectxya]
vzplan=vz.reshape(nx*ny)[selectxya]
zplan=z.reshape(nx*ny)[selectxya]
densityplan=density.reshape(nx*ny)[selectxya]
xgrplan=xgr.reshape(nx*ny)[selectxya]
ygrplan=ygr.reshape(nx*ny)[selectxya]
vr,vphi=gv.vxvy2vrvphi(xgrplan,ygrplan,v1v[0,:],v1v[1,:])
vrsim,vphisim=gv.vxvy2vrvphi(xgrplan,ygrplan,vxplan,vyplan)
R=np.sqrt(xgrplan**2+ygrplan**2)
#calculate H and vz teor
alb=0.0
H,vz,H_func,vz_func=vs.calculate_vertical_structure_bulk(xgrplan,ygrplan,a,e,cosvarpi,sinvarpi,sigma,alphab=alb*5./3.,out_func=True)#0.005*5./3.)
#H,vz=vs.calculate_vertical_structure(xgrplan,ygrplan,a,e,cosvarpi,sinvarpi,sigma)


#phi=np.linspace(0,6.28,len(H[0]))
#for i in range(len(H)):
#    plt.figure(1)
 #   plt.plot(phi,H[i])
 #   plt.figure(2)
 #   plt.plot(phi,vz[i])

x_min=xmin#-11.
x_max=xmax#11
y_min=ymin#-11.
y_max=ymax#11.

plt.figure(1)
zmax=H.max()*0.9
zmin=0
#plt.scatter(xgrplan,ygrplan,c=H,cmap="inferno",vmin=zmin,vmax=zmax)
H_matr=gv.plan2matr(H,nx,ny,selectxya)
plt.pcolormesh(x,y,H_matr,cmap="inferno",vmin=zmin,vmax=zmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.title('$H_{\\rm th}$')
plt.savefig(folderres+'/Hteor.'+img,dpi=400)

plt.figure(12)
zmax=(H/a).max()*0.95
zmin=0
#plt.scatter(xgrplan,ygrplan,c=H/a,cmap="inferno",vmin=zmin,vmax=zmax)
Ha_matr=gv.plan2matr(H/a,nx,ny,selectxya)
#Ha_matr=gv.plan2matr(H/R,nx,ny,selectxya)
plt.pcolormesh(x,y,Ha_matr,cmap="inferno",vmin=zmin,vmax=zmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$H_{\\rm th}/R$')
plt.savefig(folderres+'/HRteor.'+img,dpi=400)

plt.figure(2)
velmax=vz.max()*0.9
velmin=-velmax
#plt.scatter(xgrplan,ygrplan,c=vz,cmap="RdBu_r",vmin=velmin,vmax=velmax)
vz_matr=gv.plan2matr(vz,nx,ny,selectxya)
plt.pcolormesh(x,y,vz_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{z,{\\rm th}}$')
plt.savefig(folderres+'/vzteor.'+img,dpi=400)

plt.figure(3)
zmax=H.max()*0.9
zmin=0
#plt.scatter(xgrplan,ygrplan,c=zplan,cmap="inferno",vmin=zmin,vmax=zmax)
zsim_matr=gv.plan2matr(zplan,nx,ny,selectxya)
plt.pcolormesh(x,y,zsim_matr,cmap="inferno",vmin=zmin,vmax=zmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$H_{\\rm sim}$')
plt.savefig(folderres+'/Hsim.'+img,dpi=400)

plt.figure(32)
zmax=(H/a).max()*0.95
zmin=0
#plt.scatter(xgrplan,ygrplan,c=zplan/a,cmap="inferno",vmin=zmin,vmax=zmax)
zsima_matr=gv.plan2matr(zplan/a,nx,ny,selectxya)
#zsima_matr=gv.plan2matr(zplan/R,nx,ny,selectxya)
plt.pcolormesh(x,y,zsima_matr,cmap="inferno",vmin=zmin,vmax=zmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$H_{\\rm sim}/R$')
plt.savefig(folderres+'/HRsim.'+img,dpi=400)

plt.figure(4)
velmax=vz.max()*0.9
velmin=-velmax
#plt.scatter(xgrplan,ygrplan,c=vzplan,cmap="RdBu_r",vmin=velmin,vmax=velmax)
vzsim_matr=gv.plan2matr(vzplan,nx,ny,selectxya)
plt.pcolormesh(x,y,vzsim_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{z,{\\rm sim}}$')
plt.savefig(folderres+'/vzsim.'+img,dpi=400)

plt.figure(5)
zmax=1.#H.max()*0.9
zmin=0
#plt.scatter(xgrplan,ygrplan,c=(zplan-H)/H,cmap="inferno",vmin=zmin,vmax=zmax)
Dz_H_matr=gv.plan2matr((zplan-H)/vs.Hcirc(a),nx,ny,selectxya)
plt.pcolormesh(x,y,Dz_H_matr,cmap="RdBu_r",vmin=-0.2,vmax=0.2)
plt.colorbar()
#plt.contour(x,y,Dz_H_matr,levels=[-0.01,0,0.01], linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(H_{\\rm sim}-H_{\\rm th})/H_{\\rm circ}$')
plt.savefig(folderres+'/DH_a.'+img,dpi=400)

plt.figure(52)
zmax=1.#H.max()*0.9
zmin=0
#plt.scatter(xgrplan,ygrplan,c=(zplan-H)/H,cmap="inferno",vmin=zmin,vmax=zmax)
Dz_H_matr=gv.plan2matr((zplan-vs.Hcirc(a))/vs.Hcirc(a),nx,ny,selectxya)
plt.pcolormesh(x,y,Dz_H_matr,cmap="RdBu_r",vmin=-0.2,vmax=0.2)
plt.colorbar()
#plt.contour(x,y,Dz_H_matr,levels=[-0.01,0,0.01], linewidths=0.5, colors='k')
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(H_{\\rm sim}-H_{\\rm circ})/H_{\\rm circ}$')
plt.savefig(folderres+'/DH_a_circ.'+img,dpi=400)


plt.figure(6)
velmax=0.2#vz.max()*0.9
velmin=-velmax
#plt.scatter(xgrplan,ygrplan,c=(np.abs(vzplan)-np.abs(vz))/np.abs(vz),cmap="RdBu_r",vmin=velmin,vmax=velmax)
Dvz_vphi_matr=gv.plan2matr((vzplan-vz)/vs.cs(a),nx,ny,selectxya)
plt.pcolormesh(x,y,Dvz_vphi_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(v_{z,{\\rm sim}}-v_{z,{\\rm th}})/c_{\\rm s}$')
plt.savefig(folderres+'/Dvz_cs.'+img,dpi=400)

def sumres(x):
    xx=x.reshape(nx*ny)[selectxya]
    return np.sqrt((xx**2).sum())

sumreslast=sumres(Dvz_vphi_matr)
print('vz/vphi residuals sim-model:',sumreslast)


#In the end it is identical to fig 6.
#mask=Dvz_vphi_matr/Dvz_vphi_matr
#vzpress_interpgrid=itp.interpolator_2D_nonregular_togrid(xgrplan,ygrplan,vz/vs.cs(a),xgr,ygr)
#vsim_interpgrid=itp.interpolator_2D_nonregular_togrid(xgrplan,ygrplan,vzplan/vs.cs(a),xgr,ygr)
#residuals_simmodel=vsim_interpgrid-vzpress_interpgrid
#
#plt.figure(600)
#velmax=0.2#vz.max()*0.9
#velmin=-velmax
##plt.scatter(xgrplan,ygrplan,c=(np.abs(vzplan)-np.abs(vz))/np.abs(vz),cmap="RdBu_r",vmin=velmin,vmax=velmax)
##Dvz_vphi_matr=gv.plan2matr(residuals_simmodel,nx,ny,selectxya)
#plt.pcolormesh(x,y,residuals_simmodel*mask,cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.axis('equal')
#plt.xlim([x_min,x_max])
#plt.ylim([y_min,y_max])
#plt.colorbar()
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.title('$(v_{z,{\\rm sim}}-v_{z,{\\rm th}})/c_{\\rm s}$')
#plt.savefig(folderres+'/Dvz_cs2.'+img,dpi=400)
#

plt.figure(60)
velmax=0.2#vz.max()*0.9
velmin=-velmax
#plt.scatter(xgrplan,ygrplan,c=(np.abs(vzplan)-np.abs(vz))/np.abs(vz),cmap="RdBu_r",vmin=velmin,vmax=velmax)
Dvz_vz_matr=gv.plan2matr((np.abs(vzplan)-np.abs(vz))/vs.cs(a),nx,ny,selectxya)
plt.pcolormesh(x,y,Dvz_vz_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(|v_{z,{\\rm sim}}|-|v_{z,{\\rm th}}|)/c_{\\rm s}$')
plt.savefig(folderres+'/D|vz|_cs.'+img,dpi=400)



plt.figure(62)
velmax=0.01#vz.max()*0.9
velmin=-velmax
Dvz_vphi_matr=gv.plan2matr((vzplan-vz)/np.abs(vphi),nx,ny,selectxya)
plt.pcolormesh(x,y,Dvz_vphi_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
#plt.scatter(xgrplan,ygrplan,c=(np.abs(vzplan)-np.abs(vz))/np.abs(vphi),cmap="RdBu_r")
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(v_{z,{\\rm sim}}-v_{z,{\\rm th}})/v_{\\phi}$')
plt.savefig(folderres+'/Dvz_vphi_highcontrast.'+img,dpi=400)

plt.figure(63)
velmax=0.2#vz.max()*0.9
velmin=-velmax
#plt.scatter(xgrplan,ygrplan,c=(np.abs(vzplan)-np.abs(vz))/np.abs(vz),cmap="RdBu_r",vmin=velmin,vmax=velmax)
Dvz_vphi_matr=gv.plan2matr((vzplan)/vs.cs(a),nx,ny,selectxya)
plt.pcolormesh(x,y,Dvz_vphi_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$(v_{z,{\\rm sim}}-v_{z,{\\rm circ}})/c_{\\rm s}$')
plt.savefig(folderres+'/Dvz_circ_cs.'+img,dpi=400)




plt.figure(7)
velmax=0.035#vz.max()*0.9
velmin=-velmax
#plt.scatter(xgrplan,ygrplan,c=vz/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
vz_vphi_matr=gv.plan2matr((vz)/np.abs(vphi),nx,ny,selectxya)
plt.pcolormesh(x,y,vz_vphi_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{z,{\\rm th}}/v_{\\phi}$')
plt.savefig(folderres+'/vzteor_vphi.'+img,dpi=400)

plt.figure(72)
#plt.scatter(xgrplan,ygrplan,c=(vzplan)/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
vzsim_vz_matr=gv.plan2matr((vzplan)/np.abs(vphi),nx,ny,selectxya)
plt.pcolormesh(x,y,vzsim_vz_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{z,{\\rm sim}}/v_{\\phi}$')
plt.savefig(folderres+'/vzsim_vphi.'+img,dpi=400)

plt.figure(722)
velmax=0.8#vz.max()*0.9
velmin=-velmax
#plt.scatter(xgrplan,ygrplan,c=vz/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
vz_vphi_matr=gv.plan2matr((vz)/vs.cs(a),nx,ny,selectxya)
plt.pcolormesh(x,y,vz_vphi_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{z,{\\rm th}}/c_{\\rm s}$')
plt.savefig(folderres+'/vzteor_cs.'+img,dpi=400)

plt.figure(723)
#plt.scatter(xgrplan,ygrplan,c=(vzplan)/np.abs(vphi),cmap="RdBu_r",vmin=velmin,vmax=velmax)
vzsim_vz_matr=gv.plan2matr((vzplan)/vs.cs(a),nx,ny,selectxya)
plt.pcolormesh(x,y,vzsim_vz_matr,cmap="RdBu_r",vmin=velmin,vmax=velmax)
plt.axis('equal')
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$v_{z,{\\rm sim}}/c_{\\rm s}$')
plt.savefig(folderres+'/vzsim_cs.'+img,dpi=400)



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

plt.figure(83)
plt.plot(radii,Ma,label='simulation')
plt.plot(radii,Ma_a(radii),label='model')
plt.xlabel('$a$')
plt.ylabel('$M_a$')
plt.xlim([2,13])
plt.legend()
plt.ylim([-0.0001,Ma.max()+0.0001])
ax=plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(folderres+'/Ma.'+img,dpi=400)



#plt.draw()
#plt.pause(1)
#input("<Hit enter to close the plots>")
#plt.close('all')
