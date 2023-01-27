import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.optimize import newton
from numba import jit,prange
import matplotlib.tri as tri

#size of ellipses
ain=1.
aout=10
#vertical properties
hor=0.1
flaring=0.5
#eccentricity parameters
e0=0.3
qecc=0.5
#phase parameters 
varpi0=0
orbitfrac=0.
###########
G=1.
M=1.

#grid properties
Nx=200
Ny=Nx
xmin=-aout*(1.+e0)
xmax=aout*(1.+e0)
ymin=xmin
ymax=xmax
zmin=xmin
zmax=xmax
i0=np.pi/8
PA0=np.pi/3


def rot_z(x,theta):
    rotmatr_z=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    return np.matmul(rotmatr_z,x)

def rot_y(x,theta):
    rotmatr_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) 
    return np.matmul(rotmatr_y,x)

def rot_x(x,theta):
    rotmatr_x=np.array([[1, 0, 0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return np.matmul(rotmatr_x,x)

def e(a):
    ecc0=e0
    a0=ain
    return e0*(a/a0)**(-qecc)

def varpi(a):
    p=varpi0+2*np.pi*orbitfrac*(a-ain)/(aout-ain)
    return p

def R_func(a,theta):
    return a*(1-e(a)**2)/(1.+e(a)*np.cos(theta-varpi(a)))

def z_scale(a,theta):
    return hor*(a[:]/ain)**(1+flaring)

def vR_func(a,theta):
    return -np.sqrt(G*M/a[:])*e(a[:])*np.sin(theta[:]-varpi(a[:]))/np.sqrt(1-e(a[:])**2)

def vphi_func(a,theta):
    return np.sqrt(G*M/a[:])*(1.+e(a[:])*np.cos(theta[:]-varpi(a[:])))/np.sqrt(1-e(a[:])**2)

def vz_func(a,theta):
    return  np.zeros(len(a))

#NB: this function takes as input np.arrays, if you want to pass single values 
#use e.g. xy2aphi(np.array([1]),np.array([np.pi]).
#@jit(nopython=False,parallel=True)
def xy2aphi(x,y):
    R=np.sqrt(x**2+y**2)
    phi=np.mod(np.arctan2(y,x),2*np.pi)
    asol=np.zeros(len(R))

    for i in range(len(x)):
        def func_to_root(aa):
            return R[i]-aa*(1-e(aa)**2)/(1+e(aa)*np.cos(phi[i]-varpi(aa)))
        asol[i]=newton(func_to_root,R[i]) #R[i] is the initial guess

    return asol,phi
    
    

xmesh=np.linspace(xmin,xmax,Nx)
ymesh=np.linspace(ymin,ymax,Ny)


x0,y0=np.meshgrid(xmesh,ymesh)
xprov=x0.reshape(len(xmesh)*len(ymesh))
yprov=y0.reshape(len(xmesh)*len(ymesh))
#carve small region around the origin or newton raphson won't converge
x=xprov[np.nonzero(np.sqrt(xprov**2+yprov**2)>0.2*ain)[0]]
y=yprov[np.nonzero(np.sqrt(xprov**2+yprov**2)>0.2*ain)[0]]

#Convert x,y map to a,theta.
aprov,thetaprov=xy2aphi(x,y)

#Then create the right eccentric cavity with ain and aout
a=aprov[np.nonzero((aprov>ain)*(aprov<aout))[0]]
theta=thetaprov[np.nonzero((aprov>ain)*(aprov<aout))[0]]

#Keep only the chosen x,y (otherwise you get with a square map) and generate z
x=x[np.nonzero((aprov>ain)*(aprov<aout))[0]]
y=y[np.nonzero((aprov>ain)*(aprov<aout))[0]]
z=z_scale(a,theta)

#Calculate relevant velocity field
vR=vR_func(a[:],theta[:])
vphi=vphi_func(a[:],theta[:])

#Convert back to cartesian velocities vx,vy (do not use directly Eq. 2.36 in Murray & Dermott) and vz
vy=vphi*np.cos(theta[:])+vR*np.sin(theta[:])
vx=-vphi*np.sin(theta[:])+vR*np.cos(theta[:])
vz=vz_func(a[:],theta[:])

#Create arrays for applying rotations and mirror the disc also on the negativ z-axis
xv=np.array([x,y,z])
xvbottom=np.array([x,y,-z])
vv=np.array([vx,vy,vz])
vvbottom=np.array([vx,vy,-vz])


#Rotate positions for inclination and PA
x0v=rot_x(xv,i0)
x0vbottom=rot_x(xvbottom,i0)
x1v=rot_z(x0v,PA0) 
x1vbottom=rot_z(x0vbottom,PA0) 

#Rotate velocities for inclination
#NB you do not need to rotate velocities along z for PA, rotations along z do not change v_z
v1v=rot_x(vv,i0)
v1vbottom=rot_x(vvbottom,i0)


#Watch the system as if it was originally created face-on (line of sight is z-axis)
plt.figure(1)
plt.scatter(x1v[0,:],x1v[1,:],c=v1v[2,:],cmap="RdBu_r")
plt.colorbar()
plt.tricontour(x1v[0,:],x1v[1,:],v1v[2,:],levels=14, linewidths=0.5, colors='k')
plt.axis('equal')

#Watch the system in 3D
fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.scatter(x1v[0,:],x1v[1,:],x1v[2,:],c=v1v[2,:],cmap="RdBu_r")
ax.scatter(x1vbottom[0,:],x1vbottom[1,:],x1vbottom[2,:],c=v1vbottom[2,:],cmap="RdBu_r")
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])
ax.set_zlim([zmin,zmax])

#Watch the system as it it was created edge-on (line of sight is y-axis)
plt.figure(3)
plt.scatter(x1v[0,:],x1v[2,:],c=v1v[1,:],cmap="RdBu_r")
plt.scatter(x1vbottom[0,:],x1vbottom[2,:],c=v1vbottom[1,:],cmap="RdBu_r")
plt.colorbar()
plt.axis('equal')
