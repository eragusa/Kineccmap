import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.optimize import newton
from numba import jit,prange

Na=30
Ntheta=300
ain=4.
aout=50
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
i0=np.pi/3
PA0=0*np.pi/3


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


#NB this function takes as input np.arrays
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
    
    

a=np.linspace(ain,aout,Na)
theta=np.linspace(0,np.pi*2,Ntheta)

Rlist=[]
vRlist=[]
vphilist=[]
vzlist=[]

for i in range(len(a)):
    Rlist.append(R_func(a[i],theta))
    #from Murray & Dermott Eq. 2.31 and 2.32, do not use 2.36 it is fundamentally wrong!
    vRlist.append(-np.sqrt(G*M/a[i])*e(a[i])*np.sin(theta-varpi(a[i]))/np.sqrt(1-e(a[i])**2))
    vphilist.append(np.sqrt(G*M/a[i])*(1.+e(a[i])*np.cos(theta-varpi(a[i])))/np.sqrt(1-e(a[i])**2))
    vzlist.append(np.zeros(len(theta)))

R=np.array(Rlist)
vR=np.array(vRlist)
vphi=np.array(vphilist)
vz=np.array(vzlist)


x=R*np.cos(theta[np.newaxis,:])
y=R*np.sin(theta[np.newaxis,:])
z=hor*(R[:,:]/ain)**(1+flaring)

vy=vphi*np.cos(theta[np.newaxis,:])+vR*np.sin(theta[np.newaxis,:])
vx=-vphi*np.sin(theta[np.newaxis,:])+vR*np.cos(theta[np.newaxis,:])

xv=np.array([x,y,z]).reshape(3,len(a)*len(theta))
xvbottom=np.array([x,y,-z]).reshape(3,len(a)*len(theta))
vv=np.array([vx,vy,vz]).reshape(3,len(a)*len(theta))


#rotate positions for inclination and PA
x0v=rot_x(xv,i0)
x0vbottom=rot_x(xvbottom,i0)
x1v=rot_z(x0v,PA0) 
x1vbottom=rot_z(x0vbottom,PA0) 

#NB you do not need to rotate velocities along z for PA, rotations along z do not change v_z
v1v=rot_x(vv,i0)

#for i in range(0,len(x1v[0,:]),1):
plt.figure(1)
plt.scatter(x1v[0,:],x1v[1,:],c=v1v[2,:])
plt.colorbar()
plt.axis('equal')

fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.scatter(x1v[0,:],x1v[1,:],x1v[2,:],c=v1v[2,:])
ax.scatter(x1vbottom[0,:],x1vbottom[1,:],x1vbottom[2,:],c=v1v[2,:])

plt.figure(3)
plt.scatter(x1v[0,:],x1v[2,:],c=v1v[2,:])
plt.scatter(x1vbottom[0,:],x1vbottom[2,:],c=v1v[2,:])
plt.colorbar()
plt.axis('equal')
