import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.optimize import newton,bisect
import matplotlib.tri as tri
import discEccanalysis_pysplash as de
import sys
from joblib import Parallel, delayed

def makeadvancementbar(i,Np=1.E6):
    if(i==0):
        sys.stdout.write('[-')
        sys.stdout.flush()
    if(i%int(Np/20.) == 0 ):
        sys.stdout.write('-') 
        sys.stdout.flush()
    if(i==Np-1):
        sys.stdout.write('-]')
        sys.stdout.flush()        
    return

def vxvy2vrvphi(x,y,vx,vy):
    R=np.sqrt(x**2+y**2)
    vr=vx*x/R+vy*y/R 
    vphi=vy*x/R-vx*y/R

    return vr,vphi



def generate_velocity_map(x,y,eccinp,phaseinp,sigmainp,radprofinp,nprocs=10,aout=0.,ain=0.):#,simfield):
    sigma=sigmainp
    ecc=eccinp
    phase=phaseinp
    radprof=radprofinp
    xmesh=x
    ymesh=y
    
    
#    wheremax=np.nonzero(sigma==np.max(sigma)) 
    fracmax=0.6 # at which fraction of max to take cavity size
    wheremax=de.isclosetoArr(sigma,sigma.max()*fracmax,np.diff(sigma).max())[0]
    radIn=radprof[wheremax[0]]
    if ain==0: #if ain not passed take the inner edge of the cavity
        ain=radIn
    if aout==0: #define aout if not passed as argument as fraction of the outer grid radius
        aout=radprof[-1]*0.7
    emax=ecc[wheremax[0]]
    eout=np.mean(ecc[-20:])
    #eccentricity parameters
    e0=0.1
    qecc=0.5
    #vertical properties
    hor=0.0
    hormin=0.1*hor #sets the values between which hor oscillate
    hormax=hor
    parh=1./e0 #rules strength of artificially prescribed h perturbations due to ecc
    parvz=1./e0 #same as above but for vz
    flaring=0*0.25
    #phase parameters 
    varpi0=0.*np.pi/2.
    orbitfrac=0
    ###########
    G=1.
    M=1.
#    import pdb
  #  pdb.set_trace() 
    #grid properties
    Nx=500
    Ny=Nx
    xmin=-aout*(1.+e0)
    xmax=aout*(1.+e0)
    ymin=xmin
    ymax=xmax
    zmin=xmin
    zmax=xmax
    i0=0*np.pi/3
    PA0=0*np.pi/3
    nchannels=20
    
    npol=20
    
    def rot_z(x,theta):
        rotmatr_z=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        return np.matmul(rotmatr_z,x)
    
    def rot_y(x,theta):
        rotmatr_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) 
        return np.matmul(rotmatr_y,x)
    
    def rot_x(x,theta):
        rotmatr_x=np.array([[1, 0, 0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        return np.matmul(rotmatr_x,x)
    
    ee=np.polynomial.Chebyshev.fit(radprof,ecc,npol)
    def e(a):
        return ee(a)*(a<radprof[-1])
    
    varpivarpi=np.polynomial.Chebyshev.fit(radprof,phase,npol*2)
    def varpi(a):
        return varpivarpi(a)*(a<radprof[-1])
    
    cos_interp=np.polynomial.Chebyshev.fit(radprof,np.cos(phase),npol*2)
    def cosvarpi(a):
        return cos_interp(a)*(a<radprof[-1])
    
    sin_interp=np.polynomial.Chebyshev.fit(radprof,np.sin(phase),npol*2)
    def sinvarpi(a):
        return sin_interp(a)*(a<radprof[-1])

    sigma_interp=np.polynomial.Chebyshev.fit(radprof,sigma,3*npol)
    def sigma_a(a):
        return sigma_interp(a)*(a<radprof[-1])

                 
    def R_func(a,theta):
        return a*(1-e(a)**2)/(1.+e(a)*(np.cos(theta)*cosvarpi(a)+np.sin(theta)*sinvarpi(a)))
    
    def hor_func(a,theta):
        return hor*(a[:]/ain)**(1+flaring)
        #(hormin+(hormax-hormin)*(1-parh*0.5*e(a)*np.cos(theta-varpi(a[:]))))
    
    def z_scale(a,theta):
        #return hor*(a[:]/ain)**(1+flaring)
        #return hor_func(a,theta)*(a[:]/ain)**(1+flaring)
        return hor_func(a,theta)*(1.-3.*e(a)*(np.cos(theta)*cosvarpi(a)+np.sin(theta)*sinvarpi(a)))
    
    def vR_func(a,theta):
        return np.sqrt(G*M/a[:])*e(a[:])*(np.sin(theta[:])*cosvarpi(a)-np.cos(theta[:])*sinvarpi(a))/np.sqrt(1-e(a[:])**2)
    
    def vphi_func(a,theta):
        return np.sqrt(G*M/a[:])*(1.+e(a[:])*(np.cos(theta)*cosvarpi(a)+np.sin(theta)*sinvarpi(a)))/np.sqrt(1-e(a[:])**2)
    
    def vz_func(a,theta):
        #return  np.zeros(len(a))
        #return np.sqrt(G*M/a)*e(a)*hor*np.sin(theta-varpi(a[:]))*parvz
        return 3.*e(a)*np.sqrt(G*M/a**3)*(np.sin(theta[:])*cosvarpi(a)-np.cos(theta[:])*sinvarpi(a))*z_scale(a,theta)


    def func_to_root(aa,R,phi,i):
        return R[i]-aa*(1-e(aa)**2)/(1+e(aa)*(np.cos(phi[i])*cosvarpi(aa)+np.sin(phi[i])*sinvarpi(aa)))     
  
    #NB: this function takes as input np.arrays, if you want to pass single values 
    #use e.g. xy2aphi(np.array([1]),np.array([np.pi]).
    def xy2aphi(x,y,selection,Rguess=5.,nproc=nprocs):
        R=np.sqrt(x**2+y**2)
        phi=np.mod(np.arctan2(y,x),2*np.pi)
        asol=np.zeros(len(R))
        print("Calculating semimajor axis for each grid point")

        def root_a_int(i):
            makeadvancementbar(i,Np=len(x))
            if(selection[i]):
                try: 
                    asoli=newton(func_to_root,Rguess,maxiter=50,tol=0.005,args=(R,phi,i)) #R[i] is the initial guess 
                except RuntimeError: 
                    print("Newton failed at R: ",R[i]," at i:",i," trying bisect")   
                    asoli=bisect(func_to_root,R[i]*(1-0.5),R[i]*(1+0.5),maxiter=50,args=(R,phi,i)) #R[i] is the initial guess 
            else:
                asoli=-1

            return asoli
        asol = Parallel(n_jobs=nproc)(delayed(root_a_int)(i) for i in range(len(x)))

        return np.array(asol),phi

    def check_a(x,y,aa):
        R=np.sqrt(x**2+y**2)
        phi=np.mod(np.arctan2(y,x),2*np.pi)    
        return R[i]-aa*(1-e(aa)**2)/(1+e(aa)*(np.cos(phi[i])*cosvarpi(aa)+np.sin(phi[i])*sinvarpi(aa)))

    x0,y0=np.meshgrid(xmesh,ymesh)
    x=x0.reshape(len(xmesh)*len(ymesh))
    y=y0.reshape(len(xmesh)*len(ymesh))
    #carve small region around the origin or newton raphson won't converge
    selectingxy=(np.sqrt(x**2+y**2)>(1-emax)*ain)*(np.sqrt(x**2+y**2)<aout*(1+eout))
    
    #Convert x,y map to a,theta.
    aprov,thetaprov=xy2aphi(x,y,selectingxy)
    
    #Then create the right eccentric cavity with ain and aout
    selectxya=np.nonzero((aprov>ain)*(aprov<aout))[0]
    a=aprov[selectxya]
    theta=thetaprov[selectxya]
    
    #Keep only the chosen x,y (otherwise you get with a square map) and generate z
    x=x[selectxya]
    y=y[selectxya]
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

    return x1v,v1v,selectxya,a,e,varpi,sigma_a


