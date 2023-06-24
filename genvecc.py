import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.optimize import newton,bisect
import matplotlib.tri as tri
import discEccanalysis_pysplash as de
import sys
from joblib import Parallel, delayed
from vertical_structure import Omega,cs,Hcirc
import multiprocessing
import os
import useful_param as up
import pdb
import interpolation as itp

#Note that interpolated functions _a are defined to not return 0 for any "a" larger than (1-frgrid) of the radprof.
#This is done to avoid weird behaviour of interpolated polynomials close to the boundary and beyond
frgrid=0.05

#eccentricity parameters
e0=up.e0
qecc=up.qecc
#vertical properties
#hor=0.0
#hormin=0.1*hor #sets the values between which hor oscillate
#hormax=hor
#parh=1./e0 #rules strength of artificially prescribed h perturbations due to ecc
#parvz=1./e0 #same as above but for vz
#flaring=0*0.25
#phase parameters
#varpi0=0.*np.pi/2.
#orbitfrac=0
###########
G=up.G
M=up.M

#orienting your disc in space
#i0=0*np.pi/3
#PA0=0*np.pi/3
#for channel maps
#nchannels=20

#Npol for interpolation
npol=20

def Omega0(a):
    return np.sqrt(G*M/a**3)

def Omega_orb(a,phi,e,varpi):
    return Omega0(a)*(1.+e*np.cos(phi-varpi))**2/(1-e**2)**1.5

def vcircular(R,theta):
    return np.sqrt(G*M/R)*np.sin(theta),np.sqrt(G*M/R)*np.cos(theta),np.sqrt(G*M/R)

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

def vrvphi2vxvy(x,y,vr,vphi):
    R=np.sqrt(x**2+y**2)
    sintheta=y/R
    costheta=x/R
    vx=vr*costheta-vphi*sintheta
    vy=vr*sintheta+vphi*costheta
    return vx,vy

def pressure_corrected_vphi(a,vphi,dPda1rhoa):
    return np.sqrt(vphi**2+dPda1rhoa(a))

def killLoky():
    #returns the default signal of kill, which is 15
    out=os.system('ps -ef|grep LokyProcess|awk -v n="`ps -ef|grep LokyProcess|wc -l`" \'NR<n-3{print $2}\'|xargs kill')
    if out==15:
        success="BANGARANG"
    else:
        success='Not successful'
    print("")
    print("Killing stray LokyProcesses from parallelisation: ",success)
    return

def generate_velocity_map(x,y,eccinp,phaseinp,sigmainp,Mainp,radprofinp,nprocs=10,aout=0.,ain=0.):#,simfield):
    sigma=sigmainp
    Ma=Mainp
    ecc=eccinp
    phase=phaseinp
    radprof=radprofinp
    xmesh=x
    ymesh=y


    if ain==0: #if ain not passed 
        ain=up.ain #take value from paramfile
        index,radxxx=de.matchtime(radprof,np.array([ain])) #here used to match the value for ain
        fracmax=up.fracmax # at which fraction of max to take cavity size
        #we calculate the max beyond ain provided in paramfile, to avoid bad values in the cavity
        wheremax=de.isclosetoArr(Ma,Ma[index[0]:].max()*fracmax,np.diff(Ma).max())[0]
        radIn=radprof[wheremax[0]] 
        ain=radIn #take the inner edge of the cavity
        #ain=up.ain #take value from paramfile
    if aout==0: #define aout if not passed as argument as fraction of the outer grid radius
        aout=radprof[-1]*0.7

    emax=ecc[wheremax[0]]
    eout=np.mean(ecc[-20:])
    #grid properties
    xmin=-aout*(1.+e0)
    xmax=aout*(1.+e0)
    ymin=xmin
    ymax=xmax
    zmin=xmin
    zmax=xmax

#    def rot_z(x,theta):
#        rotmatr_z=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
#        return np.matmul(rotmatr_z,x)

#    def rot_y(x,theta):
#        rotmatr_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
#        return np.matmul(rotmatr_y,x)

#    def rot_x(x,theta):
#        rotmatr_x=np.array([[1, 0, 0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
#        return np.matmul(rotmatr_x,x)

    #Note that we fit with chebyshev also to smooth out the functions
    #Simple interpolation would be too rough to be used in the equation solver
    #for passing from xy-->a,phi
    ee=itp.interpolator(radprof,ecc,2*npol)
    def e(a):
        return ee(a)*(a<radprof[-int(frgrid*len(radprof))])

    varpivarpi=itp.interpolator(radprof,phase,npol*2)
    def varpi(a):
        return varpivarpi(a)*(a<radprof[-int(frgrid*len(radprof))])

    cos_interp=itp.interpolator(radprof,np.cos(phase),npol*2)
    def cosvarpi(a):
        return cos_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

    sin_interp=itp.interpolator(radprof,np.sin(phase),npol*2)
    def sinvarpi(a):
        return sin_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

    sigma_interp=itp.interpolator(radprof,sigma,6*npol)
    def sigma_a(a):
        return sigma_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

    Ma_interp=itp.interpolator(radprof,Ma,6*npol)
    def Ma_a(a):
        return Ma_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

    deda=np.gradient(e(radprof),radprof)
    deeda=itp.interpolator(radprof,deda,6*npol)
    def deda_f(a):
        return deeda(a)*(a<radprof[-int(frgrid*len(radprof))])

    vp=np.arctan2(sinvarpi(radprof),cosvarpi(radprof))
    #to avoid weird peaks take derivative of complex phase
    #then divide by same and take imaginary part
    dvarpida=np.imag(np.gradient(np.exp(1.j*vp),radprof)/np.exp(1.j*vp))
    dvpvpda=itp.interpolator(radprof,dvarpida,int(npol/2))
    def dvpda_f(a):
        return dvpvpda(a)*(a<radprof[-int(frgrid*len(radprof))])


    rho=sigma#/Hcirc(radprof)
    cs2rho=cs(radprof)**2*rho
    dPda1rhoa=np.nan_to_num(\
                     np.divide(\
                                np.gradient(cs(radprof)**2*rho,radprof),\
                                rho,\
                                out=np.zeros_like(rho),where=rho>0.)\
                                *radprof,\
                            posinf=0,neginf=0)
    dPda1rhoa_interp=itp.interpolator(radprof,dPda1rhoa,4*npol)
    def dPda1rhoa_a(a):
        return dPda1rhoa_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

    def R_func(a,theta):
        return a*(1-e(a)**2)/(1.+e(a)*(np.cos(theta)*cosvarpi(a)+np.sin(theta)*sinvarpi(a)))

#    def hor_func(a,theta):
#        return hor*(a[:]/ain)**(1+flaring)
#        #(hormin+(hormax-hormin)*(1-parh*0.5*e(a)*np.cos(theta-varpi(a[:]))))

    def z_scale(a,theta):
        return  np.zeros(len(a))
#        #return hor*(a[:]/ain)**(1+flaring)
#        #return hor_func(a,theta)*(a[:]/ain)**(1+flaring)
#        return hor_func(a,theta)*(1.-3.*e(a)*(np.cos(theta)*cosvarpi(a)+np.sin(theta)*sinvarpi(a)))

    def vR_func(a,theta):
        return np.sqrt(G*M/a[:])*e(a[:])*(np.sin(theta[:])*cosvarpi(a)-np.cos(theta[:])*sinvarpi(a))/np.sqrt(1-e(a[:])**2)

    def vphi_func(a,theta):
        return np.sqrt(G*M/a[:])*(1.+e(a[:])*(np.cos(theta)*cosvarpi(a)+np.sin(theta)*sinvarpi(a)))/np.sqrt(1-e(a[:])**2)

    def vz_func(a,theta):
        return  np.zeros(len(a))
#        #return np.sqrt(G*M/a)*e(a)*hor*np.sin(theta-varpi(a[:]))*parvz
#        #return 3.*e(a)*np.sqrt(G*M/a**3)*(np.sin(theta[:])*cosvarpi(a)-np.cos(theta[:])*sinvarpi(a))*z_scale(a,theta)


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
                    asoli=newton(func_to_root,Rguess,maxiter=50,tol=0.005*ain,args=(R,phi,i)) #R[i] is the initial guess
                except RuntimeError:
                    #print("Newton failed at R: ",R[i]," at i:",i," trying bisect")
                    try:
                        print("Newton failed Runtime at R: ",R[i]," at i:",i," trying bisect")
                        asoli=bisect(func_to_root,R[i]*(1-0.5),R[i]*(1+0.5),maxiter=50,args=(R,phi,i)) #R[i] initial guess
                    except ValueError:
                        print("bisect failed ValueError at R: ",R[i]," at i:",i," attributing asoli=-1")
                        asoli=-1
                except ValueError:
                    try:
                        print("Newton failed ValueError at R: ",R[i]," at i:",i," trying bisect")
                        asoli=bisect(func_to_root,R[i]*(1-0.5),R[i]*(1+0.5),maxiter=50,args=(R,phi,i)) #R[i] initial guess
                    except ValueError:
                        print("bisect failed ValueError at R: ",R[i]," at i:",i," attributing asoli=-1")
                        asoli=-1

            else:
                asoli=-1
            return asoli
#        pdb.set_trace()
        with Parallel(n_jobs=multiprocessing.cpu_count()) as parallel:
            asol = parallel(delayed(root_a_int)(i) for i in range(len(x)))

        #kill remaining loky processes from joblib parallelisation
        killLoky()
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
#    xvbottom=np.array([x,y,-z])
    vv=np.array([vx,vy,vz])
#    vvbottom=np.array([vx,vy,-vz])


    #Rotate positions for inclination and PA
#    x0v=rot_x(xv,i0)
#    x0vbottom=rot_x(xvbottom,i0)
    x1v=xv#rot_z(x0v,PA0)
#    x1vbottom=rot_z(x0vbottom,PA0)

    #Rotate velocities for inclination
    #NB you do not need to rotate velocities along z for PA, rotations along z do not change v_z
    v1v=vv#rot_x(vv,i0)
#    v1vbottom=rot_x(vvbottom,i0)

    return x1v,v1v,selectxya,a,e,cosvarpi,sinvarpi,deda_f,dvpda_f,sigma_a,Ma_a,dPda1rhoa_a
