import numpy as np
import matplotlib.pyplot as plt
import pdb
import discEccanalysis_pysplash as de
import interpolation as itp
import useful_param as up
import pickle
import genvecc as gv

npol=up.npol

def Jacobian_det(a,phi,e_f,sinvarpi_f,cosvarpi_f,deda_f,dvpda_f):

    vp=np.arctan2(sinvarpi_f(a),cosvarpi_f(a))
    cosvpi=cosvarpi_f(a)
    sinvpi=sinvarpi_f(a)
    e=e_f(a)
    deda=deda_f(a)
    dvpda=dvpda_f(a)
    cosphi_vpi=np.cos(phi)*cosvpi+np.sin(phi)*sinvpi
    sinphi_vpi=np.sin(phi)*cosvpi-np.cos(phi)*sinvpi

    R=a*(1-e**2)/(1.+e*np.cos(phi-vp))
    qcosa=((1.+e**2)*a*deda-(1-e**2)*e)/(1-e*(e+2*a*deda))
    qsina=a*e*(1-e**2)*dvpda/(1-e*(e+2*a*deda))
#    qcosxxx=qcosa*np.cos(phi-vp)+qsina*np.sin(phi-vp)
    qcosxxx=qcosa*cosphi_vpi+qsina*sinphi_vpi

#    J=R**3*(1.-e*(e+2*a*deda))/(a*(1-e**2))**2*(1-qcosxxx)
    J=R*(1.-e*(e+2*a*deda))/(1.+e*np.cos(phi-vp))**2*(1-qcosxxx)
    alpha=np.arctan2(qsina,qcosa)
    q=np.sqrt(qcosa**2+qsina**2)
    return J,alpha,q


def Jacobian_det_interp(e_f,sinvarpi_f,cosvarpi_f,deda_f,dvpda_f,ain=2.5,aout=10.):

    a=np.linspace(ain,aout,150)
    phi=np.linspace(0,np.pi*2.,200)
    agr,phigr=np.meshgrid(a,phi)
    vp=np.arctan2(sinvarpi_f(agr),cosvarpi_f(agr))
    cosvpi=cosvarpi_f(agr)
    sinvpi=sinvarpi_f(agr)
    e=e_f(agr)
    deda=deda_f(agr)
    dvpda=dvpda_f(agr)
    cosphi_vpi=np.cos(phigr)*cosvpi+np.sin(phigr)*sinvpi
    sinphi_vpi=np.sin(phigr)*cosvpi-np.cos(phigr)*sinvpi

    R=agr*(1-e**2)/(1.+e*np.cos(phigr-vp))
    qcosa=((1.+e**2)*agr*deda-(1-e**2)*e)/(1-e*(e+2*agr*deda))
    qsina=agr*e*(1-e**2)*dvpda/(1-e*(e+2*agr*deda))
    qcosxxx=qcosa*cosphi_vpi+qsina*sinphi_vpi

    J=R*(1.-e*(e+2*agr*deda))/(1.+e*np.cos(phigr-vp))**2*(1-qcosxxx)
    Delta=1./J*np.gradient(gv.Omega_orb(agr,phigr,e,vp)*J,phi,axis=0)
    alpha=np.arctan2(qsina,qcosa)
    q=np.sqrt(qcosa**2+qsina**2)
    name=up.name    
    J_f=itp.interpolator_2D(a,phi,J)
    file_path = name+'_Jacobian_interp.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(J_f, file)
    
    Delta_f=itp.interpolator_2D(a,phi,Delta)
    file_path = name+'_Delta_interp.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(Delta_f, file)



    return J,alpha,q,J_f,Delta_f



if __name__=='__main__':
    name=up.name
    frgrid=0.05
    aref=20.#3.
    index=[0]#[100]
    res=de.loadHDF5('datasim.h5')
    ecc=np.abs(res['evecA'][index[0],:])
    varpi=np.angle(res['evecA'][index[0],:])
    radprof=res['radProf'][:]
    a=aref*np.ones(100)
    phi=np.linspace(0,np.pi*2.,100)

    ee=itp.interpolator(radprof,ecc,npol)
    def e(a):
        return ee(a)*(a<radprof[-int(frgrid*len(radprof))])

    sin_interp=itp.interpolator(radprof,np.sin(varpi),npol*2)
    def sinvarpi(a):
        return sin_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

    cos_interp=itp.interpolator(radprof,np.cos(varpi),npol*2)
    def cosvarpi(a):
        return cos_interp(a)*(a<radprof[-int(frgrid*len(radprof))])


    deda=np.gradient(e(radprof),radprof)
    vp=np.arctan2(sinvarpi(radprof),cosvarpi(radprof))
    #to avoid weird peaks take derivative of complex phase
    #then divide by same and take imaginary part
    dvarpida=np.imag(np.gradient(np.exp(1.j*vp),radprof)/np.exp(1.j*vp))

    deeda=itp.interpolator(radprof,deda,6*npol)
    def deda_f(a):
        return deeda(a)*(a<radprof[-int(frgrid*len(radprof))])

    dvpvpda=itp.interpolator(radprof,dvarpida,2*npol)
    def dvpda(a):
        return dvpvpda(a)*(a<radprof[-int(frgrid*len(radprof))])


    J,alpha,q=Jacobian_det(a,phi,e,sinvarpi,cosvarpi,deda_f,dvpda)
#    J_f=itp.interpolator_2D(a,phi,J)
    #J2,alpha2,q2,J_f,Delta_f=Jacobian_det_interp(e,sinvarpi,cosvarpi,deda_f,dvpda)
    J2,alpha2,q2,J_f,Delta_f=Jacobian_det_interp(e,sinvarpi,cosvarpi,deda_f,dvpda,ain=radprof[0],aout=radprof[-1])

    plt.figure(1)
    plt.plot(radprof,dvarpida)
    plt.plot(radprof,np.arctan2(sinvarpi(radprof),cosvarpi(radprof)))
    plt.plot(radprof,varpi)
    plt.plot(radprof,dvpda(radprof))

    plt.figure(2)
    plt.plot(radprof,deda)
    plt.plot(radprof,ecc)
    plt.plot(radprof,e(radprof))

    plt.figure(3)
    plt.imshow(J2)
    
    plt.figure(4)
    plt.plot(phi,J,label='J_det')
    plt.plot(phi,J_f((np.ones_like(phi)*3,phi)),label='interp')
    plt.legend()

    plt.figure(5)
    plt.plot(phi,Delta_f((np.ones_like(phi)*3,phi)),label='interp')
    plt.legend()
