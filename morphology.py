import numpy as np
import matplotlib.pyplot as plt
import pdb
import discEccanalysis_pysplash as de
import interpolation as itp
import useful_param as up

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


if __name__=='__main__':
    frgrid=0.05
    index=[100]
    res=de.loadHDF5('datasim.h5')
    ecc=np.abs(res['evecA'][index[0],:])
    varpi=np.angle(res['evecA'][index[0],:])
    radprof=res['radProf'][:]
    a=3.*np.ones(100)
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


    plt.figure(1)
    plt.plot(radprof,dvarpida)
    plt.plot(radprof,np.arctan2(sinvarpi(radprof),cosvarpi(radprof)))
    plt.plot(radprof,varpi)
    plt.plot(radprof,dvpda(radprof))

    plt.figure(2)
    plt.plot(radprof,deda)
    plt.plot(radprof,ecc)
    plt.plot(radprof,e(radprof))
