import numpy as np
import discEccanalysis_pysplash as de
import matplotlib.pyplot as plt

frgrid=0.05
res=de.loadHDF5('datasim.h5')
index=100
ecc=np.abs(res['evecA'][index,:])
phase=np.angle(res['evecA'][index,:])
radprof=res['radProf'][:]
sigma=res['sigmaA'][index,:]
npol=20

ee=np.polynomial.Chebyshev.fit(radprof,ecc,npol)
def e(a):
    return ee(a)*(a<radprof[-int(frgrid*len(radprof))])

varpivarpi=np.polynomial.Chebyshev.fit(radprof,phase,npol*2)
def varpi(a):
    return varpivarpi(a)*(a<radprof[-int(frgrid*len(radprof))])

cos_interp=np.polynomial.Chebyshev.fit(radprof,np.cos(phase),npol*2)
def cosvarpi(a):
    return cos_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

sin_interp=np.polynomial.Chebyshev.fit(radprof,np.sin(phase),npol*2)
def sinvarpi(a):
    return sin_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

sigma_interp=np.polynomial.Chebyshev.fit(radprof,sigma,6*npol)
def sigma_a(a):
    return sigma_interp(a)*(a<radprof[-int(frgrid*len(radprof))])

aa=np.linspace(2,14,30)
varpi=np.arctan2(sinvarpi(aa),cosvarpi(aa))
ecc=e(aa)

theta=np.linspace(0,np.pi*2.)

for i in range(len(aa)):
    r=aa[i]*(1-ecc[i]**2)/(1+ecc[i]*np.cos(theta-varpi[i]))
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    plt.plot(x,y)

plt.axis('equal')
