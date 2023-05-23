import numpy as np

#size of ellipses
ain=20.
aout=100.
a0=ain#6.
#eccentricity parameters
e0=0.27
qecc=0.5
#phase parameters 
varpi0=-90./180.*np.pi
orbitfrac=0.
#vertical properties
hor=0.15
hormin=0.1*hor #sets the values between which hor oscillate
hormax=hor
parh=1./e0 #rules strength of artificially prescribed h perturbations due to ecc
parvz=1./e0#same as above but for vz
flaring=0.25
#sigma
S0=1.
qsigma=0.5
###########
G=30.**2
M=2.
#orienting your disc in space
i0=0*50./180.*np.pi
PA0=10./180.*np.pi
#for channel maps
nchannels=20
