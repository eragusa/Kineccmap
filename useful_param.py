import numpy as np

#size of ellipses
ain=2. #at the moment used as minimum radius beyond which strarting looking for the maximum density
fracmax=0.2 #which percentage of density is used to define cavity edge
aout=10.
a0=2.
#eccentricity parameters
e0=0.27
qecc=0.5
#phase parameters
varpi0=-90./180.*np.pi
orbitfrac=0.
#vertical properties
hor=0.05
hormin=0.1*hor #sets the values between which hor oscillate
hormax=hor
parh=1./e0 #rules strength of artificially prescribed h perturbations due to ecc
parvz=1./e0#same as above but for vz
flaring=0.25
#sigma
S0=1.
qsigma=0.5
###########
G=1.#30.**2 #this gives velocity in km/s:  30km/s at R=1 around a M=1 star
M=1.
#orienting your disc in space
i0=0*50./180.*np.pi
PA0=10./180.*np.pi
#for channel maps
nchannels=20
