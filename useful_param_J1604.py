import numpy as np

#folderres='./analysis_paper/'
#folderres='./sim5A2000/'
folderres='./generate_ellipse/'
img='png'
#size of ellipses
ain=56. #at the moment used as minimum radius beyond which strarting looking for the maximum density
fracmax=0.3 #which percentage of density is used to define cavity edge
aout=265.
a0=56
abin=28.
#eccentricity parameters
e0=0.03
qecc=0.5
#phase parameters
varpi0=-85./180.*np.pi
orbitfrac=0.
#vertical properties
hor=0.1
hormin=0.1*hor #sets the values between which hor oscillate
hormax=hor
parh=1./e0 #rules strength of artificially prescribed h perturbations due to ecc
parvz=1./e0#same as above but for vz
flaring=0.25
#sigma
S0=1.
qsigma=1.
###########
G=30.**2 #this gives velocity in km/s:  30km/s at R=1 around a M=1 star
M=1.24
#orienting your disc in space
i0=6./180.*np.pi
PA0=-10./180.*np.pi
#for channel maps
nchannels=20
####interpolation
npol=3
