import numpy as np
import matplotlib.pyplot as plt
import eccentric_disc.VerticalStructure as evert
from scipy.interpolate import RegularGridInterpolator
import useful_param as up
import interpolation as itp
import pdb
import harmonic_solve as hs

a0=up.ain #6. #be careful R_ref not always coincides with R_in
G=up.G
M=up.M
H0=up.hor #0.04
l=up.flaring #0. #disc flaring

def Omega(a):
    return np.sqrt(G*M/a**3)

def Hcirc(a):
    return H0*(a/a0)**l*a

def cs(a):
    return Hcirc(a)*Omega(a) 

def linear(e,x):
    return 1.0-3.0*e*np.cos(x)

def hydrostatic(e,x):
    return (1.0 - e*np.cos(x))**(1.5)

def calculate_vertical_structure(x,y,ainp,e,cosvarpi,sinvarpi,sigma):
    #it generates H and vz from the semi-major axis profile and gives back a value for each 
    #point in the grid interpolating and shifting gfor phase
    a_arr=np.linspace(ainp.min(),ainp.max(),100)
    e_arr=e(a_arr)
    Eanom_arr=np.linspace(0,2*np.pi,500)

    #creating mesh a,E
    amesh,Eanommesh=np.meshgrid(a_arr,Eanom_arr)

    # vertical structure solver
    vsolver = evert.VerticalStructureIsothermal()
    vsolver.coord_system = 'AE'

    htlist=[]
    dhlist=[]      

    for i in range(len(e_arr)):
        e0=e_arr[i]
        # need to set an inital guess h0=1 corresponds to the circular value is a reasonable starting place.
        # But setting it to the results of the e=0.35 calculation appears to be valid over a wider range of e    
        vsolver.h0=0.19629700736631261 # this is potentially a more useful initial guess
        vsolver.e=e0
        res = vsolver.solve()
    
     
        ht, dh = vsolver(Eanom_arr)
        htlist.append(ht)
        dhlist.append(dh) 

    htarr=np.array(htlist).transpose()
    dharr=np.array(dhlist).transpose()
    
    #obtain physically meaningful quantities H=ht*Hcirc, v_z=dh/ht*cs
    #quantities are dimensionless in H.
    H_arr=htarr*Hcirc(amesh)
    vz_arr=dharr*cs(amesh)#*(2./np.pi)**0.25 #*z/H   #dharr/htarr*cs(amesh)  
    #the 2/pi factor comes from the fact that the way the velocity is calculated 
    #in the sim returns value at z=sqrt(H^2*sqrt(1/2pi)) as Int[z*z/|z|^Gaussian=H*sqrt(1/2pi),{z,0,+Infty}

    #pdb.set_trace()
    interp_H = RegularGridInterpolator((a_arr, Eanom_arr), H_arr.transpose(),bounds_error=False, fill_value=None)
    def H_func(aa,EE):
        return interp_H((aa,EE))*(aa<ainp.max())

    interp_vz = RegularGridInterpolator((a_arr, Eanom_arr), vz_arr.transpose(),bounds_error=False, fill_value=None)
    def vz_func(aa,EE):
        return interp_vz((aa,EE))*(aa<ainp.max())

    #calculating the Eccentric anomaly for each gridelement shifted to account for disc eccentric phase
    f_xy=np.arctan2(y,x)
    e_xy=e(ainp)
    varpi_xy=np.arctan2(sinvarpi(ainp),cosvarpi(ainp))
    f_xy_shifted=np.mod(f_xy-varpi_xy,2*np.pi)
    Eanom_xy_shift=np.mod(2*np.arctan(np.sqrt((1 - e_xy) / (1 + e_xy)) * np.tan(f_xy_shifted/2)),2*np.pi)
    #sigma_arr=sigma(a_arr)
    
    H_xy=H_func(ainp,Eanom_xy_shift)
    vz_xy=vz_func(ainp,Eanom_xy_shift)
        
    return H_xy,vz_xy

def calculate_vertical_structure_bulk(x,y,ainp,e,cosvarpi,sinvarpi,sigma,alphab=0.005):
    #it generates H and vz from the semi-major axis profile and gives back a value for each 
    #point in the grid interpolating and shifting gfor phase
    a_arr=np.linspace(ainp.min(),ainp.max(),100)
    e_arr=e(a_arr)
    varpi_arr=np.arctan2(sinvarpi(a_arr),cosvarpi(a_arr))
    HR_arr=Hcirc(a_arr)/a_arr
    H_arr=Hcirc(a_arr)
    nphi=500
    phi_arr=np.linspace(0,2*np.pi,nphi)
    #creating mesh a,E
    amesh,phimesh=np.meshgrid(a_arr,phi_arr)

    htlist=[]
    dhlist=[]      
    for i in range(len(e_arr)):
        e0=e_arr[i]
        # need to set an inital guess h0=1 corresponds to the circular value is a reasonable starting place.
        # But setting it to the results of the e=0.35 calculation appears to be valid over a wider range of e    
        Hin=[H_arr[i],0.0]
#        if i==78: pdb.set_trace()
        phi_bulk, H_bulk, dHdphi_bulk, dHdt_bulk=hs.solve_vert_struct_vel_bulk_bvp(Hin[0],e0,varpi_arr[i],a_arr[i],Hor=HR_arr[i],alphab=alphab,Hin=Hin,numpoints=nphi)
        #phi_bulk, H_bulk, dHdphi_bulk, dHdt_bulk=hs.solve_vert_struct_vel_bulk(Hin[0],e0,a_arr[i],varpi_arr[i],Hor=HR_arr[i],alphab=alphab,Hin=Hin,numpoints=nphi)
        ht=H_bulk
        dh=dHdt_bulk
        htlist.append(ht)
        dhlist.append(dh) 
        print(e0,a_arr[i],i,ht[0],ht[-1],dh[0],dh[-1]) 

    htarr=np.array(htlist).transpose()
    dharr=np.array(dhlist).transpose()
    
    #obtain physically meaningful quantities H=ht*Hcirc, v_z=dh/ht*cs
    #quantities are dimensionless in H.
    H_arr=htarr
    vz_arr=dharr#*(2./np.pi)**0.25 #*z/H   #dharr/htarr*cs(amesh)  
    #the 2/pi factor comes from the fact that the way the velocity is calculated 
    #in the sim returns value at z=sqrt(H^2*sqrt(1/2pi)) as Int[z*z/|z|^Gaussian=H*sqrt(1/2pi),{z,0,+Infty}

    #pdb.set_trace()
    interp_H = RegularGridInterpolator((a_arr, phi_arr), H_arr.transpose(),bounds_error=False, fill_value=None)
    def H_func(aa,phiphi):
        return interp_H((aa,phiphi))*(aa<ainp.max())

    interp_vz = RegularGridInterpolator((a_arr, phi_arr), vz_arr.transpose(),bounds_error=False, fill_value=None)
    def vz_func(aa,phiphi):
        return interp_vz((aa,phiphi))*(aa<ainp.max())

    #calculating the Eccentric anomaly for each gridelement shifted to account for disc eccentric phase
    f_xy=np.mod(np.arctan2(y,x),np.pi*2.)
    e_xy=e(ainp)
    varpi_xy=np.arctan2(sinvarpi(ainp),cosvarpi(ainp))
    #sigma_arr=sigma(a_arr)
    
    H_xy=H_func(ainp,f_xy)
    vz_xy=vz_func(ainp,f_xy)
#    pdb.set_trace() 
    return H_xy,vz_xy


def vert_struct_solver(H0,e0):
    Omega0=1. #used for cs=H0*Omega0
    Eanom_arr=np.linspace(0,2*np.pi,500)
    f_arr=np.linspace(0,2*np.pi,500)
    # vertical structure solver
    vsolver = evert.VerticalStructureIsothermal()
    vsolver.coord_system = 'AE'

    vsolver.h0=0.19629700736631261 # this is potentially a more useful initial guess
    vsolver.e=e0
    res = vsolver.solve()
    
     
    ht, dh = vsolver(Eanom_arr)
    
    #obtain physically meaningful quantities H=ht*Hcirc, v_z=dh/ht*cs
    H_arr=H0*ht
    vz_arr=dh*H0*Omega0 #dharr/htarr*cs(amesh)   

    #need now to plot it as eccentric anom
    interp_H = itp.interpolator(Eanom_arr,H_arr,80)
    def H_func(EE):
        return interp_H(EE)

    interp_vz =itp.interpolator(Eanom_arr,vz_arr,80)
    def vz_func(EE):
        return interp_vz(EE)

    Eanom_shift=np.mod(2*np.arctan(np.sqrt((1 - e0) / (1 + e0)) * np.tan(f_arr/2)),2*np.pi)
    
    H_anom=H_func(Eanom_shift)
    vz_anom=vz_func(Eanom_shift)

    return f_arr,H_anom,vz_anom   
