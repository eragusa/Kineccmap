import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import vertical_structure as vs

   
def solve_vert_struct_vel(H0,ecc,varpival,a):
    # array for initial guess (we need H[0] then
    # that will be updated from previous iteration)
    H=[H0,0.]
    e=ecc
    varpi=varpival
    G=1.
    M=1.
    a=1.
    Omega0 = np.sqrt(G*M/a**3)
    
    def Omega_f(phi):
        return Omega0*(1.+e*np.cos(phi-varpi))**2/(1-e**2)**1.5

    def f(phi, Hfunc):
        # Define the parameters
        H,dH_dphi=Hfunc
        Omega=Omega_f(phi)
        Hor=0.1
        cs2=(Hor*a*Omega0)**2
        R=a*(1-e**2)/(1+e*np.cos(phi-varpi))
        dOmegadphi=-2*e*np.sin(phi-varpi)*Omega/(1.+e*np.cos(phi-varpi))
        # Compute the derivatives
        d2H_dphi2 = -G*M/R**3/Omega**2 * H -1/Omega*dOmegadphi*dH_dphi\
                                           +cs2/H/Omega**2
        return [dH_dphi, d2H_dphi2]

    def solve_differential_equation(phi_start, phi_end, num_points,initial_val):
        phi = np.linspace(phi_start, phi_end, num_points)
        dphi = phi[1] - phi[0]  # Step size
    
        # Initialize the arrays
        H = np.zeros(num_points)
        dH_dphi = np.zeros(num_points)
    
        # Initial conditions
        H[0] = initial_val  # Initial displacement
        dH_dphi[0] = 0.0  # Initial velocity
        initial_conditions=[H[0],dH_dphi[0]]

        sol=solve_ivp(f, [phi_start,phi_end],initial_conditions,t_eval=phi)
    
        return sol.t,sol.y[0],sol.y[1]
    
    def shooting_wrapper1(H0):
        phi_start = 0.0
        phi_end = 6.28
        num_points = 1000
        initial_val=H0
        # Solve the differential equation
        phi, H, dHdphi = solve_differential_equation(phi_start, phi_end, num_points,initial_val)
    
        #avoid negative values
        if(np.min(H)<0):
            return np.abs(np.min(H))
        else:
            return H[-1]-H0
    
    ##########################
    
    def shooting_wrapper2(H0):
        phi_start = 0.0
        phi_end = np.pi
        num_points = 1000
        initial_val=H0
        # Solve the differential equation
        phi, H, dHdphi = solve_differential_equation(phi_start, phi_end, num_points,initial_val)
    
        #avoid negative values
        return dHdphi[-1]
    
    #use the value from previous iteration 
    inguess=H[0]
    #find H0 for which the profile is periodic
    val=opt.newton(shooting_wrapper2,inguess,maxiter=100)
    
    #plot the profile
    phi_start = 0.0
    phi_end = 6.28
    num_points=1000
    
    phi, H, dHdphi = solve_differential_equation(phi_start,\
                                                 phi_end,num_points,val)
    dHdt=dHdphi*Omega_f(phi)
    return phi, H, dHdphi, dHdt  

if __name__=='__main__':

    eccval=[0.05,0.1,0.2,0.3,0.4]
    
    H=[0.1,0]
    varpi=0.
    a=1.
    for ecc in eccval:

        phi, H, dHdphi, dHdt=solve_vert_struct_vel(H[0],ecc,varpi,a)

        # Plot the solution
        plt.figure(1)
        plt.plot(phi, H/0.1,label="$e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$H/H_0$')
        plt.ylim([0,2.3])
        plt.legend()
        plt.savefig("../analysis_paper/vertical_analytics/Hvsphi.png")
    
        plt.figure(3)
        plt.plot(phi, dHdt/0.1,label="$e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$v_z/c_{\\rm s}$')
        plt.ylim([-1.5,1.5])
        plt.legend()
        plt.savefig("../analysis_paper/vertical_analytics/dHvsphi.png")
    
    
    for ecc in eccval:
        H0=0.1
    
        phi,H,dHdphi = vs.vert_struct_solver(H0,ecc)
        # Plot the solution
        plt.figure(2)
        plt.plot(phi, H,label="$e=$"+str(ecc))
        plt.xlabel('phi')
        plt.ylabel('H')
        plt.legend()
    
        plt.figure(4)
        plt.plot(phi, dHdphi,label="$e=$"+str(ecc))
        plt.xlabel('phi')
        plt.ylabel('${\\rm d}H/{\\rm d}\phi$')
        plt.legend()
    
        plt.figure(4)
        plt.plot(phi, np.gradient(H,phi),label="$e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('${\\rm d}H/{\\rm d}\phi$')
        plt.ylim([-0.15,0.15])
        plt.legend()
        plt.savefig("../analysis_paper/vertical_analytics/Hvsphi.png")
    
    
    
    plt.show()
        
