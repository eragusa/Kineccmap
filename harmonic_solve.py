import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,solve_bvp
import scipy.optimize as opt
import vertical_structure as vs
import pickle
import useful_param as up
import pdb

def solve_vert_struct_vel(H0,ecc,varpival,a): #a simplifies in the end, it can be kept a=1.
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
                                           +cs2/H/Omega**2#+mu0b*cs2/H/Omega**2
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

def solve_vert_struct_vel_bulk(H000,ecc,varpival,a,Hor=0.1,alphab=0.01,Hin=[0.1,0.],numpoints=1000):
    #H000 is a dummy variable
    # array for initial guess (we need H[0] then
    # that will be updated from previous iteration)
    H=Hin
    e=ecc
    varpi=varpival
    G=1.
    M=1.
    a=a
    Omega0 = np.sqrt(G*M/a**3)
    name=up.name

    ########################### NB ############################
    ## for this to work one needs to run morphology.py first ##
    ###########################################################
    with open(name+'_Jacobian_interp.pkl', 'rb') as file:
        J_f = pickle.load(file)
    with open(name+'_Delta_interp.pkl', 'rb') as file:
        Delta_f = pickle.load(file)
    
    def Omega_f(phi):
        return Omega0*(1.+e*np.cos(phi-varpi))**2/(1-e**2)**1.5

    def f(phi, Hfunc):
        # Define the parameters
        H,dH_dphi=Hfunc
        Omega=Omega_f(phi)
        J=J_f((a,phi))
        Delta=Delta_f((a,phi))
#        pdb.set_trace()
        cs2=(Hor*a*Omega0)**2
        R=a*(1-e**2)/(1+e*np.cos(phi-varpi))
        dOmegadphi=-2*e*np.sin(phi-varpi)*Omega/(1.+e*np.cos(phi-varpi))
        # Compute the derivatives
        d2H_dphi2 = -G*M/R**3/Omega**2 * H -1/Omega*dOmegadphi*dH_dphi\
                                           +cs2/H/Omega**2*(1.-alphab/Omega0*(Delta+Omega*dH_dphi/H))#+cs2/H/Omega**2*
        return [dH_dphi, d2H_dphi2]

    def solve_differential_equation(phi_start, phi_end, num_points,initial_val=Hin):
        phi = np.linspace(phi_start, phi_end, num_points)
        dphi = phi[1] - phi[0]  # Step size
    
        # Initialize the arrays
        H = np.zeros(num_points)
        dH_dphi = np.zeros(num_points)
    
        # Initial conditions
        if(not (initial_val[0]==0)):
            H[0] = initial_val[0]  # Initial displacement
        else:
            return phi,np.zeros_like(phi),np.ones_like(phi)

        dH_dphi[0] = initial_val[1]  # Initial velocity
        initial_conditions=[H[0],dH_dphi[0]]
        #pdb.set_trace()
        sol=solve_ivp(f, [phi_start,phi_end],initial_conditions,t_eval=phi)
        tt=sol.t
        hh=sol.y[0]
        vv=sol.y[1]
        
        return tt,hh,vv

    
    def shooting_wrapper1(H0):
        phi_start = 0.0
        phi_end = 6.28
        initial_val=H0
        # Solve the differential equation
        phi, H, dHdphi = solve_differential_equation(phi_start, phi_end, numpoints,initial_val)
    
        #avoid negative values
        if(np.min(H)<0):
            return [np.abs(np.min(H)),0]
        else:
            return [H[-1]-H0[0],dHdphi[-1]-dHdphi[0]]
    
    ##########################
    
    def shooting_wrapper2(H0):
        phi_start = 0.0
        phi_end = np.pi
        initial_val=H0
        # Solve the differential equation
        phi, H, dHdphi = solve_differential_equation(phi_start, phi_end, numpoints,initial_val)
    
        #avoid negative values
        return [dHdphi[-1]-dHdphi[0],H[-1]-H0[0]]
    
    #use the value from previous iteration 
    inguess=[H[0],H[1]]
    #find H0 for which the profile is periodic
   # if(ecc==0.0390673724370942):
       # pdb.set_trace()
    val=opt.newton(shooting_wrapper1,inguess,maxiter=400)
    
    #plot the profile
    phi_start = 0.0
    phi_end = 6.28
    num_points=numpoints
    
    phi, H, dHdphi = solve_differential_equation(phi_start,\
                                                 phi_end,num_points,val)
    dHdt=dHdphi*Omega_f(phi)
    return phi, H, dHdphi, dHdt  

def solve_vert_struct_vel_bulk_bvp(H000,ecc,varpival,a,Hor=0.1,alphab=0.01,Hin=[0.1,0.],numpoints=1000):
    #H000 is a dummy variable
    # array for initial guess (we need H[0] then
    # that will be updated from previous iteration)
    H=Hin
    e=ecc
    varpi=varpival
    G=1.
    M=1.
    a=a
    Omega0 = np.sqrt(G*M/a**3)
    name=up.name

    ########################### NB ############################
    ## for this to work one needs to run morphology.py first ##
    ###########################################################
    with open(name+'_Jacobian_interp.pkl', 'rb') as file:
        J_f = pickle.load(file)
    with open(name+'_Delta_interp.pkl', 'rb') as file:
        Delta_f = pickle.load(file)
    
    def Omega_f(phi):
        return Omega0*(1.+e*np.cos(phi-varpi))**2/(1-e**2)**1.5

    def f(phi, Hfunc):
        # Define the parameters
        H,dH_dphi=Hfunc
        Omega=Omega_f(phi)
        J=J_f((a,phi))
        Delta=Delta_f((a,phi))
#        pdb.set_trace()
        cs2=(Hor*a*Omega0)**2
        R=a*(1-e**2)/(1+e*np.cos(phi-varpi))
        dOmegadphi=-2*e*np.sin(phi-varpi)*Omega/(1.+e*np.cos(phi-varpi))
        # Compute the derivatives
        d2H_dphi2 = -G*M/R**3/Omega**2 * H -1/Omega*dOmegadphi*dH_dphi\
                                           +cs2/H/Omega**2*(1.-alphab/Omega0*(Delta+Omega*dH_dphi/H))#+cs2/H/Omega**2*
        return [dH_dphi, d2H_dphi2]

    def solve_differential_equation(phi_start, phi_end, num_points,initial_val=Hin):
        phi = np.linspace(phi_start, phi_end, num_points)
        dphi = phi[1] - phi[0]  # Step size
    
        # Initialize the arrays
        H = np.ones(num_points)*initial_val[0]
        dH_dphi = np.ones(num_points)*initial_val[1]
    
        # Initial conditions
        if(not (initial_val[0]==0)):
            H[0] = initial_val[0]  # Initial displacement
        else:
            return phi,np.zeros_like(phi),np.ones_like(phi)

        def bound_con(ya,yb):
            return np.array([ya[0]-yb[0],ya[1]-yb[1]])    
 
        dH_dphi[0] = initial_val[1]  # Initial velocity
        initial_conditions=np.vstack([H,dH_dphi])
        #pdb.set_trace()
        sol=solve_bvp(f,bound_con, phi,initial_conditions)
        tt=phi
        hh=sol.y[0]
        vv=sol.y[1]
        
        return tt,hh,vv

  
    #plot the profile
    phi_start = 0.0
    phi_end = 6.28
    num_points=numpoints
    
    phi, H, dHdphi = solve_differential_equation(phi_start,\
                                                 phi_end,num_points,Hin)
    dHdt=dHdphi*Omega_f(phi)
    return phi, H, dHdphi, dHdt  

def solve_vert_struct_vel_bulk_bvp_nostruct(H000,ecc,varpival,a,Hor=0.1,alphab=0.01,Hin=[0.1,0.],numpoints=1000):
    #because it assumes J=R**2/a and delta accordingly, i.e. constant e[a] constant phase[a]
    #H000 is a dummy variable
    # array for initial guess (we need H[0] then
    # that will be updated from previous iteration)
    H=Hin
    e=ecc
    varpi=varpival
    G=1.
    M=1.
    a=a
    Omega0 = np.sqrt(G*M/a**3)
    name=up.name

    ########################### NB ############################
    ## for this to work one needs to run morphology.py first ##
    ###########################################################
    with open(name+'_Jacobian_interp.pkl', 'rb') as file:
        J_f = pickle.load(file)
    with open(name+'_Delta_interp.pkl', 'rb') as file:
        Delta_f = pickle.load(file)
    
    def Omega_f(phi):
        return Omega0*(1.+e*np.cos(phi-varpi))**2/(1-e**2)**1.5

    def f(phi, Hfunc):
        # Define the parameters
        H,dH_dphi=Hfunc
        Omega=Omega_f(phi)
#        pdb.set_trace()
        cs2=(Hor*a*Omega0)**2
        R=a*(1-e**2)/(1+e*np.cos(phi-varpi))
        J=R**2/a#J_f((a,phi))
        Delta=1./J*np.gradient(J*Omega,phi)#Delta_f((a,phi))
        dOmegadphi=-2*e*np.sin(phi-varpi)*Omega/(1.+e*np.cos(phi-varpi))
        # Compute the derivatives
        d2H_dphi2 = -G*M/R**3/Omega**2 * H -1/Omega*dOmegadphi*dH_dphi\
                                           +cs2/H/Omega**2*(1.-alphab/Omega0*(Delta+Omega*dH_dphi/H))#+cs2/H/Omega**2*
        return [dH_dphi, d2H_dphi2]

    def solve_differential_equation(phi_start, phi_end, num_points,initial_val=Hin):
        phi = np.linspace(phi_start, phi_end, num_points)
        dphi = phi[1] - phi[0]  # Step size
    
        # Initialize the arrays
        H = np.ones(num_points)*initial_val[0]
        dH_dphi = np.ones(num_points)*initial_val[1]
    
        # Initial conditions
        if(not (initial_val[0]==0)):
            H[0] = initial_val[0]  # Initial displacement
        else:
            return phi,np.zeros_like(phi),np.ones_like(phi)

        def bound_con(ya,yb):
            return np.array([ya[0]-yb[0],ya[1]-yb[1]])    
 
        dH_dphi[0] = initial_val[1]  # Initial velocity
        initial_conditions=np.vstack([H,dH_dphi])
        #pdb.set_trace()
        sol=solve_bvp(f,bound_con, phi,initial_conditions)
        tt=phi
        hh=sol.y[0]
        vv=sol.y[1]
        
        return tt,hh,vv

  
    #plot the profile
    phi_start = 0.0
    phi_end = 6.28
    num_points=numpoints
    
    phi, H, dHdphi = solve_differential_equation(phi_start,\
                                                 phi_end,num_points,Hin)
    dHdt=dHdphi*Omega_f(phi)
    return phi, H, dHdphi, dHdt  




if __name__=='__main__':

    eccval=[0.05,0.1,0.2,0.3,0.4]
    
    H=[0.05,0]
    varpi=0.
    a=1.
    al=0.05
    for ecc in eccval:

        phi, H, dHdphi, dHdt=solve_vert_struct_vel(H[0],ecc,varpi,a)
        phi_bulk, H_bulk, dHdphi_bulk, dHdt_bulk=solve_vert_struct_vel_bulk_bvp(H[0],ecc,varpi,a,Hor=0.1,alphab=al)

        # Plot the solution
        plt.figure(1)
        plt.plot(phi, H/0.1,label="$\\alpha_b=$"+str(al)+", $e=$"+str(ecc))
#        plt.plot(phi_bulk, H_bulk/0.1,label="$\\alpha_b=0.1$ $e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$H/H_0$')
        plt.ylim([0,2.3])
        plt.legend()
        plt.savefig("/Users/enricoragusa/Works/eccMap/sim2A500/vertical_analytics/Hvsphi.png")

        plt.figure(12)
        plt.plot(phi_bulk, H_bulk/0.1,label="$\\alpha_b=$"+str(al)+", $e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$H/H_0$')
        plt.ylim([0,2.3])
        plt.legend()
        plt.savefig("/Users/enricoragusa/Works/eccMap/sim2A500/vertical_analytics/Hvsphi_bulk.png")
    
        plt.figure(3)
        plt.plot(phi, dHdt/0.1,label="$\\alpha_b=$"+str(al)+", $e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$v_z/c_{\\rm s}$')
        plt.ylim([-1.5,1.5])
        plt.legend()
        plt.savefig("/Users/enricoragusa/Works/eccMap/sim2A500/vertical_analytics/dHdtvsphi.png")
 
        plt.figure(32)
        plt.plot(phi_bulk, dHdt_bulk/0.1,label="$\\alpha_b=$"+str(al)+", $e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$v_z/c_{\\rm s}$')
        plt.ylim([-1.5,1.5])
        plt.legend()
        plt.savefig("/Users/enricoragusa/Works/eccMap/sim2A500/vertical_analytics/dHdtvsphi_bulk.png")
 
        # Plot the solution
        plt.figure(111)
        plt.plot(phi, H/0.1,label="$e=$"+str(ecc))
        plt.plot(phi_bulk, H_bulk/0.1,label="bulk $e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$H/H_0$')
        plt.ylim([0,2.3])
        plt.legend()
 #       plt.savefig("../analysis_paper/vertical_analytics/Hvsphi.png")
    
        plt.figure(333)
        plt.plot(phi, dHdt/0.1,label="$e=$"+str(ecc))
        plt.plot(phi_bulk, dHdt_bulk/0.1,label="bulk $e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$v_z/c_{\\rm s}$')
        plt.ylim([-1.5,1.5])
        plt.legend()
 #       plt.savefig("../analysis_paper/vertical_analytics/dHdtvsphi.png")
 
  
        phi2=phi*1. 
    
    for ecc in eccval:
        H0=0.1
        #compare with Elliot's H, compare with fig1
        phi,H,dHdt = vs.vert_struct_solver(H0,ecc)
        # Plot the solution
        plt.figure(2)
        plt.plot(phi, H/0.1,label="$e=$"+str(ecc))
        plt.xlabel('phi')
        plt.ylabel('$H/H_0$')
        plt.legend()

        #compare dH/dphi of my solver with Elliot's, obtained taking the gradient of Elliot's H.
        plt.figure(4)
        plt.plot(phi2, dHdphi,label="$e=$"+str(ecc))
        plt.xlabel('phi')
        plt.ylabel('${\\rm d}H/{\\rm d}\phi$')
        plt.legend()
    
        plt.figure(4)
        plt.plot(phi, np.gradient(H,phi),label="$e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('${\\rm d}H/{\\rm d}\phi$')
        plt.ylim([-0.15,0.15])
        plt.legend()
#        plt.savefig("../analysis_paper/vertical_analytics/dHdphivsphi.png")
    
        plt.figure(5)
        plt.plot(phi, dHdt/0.1,label="$e=$"+str(ecc))
        plt.xlabel('$\phi$')
        plt.ylabel('$v_z/c_{\\rm s}$')
        plt.ylim([-1.5,1.5])
        plt.legend()
 #       plt.savefig("../analysis_paper/vertical_analytics/dHdtvsphi.png")
    
    
    plt.show()
        
