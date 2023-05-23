import numpy as np

def rot_z(x,theta):
    rotmatr_z=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    return np.matmul(rotmatr_z,x)

def rot_y(x,theta):
    rotmatr_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]) 
    return np.matmul(rotmatr_y,x)

def rot_x(x,theta):
    rotmatr_x=np.array([[1, 0, 0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return np.matmul(rotmatr_x,x)

