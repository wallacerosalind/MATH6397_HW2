#HW2 1b
import numpy as np
from Data import *

dat = Data()
xtrue, y, A = dat.get_sparse_reg_dat() #A:mxn = 1500x5000
rho = 1 #given
eps_abs = .0001 #given
eps_rel = .01 #given
alpha = np.max(np.matmul(A.transpose(),y))/10 #given; L-inf norm

r = np.matmul(A, x) - y
fx = 0.5*np.inner(r,r)
gz = alpha * np.linalg.norm(z, ord=1)
l = fx + gz + (rho/2) * np.inner(x-z+u, x-z+u)

m = A.shape[0] #m = 1500
n = A.shape[1] #n = 5000

#ADMM algorithm 1
#terminate when primal residual rk <= eps_pri AND dual residual sk <= eps_dual
def getxk1(zk, uk)
    return np.matmul(np.linalg.inv(np.matmul(A.transpose(),A) + rho * np.identity(n)), np.matmul(A.transpose().y) + rho * (zk - uk))
def getzk1(xk1, uk)
    return np.sign(xk1+uk) * np.maximum(np.abs(xk1+uk) - alpha/rho, 0)
def getuk1(uk, xk1, zk1)
    return uk + xk1 -zk1
def geteps_pri(xk, zk)
    return np.sqrt(n) * eps_abs + eps_rel * np.max(np.sqrt(mp.inner(xk,xk)), np.sqrt(mp.inner(-zk,-zk)))
def geteps_dual(uk)
    return np.sqrt(n) * eps_abs + eps_rel * np.sqrt(mp.inner(rho*uk, rho*uk))
def getrk1(xk1, zk1)#primal residual
    return np.inner(xk1 - zk1, xk1 - zk1)
def getsk1(zk1, zk)#dual residual
    return np.inner(-rho*(zk1 - zk),-rho*(zk1 - zk))

#Initial values:
u_0 = 0
z_0 = 0
x = getxk1(z_0, u_0) #x_1
z = getzk1(x,u_0) #z_1
u = getuk1(u_0, x, z) #u_1
r = getrk1(x, z) #r_1
s = getsk1(z, z_0) #s_1
eps_pri = geteps_pri(x, z) #eps_pri_1
eps_dual = geteps_dual(u) #eps_dual_1
while (r > eps_pri) and (s > eps_dual)


# eps_pri = np.sqrt(n) * eps_abs + eps_rel * np.max(np.sqrt(mp.inner(xk,xk)), np.sqrt(mp.inner(-zk,-zk)))
# eps_dual = np.sqrt(n) * eps_abs + eps_rel * np.sqrt(mp.inner(rho*uk, rho*uk))
#xk1 = np.matmul(np.linalg.inv(np.matmul(A.transpose(),A) + rho * np.identity(n)), np.matmul(A.transpose().y) + rho * (zk - uk))
#zk1 = np.sign(xk1+uk) * np.maximum(np.abs(xk1+uk) - alpha/rho, 0)
#uk1 = uk + xk1 -zk1
