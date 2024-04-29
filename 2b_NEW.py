import numpy as np
from scipy.sparse import identity
from Data import *
import scipy
dat = Data()
xtrue, y = dat.get_denoise_data1D()
#print('y.shape[0]')
#print(y.shape[0])
#print(xtrue.shape)
y=y.reshape(y.shape[0], 1, order='F')
alpha=5
rho=1
eps_abs = .0001
eps_rel = .01
n=100
m=1500
e = np.ones(n)
F = scipy.sparse.spdiags([e,-e],[0,1],n,n).toarray()
#print('F shape')
#print(F.shape)
FT = F.transpose()
A = identity(n).toarray()

def getxk1(zk, uk): #2b_NEW
    uk = uk.reshape(n, 1, order='F')
    zk = zk.reshape(n, 1, order='F')
    #print('zk - uk')
    #print((zk - uk).shape)
    #print('(np.matmul(FT, zk - uk)).shape')
    #print((np.matmul(FT, zk - uk)).shape)
    #print('y')
    #print(y.shape)
    S1 = (np.linalg.inv(A + np.matmul(FT,F)))
    S2 = (y + np.matmul(FT, zk - uk))#.reshape(n, 1, order='F')
    #print('S1')
    #print(S1.shape)
    #print('S2')
    #print(S2.shape)
    return np.matmul(S1,S2)
    #return np.matmul((np.linalg.inv(A + rho * np.matmul(FT,F))).transpose(), (y + rho * np.matmul(FT, zk - uk)).transpose())
def getzk1(xk1, uk):
    xk1 = xk1.reshape(n,1,order='F')
    uk = uk.reshape(n,1,order='F')
    return np.sign(np.matmul(F, xk1 + uk)) * np.maximum(np.abs(np.matmul(F, xk1 + uk)) - alpha / rho, 0)

def getuk1(uk, xk1, zk1):
    return uk + xk1 -zk1
def geteps_pri(xk, zk):
    xk = xk.reshape(n, 1, order='F')
    zk = zk.reshape(n, 1, order='F')
    #return (np.sqrt(n) * eps_abs + eps_rel * np.maximum(np.sqrt(np.inner((xk),(xk))), np.sqrt(np.inner((-zk),(-zk)))))
    return (np.sqrt(n) * eps_abs + eps_rel * np.maximum(np.sqrt(np.inner((xk).transpose(),(xk).transpose())), np.sqrt(np.inner((-zk).transpose(),(-zk).transpose()))))
def geteps_dual(uk):
    uk = uk.reshape(n, 1, order='F')
    return (np.sqrt(n) * eps_abs + eps_rel * np.sqrt(np.inner(uk.transpose(), uk.transpose())))
def getrk1(xk1, zk1):#primal residual
    xk1 = xk1.reshape(n, 1, order='F')
    zk1 = zk1.reshape(n, 1, order='F')
    rk1 = np.inner((xk1 - zk1).transpose(), (xk1 - zk1).transpose())
    return rk1
def getsk1(zk1, zk):#dual residual
    zk1 = zk1.reshape(n, 1, order='F')
    zk = zk.reshape(n, 1, order='F')
    return np.inner((-rho*(zk1 - zk).transpose()),(-rho*(zk1 - zk).transpose()))

#Initial values:
u_0 = np.zeros(n)
u_0 = u_0.reshape(n, 1, order='F')
z_0 = np.zeros(n)
z_0 = z_0.reshape(n, 1, order='F')
x = getxk1(z_0, u_0)
zk = getzk1(x,u_0) #z_1   *******************
uk = getuk1(u_0, x, zk) #u_1
r = getrk1(x, zk) #r_1
s = getsk1(zk, z_0) #s_1
eps_pri = geteps_pri(x, zk) #eps_pri_1
eps_dual = geteps_dual(uk) #eps_dual_1


iter = 0 #debug: count how many iterations of while loop
while (r > eps_pri) and (s > eps_dual):
    iter += 1
    xk1 = getxk1(zk,uk) #x_2 is first assignment
    zk1 = getzk1(xk1,uk)
    uk1 = getuk1(uk,xk1,zk1)
    r = getrk1(xk1, zk1) #update
    s = getsk1(zk1, zk) #update
    eps_pri = geteps_pri(xk1, zk1) #update
    eps_dual = geteps_dual(uk1) #update
    zk = zk1 #iterate
    uk = uk1 #iterate

#dat.check_class(xk1,xtrue)
#xtrue = xtrue.reshape(n, 1, order='F')
#diff = np.inner(xk1.transpose(),xtrue.transpose())
print('Count of while loop iterations:')
print(iter)
print('Terminal ||r||^2_2 = ')
print(r)
print('Terminal ||s||^2_2 = ')
print(s)
