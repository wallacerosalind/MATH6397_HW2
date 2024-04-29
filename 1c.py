#HW2 1c
from Data import *
from scipy.sparse import identity
import scipy

dat = Data()
xtrue, y, A = dat.get_sparse_reg_dat() #A:mxn = 1500x5000
m = A.shape[0] #m = 1500
n = A.shape[1] #n = 5000
rho = 1 #given
eps_abs = .0001 #given
eps_rel = .01 #given
alpha = np.max(np.matmul(A.transpose(),y))/10 #given; L-inf norm
M = identity(m).toarray() + rho * np.matmul(A,A.transpose())
L= scipy.linalg.cholesky(M,lower=True)
U= scipy.linalg.cholesky(M,lower=False)
W = identity(n).toarray() - rho * np.matmul(np.matmul(A.transpose(),np.linalg.inv(M)),A)

def getfx(A,x,y):
    res = np.matmul(A, x) - y
    return 0.5*np.inner(res.transpose(),res.transpose())
def getgz(z):
    return alpha * np.linalg.norm(z, ord=1)
def getlag(A,x,y,z,u):
    return getfx(A,x,y) + getgz(z) + (rho/2) * np.inner((x-z+u).transpose(), (x-z+u).transpose())

#ADMM
#terminate when primal residual rk <= eps_pri AND dual residual sk <= eps_dual
def getxk1(zk, uk): #CHOLESKY
    return np.matmul(W, np.matmul(A.transpose(),y) + rho * (zk - uk))
def getzk1(xk1, uk):
    return np.sign(xk1+uk) * np.maximum(np.abs(xk1+uk) - alpha/rho, 0)
def getuk1(uk, xk1, zk1):
    return uk + xk1 -zk1
def geteps_pri(xk, zk):
    return (np.sqrt(n) * eps_abs + eps_rel * np.maximum(np.sqrt(np.inner((xk).transpose(),(xk).transpose())), np.sqrt(np.inner((-zk).transpose(),(-zk).transpose()))))
def geteps_dual(uk):
    return (np.sqrt(n) * eps_abs + eps_rel * np.sqrt(np.inner((rho*uk).transpose(), (rho*uk).transpose())))
def getrk1(xk1, zk1):#primal residual
    rk1 = np.inner((xk1 - zk1).transpose(), (xk1 - zk1).transpose())
    return rk1
def getsk1(zk1, zk):#dual residual
    return np.inner((-rho*(zk1 - zk)).transpose(),(-rho*(zk1 - zk)).transpose())

#Initial values:
u_0 = np.zeros(n)
z_0 = np.zeros(n)
x = getxk1(z_0, u_0) #x_1
zk = getzk1(x,u_0) #z_1
uk = getuk1(u_0, x, zk) #u_1
r = getrk1(x, zk) #r_1
s = getsk1(zk, z_0) #s_1
eps_pri = geteps_pri(x, zk) #eps_pri_1
eps_dual = geteps_dual(uk) #eps_dual_1
while (np.inner(r.transpose(),r.transpose()) > eps_pri) and (np.inner(s.transpose(),s.transpose()) > eps_dual):
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

iter = 0 #debug: count how many iterations of while loop


print('Count of while loop iterations:')
print(iter)
print('Terminal ||r||^2_2 = ')
print(r)
print('Terminal ||s||^2_2 = ')
print(s)