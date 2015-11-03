import scipy.io as sio
import scipy as  sp
import scipy.linalg as SLA
import scipy.signal as sig
from numpy import mean, diff, array, std, zeros
import numpy as np
import matplotlib.pyplot as plt
import code


def sort_eig(D,E):
    idx = D.argsort()[::-1]
    D = D[idx]
    E = E[:,idx]
    return D,E

def polynomial_features(x, order):
    x = np.asarray(x).T[np.newaxis]
    n = x.shape[1]
    power_matrix = np.tile(np.arange(order + 1), (n, 1)).T[..., np.newaxis]
    X = np.power(x, power_matrix)
    I = np.indices((order + 1, ) * n).reshape((n, (order + 1) ** n)).T
    F = np.product(np.diagonal(X[I], 0, 1, 2), axis=2)
    return F.T

params = sio.loadmat("params.mat")
thetas= params["thetas"][0][::5]

thetadots = diff(thetas, n=1, axis=0)
data = array([thetas[1::], thetadots])

print(data.shape)


# datamatrix: form order two degree matrix
dm = array([data[0, :],
               data[1, :],
               data[0, :]**2,
               data[0, :] * data[1, :],
               data[1, :]**2,
               ])
dm -= mean(dm,axis=1).reshape(5,1)

dm = dm.T

C = np.dot(dm.T, dm)/data[0,:].shape[0]

D, E = sort_eig(*np.linalg.eig(C))

dw= np.dot(dm, SLA.inv(SLA.sqrtm(C)))

C2  = np.dot(dw.T, dw)/len(thetas)

D2,E2 = sort_eig(*np.linalg.eig(C2))


dslow = zeros((dw.shape[0]-1, dw.shape[1]))
for i in xrange(5):
    dslow[:, i]= sig.convolve(dw[:,i], array([1, 1])/2.,'valid')


C3 = np.dot(dslow.T, dslow) /len(thetas)
D3, E3 = sort_eig(*np.linalg.eig(C3))
plt.figure(1)
plt.subplot(235)
plt.plot(D3)


sfa = np.dot(dw, E3)


plt.figure(num=1, figsize=(20,10))
plt.subplot(231)
plt.plot(thetas)
plt.subplot(232)
plt.plot(thetadots)
plt.subplot(233)
plt.plot(data[0, :], data[1, :])
plt.subplot(234)
plt.plot(D, label="eigen val, before")
plt.plot(D2, label="eigen val, after")
plt.legend()


plt.figure(num=2, figsize=(20,10))
for i in xrange(2 * 3):
    if i < 5:
        print i
        plt.subplot(231 + i)
        plt.plot(dw[:,i])

plt.figure(num=3, figsize=(20,10))
for i in xrange(2 * 3):
    if i < 5:
        plt.subplot(231 + i)
        plt.plot(dslow[:,i])

plt.figure(num=4, figsize=(20,10))
for i in xrange(2 * 3):
    if i < 5:
        plt.subplot(231 + i)
        plt.plot(sfa[:,i])


plt.subplot(236)

plt.plot(sfa[:,0],'o')
plt.plot(sfa[:,1])


plt.figure(num=5, figsize=(20,10))
plt.subplot(411)
plt.plot(sfa[:,0])
plt.subplot(412)
import pickle
forces = pickle.load(open("forces.pickle","rb"))
plt.subplot(413)
plt.plot(forces)
plt.subplot(414)
diffslow = diff(sfa[:,0],n = 1, axis=0)
plt.plot(diff(sfa[:,0],n = 1, axis=0))

# prediction increment in the slowest component
pickle.dump(forces[:-2], open("forces_train.pickle","wb"))
pickle.dump(diffslow, open("diffslow_train.pickle","wb"))
pickle.dump(sfa[:,0][:-1], open("sfa_train.pickle","wb"))



plt.ion()
plt.show()

code.interact(local=locals())

