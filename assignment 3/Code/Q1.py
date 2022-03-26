import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2


mat = h5py.File('../data/hands2D.mat')
ITERS=5

## start code 11 (assume standardized data, find rotation)
def find_rotation(Z1,Z2):
    Z1=np.transpose(Z1)
    # print(Z1.shape,Z2.shape)
    product=np.matmul(Z1,Z2)
    U,_,V=np.linalg.svd(product)
    R = np.matmul(V,np.transpose(U))

    if np.linalg.det(R)==-1:
        temp=np.array([[1,0],[0,-1]])
        R=V*temp*np.transpose(U)
    
    return R
## end code 1 (assume standardized data, find rotation)

shapes=mat["shapes"]

for i in range(40):
    plt.scatter(shapes[i,:,0], shapes[i,:,1],s=5)

plt.title("Initial pointsets")
plt.show()



centroid=np.mean(shapes,axis=1).reshape(-1,1,2)
preshapePointSets=shapes-centroid
norm=np.sqrt(np.sum(np.sum(preshapePointSets**2,axis=1),axis=1)).reshape(-1,1,1)
preshapePointSets=preshapePointSets/norm


# initialize the mean
mean_shape=preshapePointSets[0,:,:]

Errors=[]

for iter in range(ITERS):
    ## start code 11
    for i in range(40):
        R=find_rotation(preshapePointSets[i,:,:],mean_shape)

        preshapePointSets[i,:,:]=np.transpose(np.matmul(R,np.transpose(preshapePointSets[i,:,:])))

    ## end code 11

    

    ## start code 22 
    new_mean_shape = np.mean(preshapePointSets,axis=0)
    norm=np.sqrt(np.sum(new_mean_shape**2))
    new_mean_shape = new_mean_shape/norm

    ## end code 22 

    error = np.sqrt(np.sum((new_mean_shape - mean_shape)**2))
    Errors.append(error)

    mean_shape=new_mean_shape





## plot standardized data with mean
plt.plot(mean_shape[:,0],mean_shape[:,1])
for i in range(40):
    plt.scatter(preshapePointSets[i,:,0], preshapePointSets[i,:,1],s=5)
plt.title("sample mean with aligned pointset")
plt.show()

new_shapes=(preshapePointSets-mean_shape).reshape(2*56,40)
covariance = np.matmul(new_shapes,np.transpose(new_shapes))/40

D,V=np.linalg.eig(covariance)

idx = D.argsort()[::-1]   
eigenValues = np.real(D[idx])
eigenVectors = np.real(V[:,idx])

for n in range(3):
    plt.plot(mean_shape[:,0],mean_shape[:,1],label="mean")
    for i in range(40):
        plt.scatter(preshapePointSets[i,:,0], preshapePointSets[i,:,1],s=5)
    pm31 = mean_shape + 2 * (np.sqrt(eigenValues[n])*eigenVectors[:,n]).reshape(-1,2)
    pm32 = mean_shape - 2 * (np.sqrt(eigenValues[n])*eigenVectors[:,n]).reshape(-1,2)

    plt.plot(pm31[:,0], pm31[:,1],label="+ 2sd")
    plt.legend("+")

    plt.plot(pm32[:,0], pm32[:,1],label="- 2sd")
    plt.legend()
    plt.title(str("Principle mode "+str(n)+"(+-2 SD)"))
    plt.show()
