import h5py
# import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2


# with h5py.File('../data/hands2D (2).mat', 'r') as f:
#     print(list(f.keys()))

mat = h5py.File('../data/hands2D.mat')
ITERS=5


def find_rotation(Z1,Z2):
    Z1=np.transpose(Z1)
    # print(Z1.shape,Z2.shape)
    product=np.matmul(Z1,Z2)
    U,_,V=np.linalg.svd(product)
    R = np.matmul(V,np.transpose(U))

    if np.linalg.det(R)==-1:
        temp=np.array([[1,0],[0,-1]])
        R=V*temp*np.transpose(U)
    
    print("###############################",R)
    return R

shapes=mat["shapes"]

# for i in range(40):
#     plt.scatter(shapes[i,:,0], shapes[i,:,1],s=5)

# for i in range(ITERS):
#     for j in range()

centroid=np.mean(shapes,axis=1).reshape(-1,1,2)
# temp=np.tile(centroid,[1,56,1])

# print(centroid.reshape(2,-1))
# print(centroid)
preshapePointSets=shapes-centroid
norm=np.sqrt(np.sum(np.sum(preshapePointSets**2,axis=1),axis=1)).reshape(-1,1,1)
preshapePointSets=preshapePointSets/norm


# initialize the mean
mean_shape=preshapePointSets[0,:,:]

Errors=[]

for iter in range(ITERS):
    for i in range(40):
        R=find_rotation(preshapePointSets[i,:,:],mean_shape)

        preshapePointSets[i,:,:]=np.transpose(np.matmul(R,np.transpose(preshapePointSets[i,:,:])))

    # print(preshapePointSets[23,34,1])

    new_mean_shape = np.mean(preshapePointSets,axis=0)
    norm=np.sqrt(np.sum(new_mean_shape**2))
    print("noooorm",new_mean_shape)
    new_mean_shape = new_mean_shape/norm

    error = np.sqrt(np.sum((new_mean_shape - mean_shape)**2))
    # print("###########",iter,norm,error)
    Errors.append(error)

    mean_shape=new_mean_shape

plt.show()

print("~~~~~~~~~~~~~~~~~",preshapePointSets[0,:,:])
print(Errors)


# plt.plot(mean_shape[:,0],mean_shape[:,1])
# for i in range(40):
#     plt.scatter(preshapePointSets[i,:,0], preshapePointSets[i,:,1],s=5)



plt.show()
