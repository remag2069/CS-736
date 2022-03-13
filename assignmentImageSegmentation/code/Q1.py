from cmath import inf
import numpy as np
# import suppliment as s
import cv2
import matplotlib.pyplot as plt
import h5py


mat = h5py.File('../data/assignmentSegmentBrain.mat', 'r')

imageMask=np.array(mat["imageMask"])
imageData=np.array(mat["imageData"])
inputData=imageData*imageMask

cv2.imshow("imageMask",imageMask)
cv2.imshow("imageData",imageData)
cv2.imshow("input",inputData)

# print(imageMask,imageMask.shape)

K=3
ITER=1
U =   ## membership matrix
W = [] ## weight matrix
Q = 1.75 ## parameter

## intitalization


## functions

def update_class_means(K, U, image, mask, bias, q):
    class_means=np.zeros(K)
    wij_bi=cv2.filter2D(bias,-1,mask)
    wij_bi2=cv2.filter2D(bias*bias,-1,mask)

    for k in range(K):
        numerator=pow(U[:,:,k],q)*image*wij_bi
        denominator=pow(U[:,:,k],q)*wij_bi2

        class_means[k]=sum(numerator)/sum(denominator)



cv2.waitKey()