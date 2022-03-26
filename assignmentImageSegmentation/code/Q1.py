from cmath import inf, nan
from email.mime import image
import numpy as np
# import suppliment as s
import cv2
import matplotlib.pyplot as plt
import h5py
from sklearn.cluster import KMeans as kmeans


mat = h5py.File('../data/assignmentSegmentBrain.mat', 'r')

imageMask=np.array(mat["imageMask"])
imageData=np.array(mat["imageData"])
input_image=imageData*imageMask

# cv2.imshow("imageMask",imageMask)
cv2.imshow("imageData",imageData)
cv2.imshow("input",input_image)





## functions

def update_class_means(K, U, image, mask, bias, Q):
    class_means=np.zeros(K)
    
    wij_bi=cv2.filter2D(bias,-1,mask)
    wij_bi2=cv2.filter2D(bias*bias,-1,mask)

    for k in range(K):
        numerator=(pow(U[:,:,k],Q)*wij_bi)*(image)
        temp=pow(U[:,:,k],Q)
        
        denominator=pow(U[:,:,k],Q)*wij_bi2

        class_means[k]=np.mean(numerator)/np.mean(denominator)

    return class_means


def update_membership(class_means, image, mask, bias, Q):
    I,J=image.shape
    U=np.zeros([I,J,len(class_means)])

    wij_bi=cv2.filter2D(bias,-1,mask)
    wij_bi2=cv2.filter2D(bias*bias,-1,mask)

    for k in range(len(class_means)):
        djk=(class_means[k]*class_means[k]*wij_bi2)+ image*image -2*class_means[k]*(wij_bi*image) 

        djk=djk+(djk==0)*np.mean(djk)
        U[:,:,k]=pow(djk,(1/(1-Q)))/100
        # print(djk[177][127])

    # print(np.argwhere(np.isnan(U)))
    # print("U",np.any(U==0))
    for k in range(len(class_means)):
        U[:,:,k]=U[:,:,k]/np.sum(U,2)
    
    

    return U

def update_bias_field(class_means, U, image, mask, Q):

    ujk__q_ck=pow(U,Q)
    ujk__q_ck__2=pow(U,Q)

    for k in range(len(class_means)):
        ujk__q_ck[:,:,k]=ujk__q_ck[:,:,k]*class_means[k]
        ujk__q_ck__2[:,:,k]=ujk__q_ck__2[:,:,k]*class_means[k]*class_means[k]

    ujk__q_ck=np.sum(ujk__q_ck,2)
    ujk__q_ck__2=np.sum(ujk__q_ck__2,2)

    numerator=image*ujk__q_ck
    numerator=cv2.filter2D(numerator,-1,mask)

    denominator=cv2.filter2D(ujk__q_ck__2,-1,mask)

    bias=numerator/denominator


    return bias

def loss_fn(class_means, U, image, mask, Q, bias):
    wij_bij__2=cv2.filter2D(bias*bias,-1,mask)
    wij_bij=cv2.filter2D(bias,-1,mask)

    loss=0

    for k in range(len(class_means)):
        temp=pow(U[:,:,k],Q)*(image**2 + class_means[k]*wij_bij__2 - 2*class_means[k]*image*wij_bij)
        loss+=np.sum(temp)
    
    return loss


K=3
MAX_ITER=20
U = [] ## membership matrix
W = [] ## weight mask
Q = 1.55 ## parameter


## intitalization
gaussian1D = cv2.getGaussianKernel(3, 1)
W = np.outer(gaussian1D, gaussian1D)
kmeans_model = kmeans(n_clusters=3).fit(input_image.reshape(-1,1))
# C = kmeans_model.cluster_centers_
C = np.array([0.456,0.635,0.0006])


U = np.ones([256,256,K])/K
bias = np.ones([256,256])/2



for i in range(3):
    t=U[:,:,i]
    t_norm = (t-t.min())/(t.max()-t.min())
    # for m in range(256):
    #     for n in range(256):
    #         if t[m][n]>0.5:
    #             t[m][n]=1
    #         else:
    #             t[m][n]=0
    
    plt.imsave("Initial_class_"+str(i+1)+".jpg",t_norm,cmap=plt.cm.gray)

## main
losses=[]

for iter in range(MAX_ITER):
    losses.append(1/np.log10(loss_fn(C,U,input_image,W,Q,bias)))
    if iter%1==0:
        print("Iteration:",iter+1,"====> Log loss:",losses[-1])
        # print(C,"\n mask\n",W,"\n bias\n",bias,"\n Q\n",Q)
    
    C=abs(update_class_means(3,U,input_image,W,bias,Q))
    U=abs(update_membership(C,input_image,W,bias,Q))
    bias=abs(update_bias_field(C,U,input_image,W,Q))


cv2.imshow("bias",bias)
cv2.imshow("input-bias",input_image-bias)
plt.plot(losses)
plt.title("Loss function")
plt.show()
plt.imshow(W,cmap=plt.cm.gray)
plt.title("neighbourhood mask")
plt.show()


A=np.zeros([256,256])

for k in range(len(C)):
    A=A+U[:,:,k]*C[k]

R=input_image-A*bias

cv2.imshow("residual",R)
cv2.imshow("a",A)
for i in range(3):
    t=U[:,:,i]
    t_norm = (t-t.min())/(t.max()-t.min())
    # for m in range(256):
    #     for n in range(256):
    #         if t[m][n]>0.5:
    #             t[m][n]=1
    #         else:
    #             t[m][n]=0
    
    plt.imsave("class"+str(i+1)+".jpg",t_norm,cmap=plt.cm.gray)

plt.imsave("U.jpg",U,cmap=plt.cm.jet)
cv2.waitKey()



print(C)