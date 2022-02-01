import numpy as np
import suppliment as s
import cv2
import matplotlib.pyplot as plt
import scipy.io

# quadratic- alpha=0.8
# huber- alpha=0.8, gamma=0.1 0.33425

mat = scipy.io.loadmat('../data/assignmentImageDenoisingPhantom.mat')

Y=mat["imageNoisy"]
X=mat["imageNoiseless"]

ALPHA=0.8
SIGMA=1
STEP_SIZE=0.1

data = np.array(mat) 
cv2.imshow("Noisy",mat["imageNoisy"])
cv2.imshow("Noiseless",mat["imageNoiseless"])

def potential(image,mode="q",gamma=0.1):
    gradient=0
    
    if mode=="h":
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            gradient+=s.huber(image-yi,gamma)*(1-ALPHA)
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ##prior
            yi=np.roll(image,shift_index,0)
            gradient+=s.huber(image-yi,gamma)*(1-ALPHA)
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ##prior
    
    if mode=="r":
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            gradient+=s.rice(image-yi,gamma)*(1-ALPHA)
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ##prior
            yi=np.roll(image,shift_index,0)
            gradient+=s.rice(image-yi,gamma)*(1-ALPHA)
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ##prior

    else:
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            gradient+=s.quadratic(image-yi)*(1-ALPHA)
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ##prior
            yi=np.roll(image,shift_index,0)
            gradient+=s.quadratic(image-yi)*(1-ALPHA)
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ##prior            


    return gradient




def dyn_step_sizer(STEP_SIZE,epoch):
    # print(STEP_SIZE)
    return STEP_SIZE-0.0495/2**epoch


for ALPHA in [x*0.1 for x in range(1,10)]:
    for gamma in [x*0.1 for x in range(1,10)]:
        output=Y
        STEP_SIZE=0.1
        for epoch in range(1000):
            STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
            gradient=potential(output,"h",gamma)
            # print(gradient)
            previous=output
            output=output-STEP_SIZE*gradient
            if s.rrmse(output,previous)<0.001:
                print("EPOCH::::::",epoch)
                break
    
        print(ALPHA,gamma,s.rrmse(output,X))




# output=Y
# for epoch in range(1000):
#     STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
#     gradient=potential(output,"h",0.1)
#     # print(gradient)
#     previous=output
#     output=output-STEP_SIZE*gradient
#     if s.rrmse(output,previous)<0.0001:
#         print("EPOCH::::::",epoch)
#         break

# print(ALPHA,s.rrmse(output,X))



# cv2.imshow("output",output)
# # cv2.imwrite("output_0_3.jpg",output*255)

# cv2.waitKey()



