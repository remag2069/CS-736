from cmath import inf
import numpy as np
import suppliment as s
import cv2
import matplotlib.pyplot as plt
import scipy.io

# quadratic- alpha=0.8
# huber- alpha=0.8, gamma=0.1 0.33425
#0.1,0.3
#0.1,0.4
#0.2,0.4
# Rice- alpha=0.8, gamma=0.1 0.33425

mat = scipy.io.loadmat('../data/assignmentImageDenoisingPhantom.mat')
LOSSES=[]
Y=mat["imageNoisy"]
X=mat["imageNoiseless"]

ALPHA=0.5
GAMMA=0.4
SIGMA=1
STEP_SIZE=0.1

data = np.array(mat) 
cv2.imshow("Noisy",mat["imageNoisy"])
cv2.imshow("Noiseless",mat["imageNoiseless"])

def potential(image,mode="q",gamma=0.1):
    gradient=0
    loss=0
    
    if mode=="h":
        for shift_index in [-1,1]:
            # print(ALPHA)
            yi=np.roll(image,shift_index,1)
            new_grad,new_loss=s.huber(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)                 ## 
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ## prior grads
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     ## prior losses

            yi=np.roll(image,shift_index,0)
            new_grad,new_loss=s.huber(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)                 ## 
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ## prior grads
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     ## prior losses

    
    if mode=="r":
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            new_grad,new_loss=s.rice(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)                 ## 
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ## prior grads
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     ## prior losses

            yi=np.roll(image,shift_index,0)
            new_grad,new_loss=s.rice(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)                 ## 
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ## prior grads
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     ## prior losses

    else:
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            new_grad,new_loss=s.quadratic(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)                 ## 
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ## prior grads
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     ## prior losses

            yi=np.roll(image,shift_index,0)
            new_grad,new_loss=s.quadratic(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)                 ## 
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)  ## prior grads
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     ## prior losses           


    return gradient,loss




def dyn_step_sizer(STEP_SIZE,epoch):
    # print(STEP_SIZE)
    return STEP_SIZE-0.0495/2**epoch


# for gamma in [x*0.4 for x in range(1,10)]:
#     for ALPHA in [x*0.1 for x in range(1,10)]:
#         output=Y
#         STEP_SIZE=0.1
#         for epoch in range(100):
#             STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
#             gradient,loss=potential(output,"h",gamma)
#             # print(gradient)
#             previous=output
#             output=output-STEP_SIZE*gradient
#             if s.rrmse(output,previous)<0.001:
#                 print("EPOCH::::::",epoch)
#                 break
    
#         print(ALPHA,gamma,s.rrmse(output,X))

#         # cv2.imshow("output",output)
#         cv2.imwrite("output_"+str(gamma*10)+".jpg",output*255)
#         # cv2.waitKey()

#     break


# for ALPHA in [0.5*0.8,0.5,0.5*1.2]:
#     for gamma in [0.4*0.8,0.4,0.4*1.2]:
        
#         loss=inf

#         output=Y
#         prev_loss=0
#         previous=0
#         for epoch in range(100):
#             prev_loss=loss
#             gradient,loss=potential(output,"h",GAMMA)
#             loss=np.sum(loss)
#             LOSSES.append(loss)
#             # if loss >= prev_loss:
#                 # print(prev_loss)
#             STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
#             # else:
#             #     STEP_SIZE*=1.1
#             # print(STEP_SIZE)
#             # print(gradient)
#             previous=output
#             output=output-STEP_SIZE*gradient
            
#             if s.rrmse(output,previous)<0.0001:
#                 print("STOP EPOCH::::::",epoch)
#                 break
#             if epoch%10==0:
#                 print("curr epoch=",epoch,"\nLoss=",loss)

#         print(ALPHA,gamma,s.rrmse(output,X))



#         cv2.imshow("output_"+str(ALPHA)+"_"+str(gamma),output)
#         # cv2.imwrite("output_1000epoch.jpg",output*255)

# cv2.waitKey()


# plt.plot(LOSSES)
# plt.show()



temp1=0.00001
temp2=0.216

for ALPHA in [temp1*0.8,temp1,temp1*1.2]:
    for gamma in [temp2*0.8,temp2,temp2*1.2]:
        output=Y
        STEP_SIZE=0.1
        for epoch in range(100):
            STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
            gradient,loss=potential(output,"r",gamma)
            # print(gradient)
            previous=output
            output=output-STEP_SIZE*gradient
            if s.rrmse(output,previous)<0.001:
                print("EPOCH::::::",epoch)
                break
    
        print(ALPHA,gamma,s.rrmse(output,X))

        # cv2.imshow("output",output)
        cv2.imwrite("output_"+str(ALPHA*10)+str(gamma*10)+".jpg",output*255)
        # cv2.waitKey()