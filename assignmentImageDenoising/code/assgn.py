from cmath import inf
import numpy as np
import suppliment as s
import cv2
import matplotlib.pyplot as plt
import scipy.io ## to read .mat file in python

# quadratic- alpha=0.8
# huber- alpha=0.8, gamma=0.1 0.33425
#0.1,0.3
#0.1,0.4
#0.2,0.4
# Rice- alpha=0.8, gamma=0.1 0.33425

# mat = scipy.io.loadmat('../data/brainMRIslice.mat')    ## for MRI data
mat = scipy.io.loadmat('../data/assignmentImageDenoisingPhantom.mat')

LOSSES=[]
Y=mat["imageNoisy"]
X=mat["imageNoiseless"]

# Y=mat["brainMRIsliceNoisy"]    ## for MRI data
# X=mat["brainMRIsliceOrig"]     ## for MRI data

ALPHA=0.007253554917687786
GAMMA=0.24
SIGMA=1
STEP_SIZE=0.1

data = np.array(mat) 
cv2.imshow("Noisy",Y)
cv2.imshow("Noiseless",X)



## defining the noise models and calling the prioirs to generate gradients
def noise_model(image,mode="q",gamma=0.1):
    gradient=0
    loss=0
    
    if mode=="h":
        ## using np.roll to emulate CIRCSHIFT, rolling on +-1 in x
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            new_grad,new_loss=s.huber(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)     
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)   ## noise model
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     
            
            ## using np.roll to emulate CIRCSHIFT, rolling on +-1 in y
            yi=np.roll(image,shift_index,0)
            new_grad,new_loss=s.huber(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)     
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)   ## noise model
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     

    
    if mode=="r":
        ## using np.roll to emulate CIRCSHIFT, rolling on +-1 in x
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            new_grad,new_loss=s.rice(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)     
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)   ## noise model
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     

            ## using np.roll to emulate CIRCSHIFT, rolling on +-1 in y
            yi=np.roll(image,shift_index,0)
            new_grad,new_loss=s.rice(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)     
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)   ## noise model
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     

    else:
        ## using np.roll to emulate CIRCSHIFT, rolling on +-1 in x
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            new_grad,new_loss=s.quadratic(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)     
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)   ## noise model
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)     
            ## using np.roll to emulate CIRCSHIFT, rolling on +-1 in y
            yi=np.roll(image,shift_index,0)
            new_grad,new_loss=s.quadratic(image-yi,gamma)
            gradient+=new_grad*(1-ALPHA)
            loss+=new_loss*(1-ALPHA)                  
            gradient+=2*ALPHA*(image-yi)/(SIGMA**2)    ## noise model
            loss+=ALPHA*(image-yi)**2/(SIGMA**2)      


    return gradient,loss



## dynamic step sizer, to change the step size through the epochs
def dyn_step_sizer(STEP_SIZE,epoch):
    # print(STEP_SIZE)
    return STEP_SIZE-0.0495/2**epoch


#######################CODE FOR FINDING OPTIMAL PARAM#######################

# for gamma in [x*0.1 for x in range(1,10)]:
#     for ALPHA in [x*0.1 for x in range(1,10)]:
#         output=Y
#         STEP_SIZE=0.1
#         for epoch in range(100):
#             STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
#             gradient,loss=noise_model(output,"h",gamma)
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

#######################CODE FOR FINDING OPTIMAL PARAM#######################


loss=inf
output=Y
prev_loss=0
previous=0
LOSSES_q=[]
LOSSES_h=[]
LOSSES_r=[]

ALPHA=0.5
GAMMA=0.4

for epoch in range(100):
    prev_loss=loss
    ####################### use mode = "h" for huber prior, "r" for discontinuity-adaptive function, "q" for quadratic
    gradient,loss=noise_model(output,mode="h",gamma=GAMMA)
    loss=np.sum(loss)
    LOSSES_q.append(loss)
    # if loss >= prev_loss:
        # print(prev_loss)
    STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
    # else:
    #     STEP_SIZE*=1.1
    # print(STEP_SIZE)
    # print(gradient)
    previous=output
    output=output-STEP_SIZE*gradient
    
    if s.rrmse(output,previous)<0.0001:
        print("STOP EPOCH::::::",epoch)
        break
    if epoch%10==0:
        print("curr epoch=",epoch,"\nLoss=",loss)
print(ALPHA,GAMMA,s.rrmse(output,X))
cv2.imshow("output_"+str(ALPHA)+"_"+str(GAMMA),output)
cv2.imwrite("output.jpg",output*255)


################################## repeated CODE for PLOTTING LOSS ##################################

# loss=inf
# output=Y
# prev_loss=0
# previous=0
# ALPHA=0.8
# GAMMA=0.1
# STEP_SIZE=0.1

# for epoch in range(100):
#     prev_loss=loss
#     ####################### use mode = "h" for huber prior, "r" for discontinuity-adaptive function, "q" for quadratic
#     gradient,loss=noise_model(output,mode="h",gamma=GAMMA)
#     loss=np.sum(loss)
#     LOSSES_h.append(loss)
#     # if loss >= prev_loss:
#         # print(prev_loss)
#     STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
#     # else:
#     #     STEP_SIZE*=1.1
#     # print(STEP_SIZE)
#     # print(gradient)
#     previous=output
#     output=output-STEP_SIZE*gradient
    
#     if s.rrmse(output,previous)<0.0001:
#         print("STOP EPOCH::::::",epoch)
#         break
#     if epoch%10==0:
#         print("curr epoch=",epoch,"\nLoss=",loss)
# print(ALPHA,GAMMA,s.rrmse(output,X))
# cv2.imshow("output_"+str(ALPHA)+"_"+str(GAMMA),output)
# cv2.imwrite("output.jpg",output*255)

# loss=inf
# output=Y
# prev_loss=0
# previous=0
# ALPHA=0.189
# GAMMA=0.0000000000000051
# STEP_SIZE=0.1

# for epoch in range(100):
#     prev_loss=loss
#     ####################### use mode = "h" for huber prior, "r" for discontinuity-adaptive function, "q" for quadratic
#     gradient,loss=noise_model(output,mode="r",gamma=GAMMA)
#     loss=np.sum(loss)
#     LOSSES_r.append(loss)
#     # if loss >= prev_loss:
#         # print(prev_loss)
#     STEP_SIZE=dyn_step_sizer(STEP_SIZE,epoch)
#     # else:
#     #     STEP_SIZE*=1.1
#     # print(STEP_SIZE)
#     # print(gradient)
#     previous=output
#     output=output-STEP_SIZE*gradient
    
#     if s.rrmse(output,previous)<0.0001:
#         print("STOP EPOCH::::::",epoch)
#         break
#     if epoch%10==0:
#         print("curr epoch=",epoch,"\nLoss=",loss)
# print(ALPHA,GAMMA,s.rrmse(output,X))
# cv2.imshow("output_"+str(ALPHA)+"_"+str(GAMMA),output)
# cv2.imwrite("output.jpg",output*255)

################################## repeated CODE for PLOTTING LOSS ##################################


cv2.waitKey()

## plotting the losses, through the epochs
plt.plot(LOSSES_q)
# plt.plot(LOSSES_h)
# plt.plot(LOSSES_r)
# plt.legend(["Quadratic prior","Huber prior","Discontinuity-Adaptive prior"])
plt.title("Losses Vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Negative log objective")

plt.show()


