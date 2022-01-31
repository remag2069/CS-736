import numpy as np
import suppliment as s
import cv2
import matplotlib.pyplot as plt
import scipy.io


mat = scipy.io.loadmat('../data/assignmentImageDenoisingPhantom.mat')
ALPHA=0.8


data = np.array(mat) 
cv2.imshow("Noisy",mat["imageNoisy"])
cv2.imshow("Noiseless",mat["imageNoiseless"])

def potential(image,mode="q"):
    gradient=0
    
    if mode=="q":
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            gradient+=s.quadratic(image-yi)*(1-ALPHA)
            gradient+=2*ALPHA*(output-yi)/(SIGMA**2)  ##prior
            yi=np.roll(image,shift_index,0)
            gradient+=s.quadratic(image-yi)*(1-ALPHA)
            gradient+=2*ALPHA*(output-yi)/(SIGMA**2)  ##prior
    
    if mode=="h":
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            gradient+=s.huber(image-yi)*(1-ALPHA)
            gradient+=2*ALPHA*(output-yi)/(SIGMA**2)  ##prior
            yi=np.roll(image,shift_index,0)
            gradient+=s.huber(image-yi)*(1-ALPHA)
            gradient+=2*ALPHA*(output-yi)/(SIGMA**2)  ##prior
    
    if mode=="r":
        for shift_index in [-1,1]:
            yi=np.roll(image,shift_index,1)
            gradient+=s.rice(image-yi)
            gradient+=2*(output-yi)/(SIGMA**2)  ##prior
            yi=np.roll(image,shift_index,0)
            gradient+=s.rice(image-yi)
            gradient+=2*(output-yi)/(SIGMA**2)  ##prior

    


    return gradient

Y=mat["imageNoisy"]
X=mat["imageNoiseless"]

SIGMA=1
STEP_SIZE=0.001

# gradients=2*(X-Y)/SIGMA**2 +
output=Y
for epoch in range(100):
    gradient=potential(output,"d")
    previous=output
    output=output-STEP_SIZE*gradient
    if s.rrmse(output,previous)<0.0003:
        print(epoch)
        break



cv2.imshow("output",output)

cv2.waitKey()

print()

#hello dharshan from senthil
