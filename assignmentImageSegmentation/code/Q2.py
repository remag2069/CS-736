
import numpy as np
import h5py
# import cv2
import matplotlib.pyplot as plt

from getLabel import getLabel
from getEMLabels import getEMLabels

# with h5py.File('assignmentImageSegmentation/data/assignmentSegmentBrainGmmEmMrf.mat', 'r') as f:
#     print(f.keys())
mat = h5py.File('assignmentImageSegmentation/data/assignmentSegmentBrainGmmEmMrf.mat')

Y = mat["imageData"]
M = mat["imageMask"]
K = 3
X = getLabel(Y, M, K)
u = np.zeros((1, K))
s = np.zeros((1, K))
beta = 0.16

for label in range(K):
    positions = (X == label)
    if(np.sum(Y[positions]) == 0):
        u[0][label] = 0
        s[0][label] = 0
    u[0][label] = np.mean(Y[positions])
    s[0][label] = np.std(Y[positions])
    # print(s[0][label], "s[0]==========================", u[0][label], label, X, "sum", np.sum(Y[positions]))
    # print("Argwheres:", np.argwhere(np.isnan(positions)), np.argwhere(np.isnan(Y[positions])))


[L, G] = getEMLabels(Y, M, K, X, u, s, beta, 1)
_, axs = plt.subplots(2,5, figsize=(18, 42))
axs[0,0].set_title('Original Corrupted Image')
axs[0,0].imshow((mat["imageData"]))
axs[0,1].set_title(f'Label 1, beta = {beta}')
axs[0,1].imshow(G[:, :, 0])
axs[0,2].set_title(f'Label 2, beta = {beta}')
axs[0,2].imshow(G[:, :, 1])
axs[0,3].set_title(f'Label 3, beta = {beta}')
axs[0,3].imshow(G[:, :, 2])
axs[0,4].set_title(f'GMM-MRF-EM Optimised Image Segmentation for beta = {beta}')
axs[0,4].imshow(L)
plt.show()
[L, G] = getEMLabels(Y, M, K, X, u, s, 0, 1)

axs[1,1].set_title('Label 1, beta = 0')
axs[1,1].imshow(G[:, :, 0])
axs[1,2].set_title('Label 2, beta = 0')
axs[1,2].imshow(G[:, :, 1])
axs[1,3].set_title('Label 3, beta = 0')
axs[1,3].imshow(G[:, :, 2])
axs[1,4].set_title('GMM-MRF-EM Optimised Image Segmentation for beta = 0')
axs[1,4].imshow(L)

plt.show()













