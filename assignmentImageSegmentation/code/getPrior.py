import numpy as np

def getPrior( X, x, i, j, M, beta):
    diff_count = 0
    if M[i-1][j] == 1:
        diff_count = diff_count + (X[i-1][j] != x)
    if M[i+1][j] == 1:
        diff_count = diff_count + (X[i+1][j] != x)
    if M[i][j-1] == 1:
        diff_count = diff_count + (X[i][j-1] != x)
    if M[i][j+1] == 1:
        diff_count = diff_count + (X[i][j+1] != x)

    out = np.exp(-diff_count * beta)
    return [out]