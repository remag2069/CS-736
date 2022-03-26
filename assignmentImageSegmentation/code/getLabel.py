import numpy as np

def getLabel( Y, M, L ):
    minimum = np.min(Y)
    Y = Y - minimum
    maximum = np.max(Y)
    # print(len(Y[0]))
    X = np.zeros((len(Y[0]), len(Y[1])))
    # print("Y==============", Y)
    # positions = np.logical_and(Y <= maximum, M == 1)
    positions = np.ma.masked_array(Y <= maximum, M == 1, fill_value=1).filled()
    X = np.ma.masked_array(X, positions, fill_value=2).filled()
    # print('np.sum(Y <= maximum) =', np.sum(Y <= maximum))
    # print("np.sum(positions3333) =", np.sum(positions))
    
    
    # positions = np.logical_and(Y <= 2 * maximum / 3, M == 1)
    positions = np.ma.masked_array(Y <= 2 * maximum / 3, M == 1, fill_value=1).filled()
    X = np.ma.masked_array(X, positions, fill_value=1).filled()
    # print('np.sum(Y <= 2 * maximum / 3) =', np.sum(Y <= 2 * maximum / 3))
    # print("np.sum(positions3333) =", np.sum(positions))

    # positions = np.logical_and(Y <= maximum / 3, M == 1)
    positions = np.ma.masked_array(Y <= maximum / 3, M == 1, fill_value=1).filled()
    X = np.ma.masked_array(X, positions, fill_value=0).filled()
    # print('np.sum(Y <= maximum / 3) =', np.sum(Y <= maximum / 3))
    # print("np.sum(positions3333) =", np.sum(positions))

    return X









    