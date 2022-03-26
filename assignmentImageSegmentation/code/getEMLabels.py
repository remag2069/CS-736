import numpy as np

from getlcmLabels import getlcmLabels
from getPrior import getPrior

def getEMLabels( Y, M, K, X, u, s, beta, show_values):
   [r, c] = np.shape(Y)
   
   Gamma = np.zeros((r, c, K))
   membership_beta = beta
   
   for count in range(1,20):
       [X, u, s] = getlcmLabels( Y, M, K, X, u, s, beta, show_values)
    #    print("getlcmLabels", X,u,s)
       for i in range(1,r-1):
           for j in range(1,c-1):
               if (M[i][j] == 0):
                   continue
               memberships = np.zeros((1, K))
               for x in range(K):
                   prior = getPrior(X, x, i, j, M, membership_beta)
                   u_term = np.dot((u[0][int(x)]), (u[0][int(x)]))
                   posterior = np.exp(-(1-membership_beta) * (Y[i][j] - u_term/ (2 * s[0][int(x)] * s[0][int(x)])))
                #    print(x, " and s[0][int(x)] = ", s[0][int(x)], "s IS: ", s)
                   if np.isnan(posterior):
                    posterior = np.nan_to_num(posterior)
                   if np.isnan(float(prior[0])):
                    prior[0] = np.nan_to_num(float(prior[0]))
                   memberships[0][x] = float(posterior) * float(prior[0])
                #    print("Posterior:", float(posterior) , "Prior", float(prior[0]), "Product", memberships[0][x])

               memberships = memberships/np.sum(memberships)
               if (np.sum(memberships == False) <= 0):
                Gamma[i][j][:] = memberships
       for label in range(K):
        #    print("Gammmma", Gamma[:,:,label], np.sum(Gamma))
        #    print("argwhere", np.argwhere(np.isnan(Gamma)))
           D = np.sum(Gamma[:,:,label])
           S = Gamma[:,:,label] * Y
           S = np.sum(S)
        #    print("1111111111111111=====",  u[0][label], S, D)
           u[0][label] = float(S / D)

           S = np.square(Y-u[0][label])
           S = Gamma[:,:,label] * S
           S = np.sum(S)
        #    print("2222222222222222=====",  u[0][label], S, D)
           s[0][label] = np.sqrt(S / D)
       print("Big count = ", count)
   return [X, Gamma]







