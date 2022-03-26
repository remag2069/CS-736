import numpy as np
from getPrior import getPrior

def getlcmLabels( Y, M, K, X, u, s, beta, show_values ):
   [r, c] = np.shape(Y)
   X_c = X
   original_cummulative_value = 0
   cummulative_value = 0
   for count in range(1,11):
       for i in range(1,r-1):
           for j in range(1,c-1):
               if (M[i, j] == 0):
                   continue
               x = X[i][j]
               original_prior = getPrior(X, x, i, j, M, beta)
               u_term = np.dot((u[0][int(x)]), (u[0][int(x)]))
               original_posterior = np.exp(-(1 - beta) * (Y[i][j] - u_term / (2 * s[0][int(x)] * s[0][int(x)])))
               if np.isnan(original_posterior):
                    original_posterior = np.nan_to_num(original_posterior)
            #    print("1111111:", original_prior, original_posterior)
               original_cummulative_value = original_cummulative_value + float(original_prior[0]) * float(original_posterior)
               
               values = np.zeros((1, K))
               for x in range(K):
                   prior = getPrior(X, x, i, j, M, beta)
                   u_term = np.dot((u[0][int(x)]), (u[0][int(x)]))
                   posterior = np.exp(-(1 - beta) * (Y[i][j] - u_term / (2 * s[0][int(x)] * s[0][int(x)])))
                   if np.isnan(posterior):
                       posterior = np.nan_to_num(posterior)
                #    print("222222", prior, posterior)
                #    print(" s[0][int(x)]",  s[0][int(x)])
                #    print(" u[0][int(x)]",  u[0][int(x)])
                   values[0][x] = float(posterior) * float(prior[0])
               [value, index] = [values.max(axis=1), values.argmax(axis=1)]
            #    print("Values", values)
            #    print("values.max(axis=1)", values.max(axis=1))
            #    print("values.argmax(axis=1)", values.argmax(axis=1))
               cummulative_value = cummulative_value  + np.log(value)
               X_c[i][j] = index
       X = X_c*M
       print("count", count)
   
   if show_values == 1:
    print(f'ICM : P(x | y, beta, theta) : {original_cummulative_value} => {cummulative_value})')

   return [X, u, s]











