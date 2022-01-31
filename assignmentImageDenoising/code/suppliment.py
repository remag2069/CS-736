import numpy as np

def quadratic(u):
    loss = np.sum(u**2)
    # print(loss)
    gradient=2*abs(u)
    # print(gradient)
    return gradient


def huber(u):
    gamma=0.1
    loss = np.sum(u**2)
    # print(loss)
    gradient=u
    gradient[abs(gradient)<=gamma]=gamma*gradient[abs(gradient)<=gamma]/abs(gradient[abs(gradient)<=gamma])
    
    # print(gradient)
    return gradient


def rice(u):
    gamma=0.006
    gradient=abs(u)/(1+abs(u)/gamma)*(u/abs(u))
    return gradient



def rrmse(a,b):
    error=np.sqrt(np.sum((a-b)**2)/np.sum(a**2))
    return error