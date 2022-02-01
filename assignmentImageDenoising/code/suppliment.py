import numpy as np

def quadratic(u,gamma=0):
    loss = np.sum(u**2)
    # print(loss)
    gradient=2*abs(u)
    # print(gradient)
    return gradient,loss


def huber(u,gamma=0.1):
    loss = gamma*abs(u)-0.5*gamma**2
    loss[loss+0.5*gamma**2>gamma**2]=0.5*((loss[loss+0.5*gamma**2>gamma**2]+0.5*gamma**2)/gamma)**2
    # print(loss)
    gradient=u
    gradient[abs(gradient)<=gamma]=gamma*gradient[abs(gradient)<=gamma]/abs(gradient[abs(gradient)<=gamma])
    
    # print(gradient)
    return gradient,loss


def rice(u,gamma=0.006):
    loss=gamma*abs(u)-(gamma**2)*np.log10(1+abs(u)/gamma)
    gradient=abs(u)/(1+abs(u)/gamma)*(u/abs(u))
    return gradient,loss



def rrmse(a,b):
    error=np.sqrt(np.sum((a-b)**2)/np.sum(a**2))
    return error