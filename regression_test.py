import numpy as np

from sklearn.linear_model import LinearRegression
w = np.expand_dims(np.asarray([1,1,-2]),1)
w =  w

n = 1000
x = np.asarray([np.arange(i,i+w.shape[0],1)+np.random.rand(len(w)) for i in range(n)])
y = np.matmul(x,w)
xx = x #+ 0.01*np.mean(x)*np.random.normal(size=x.shape, scale=1)


A = np.matmul(xx.transpose(), xx)
b = np.matmul(xx.transpose(),y)
pred = np.linalg.solve(A,b)
print(pred)
#test
xxx = x + 0.01*np.mean(x)*np.random.normal(size=x.shape, scale=1)
print(np.mean(np.abs(y-np.matmul(xxx,pred)))/np.mean(np.abs(y)))

import matplotlib.pyplot as plt
plt.plot(np.matmul(xxx,pred),y,'r.')
plt.show()
pass

#print('cond: ', np.linalg.cond(A))

#independent of the magnitude of: w, x, y, length of w,
            # number of x makes prediction fluctuation smaller
#very related to the magnitude of noise, and the scale of variance



#
# import numpy as np
#
#
# w = np.expand_dims(np.asarray([1,-2,10]),1)
# w =  w
#
# n = 1000
# x = np.asarray([np.arange(i,i+w.shape[0],1)+np.random.rand(len(w)) for i in range(n)])
#
# noise = 0.001*np.mean(x)*np.random.normal(size=x.shape, scale=1)
# xx = x
# yy = np.matmul(x+noise,w)
#
#
# A = np.matmul(xx.transpose(), xx)
#
# b = np.matmul(xx.transpose(),yy)
#
# pred = np.linalg.solve(A,b)
# print(pred/w)
# print('average err: ',1-np.mean((pred/w)))
# print('cond: ', np.linalg.cond(A))
