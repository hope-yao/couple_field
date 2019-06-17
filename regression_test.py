import numpy as np


w = np.expand_dims(np.asarray([1,-2,1]*27),1)
w = 10* w

n = 1000
x = np.asarray([np.arange(i,i+len(w),len(w))*100./n+np.random.rand(len(w))/10. for i in range(n)])
y = np.matmul(x,w)


xx = x + 0.1*x*np.random.normal(size=x.shape, scale=1)

A = np.matmul(xx.transpose(), xx)
invA = np.linalg.inv(A)

#maxA = np.max(A)
#invA2 = np.linalg.inv(A/maxA)/maxA

b = np.matmul(xx.transpose(),y)

pred = np.matmul(invA,b)
print(pred/w)
print('average err: ',np.mean((pred/w)))
print('cond: ', np.linalg.cond(A))

#independent of the magnitude of: w, x, y, length of w,
            # number of x makes prediction fluctuation smaller
#very related to the magnitude of noise, and the scale of variance