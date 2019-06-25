import numpy as np

X = np.random.rand(1000,3)
w = np.asarray([[1,2,3],[11,12,13],[111,222,333]])
y = np.matmul(X,w)


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
reg.score(X, y)
print(reg.coef_)
print(reg.intercept_)
