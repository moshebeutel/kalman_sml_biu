import numpy as np
import matplotlib.pyplot as plt
import datetime
from numpy.core.fromnumeric import shape
from scipy import stats

#initialize parameters
num_of_samples  = 100
Q = np.array([[0.0,0.0],[0.0,1.0]])
z = np.zeros((num_of_samples,2))
x = np.zeros(num_of_samples)
# v = stats.multivariate_normal(cov=Q).rvs(num_of_samples)
v = np.random.normal(loc=0.0,scale=1.0, size=num_of_samples)
w = np.random.normal(loc=0,scale=10.0,size=num_of_samples)
# sample starting point
z[0,0] = np.random.normal(loc=0.0,scale=1.0,size=1)
z[0,1] = np.random.normal(loc=0.0,scale=1.0,size=1)
x[0] = z[0,0] + w[0]
# sample points and measurements
for t in range(1,num_of_samples):
	z[t,0] = z[t-1,0] + z[t-1,1]
	z[t,1] = 0.98 * z[t-1,1] + v[t]
	x[t] = z[t,0] + w[t]

# plot points
fig = plt.figure()
plt.scatter(range(num_of_samples),z[:,0])
# plt.scatter(range(num_of_samples),z[:,1])
plt.scatter(range(num_of_samples),x)

plt.show()