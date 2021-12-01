import numpy as np
import matplotlib.pyplot as plt
import datetime
from numpy.core.fromnumeric import shape
from scipy import stats

# initilize points and measurments vectors
num_of_samples  = 100
z = np.zeros((2,num_of_samples))
x = np.zeros(num_of_samples)

#initialize model parameters
Q = np.array([[0.0,0.0],[0.0,1.0]])
A = np.array([[1.0,1.0],[0.0,0.98]])
C = np.array([1.0,0.0]).reshape(1,2)
I = np.eye(2)
R = np.array(100.0).reshape(1,1)
mu_0 = np.array([0.0,0.0])
Sigma_0 = I
# v = stats.multivariate_normal(cov=Q).rvs(num_of_samples)
v = np.random.normal(loc=0.0,scale=1.0, size=num_of_samples)
w = np.random.normal(loc=0,scale=R,size=num_of_samples)
# sample starting point
z[:,0] = np.random.multivariate_normal(mu_0, cov=Sigma_0,size=1)
x[0] = z[0,0] + w[0]
# sample points and measurements
for t in range(1,num_of_samples):
	z[0,t] = z[0,t-1] + z[1,t-1]
	z[1,t] = 0.98 * z[1,t-1] + v[t]
	x[t] = z[0,t] + w[t]


# initialize Kalman filter vectors
P_t_t = np.array([0.0,0.0]).reshape(2,1)
z_t_tMinus1 = np.array([0.0,0.0])
z_t_t = np.array([0.0, 0.0])

predictions, corrections = [], []
# Kalman filter
for t in range(num_of_samples):
	#prediction
	if t == 0:
		z_t_tMinus1 = mu_0.reshape(2,1)
		P_t_tMinus1 = Sigma_0
	else:
		P_t_tMinus1 = A@P_t_t@A.T + Q
		z_t_tMinus1 = A@z_t_t
	#correction
	K = P_t_tMinus1@C.T@(C@P_t_tMinus1@C.T+R)**(-1)
	P_t_t=(I-K@C)@P_t_tMinus1@(I-K@C).T + K@R@K.T
	z_t_t = z_t_tMinus1 + K@(x[t] - C@z_t_tMinus1)
	#store estimations for debugging and plotting
	predictions.append(z_t_tMinus1)
	corrections.append(z_t_t)


# plot 
fig, axs = plt.subplots(2, 1)
axs[0].scatter(range(num_of_samples),z[0,:], label='Ground True - Positions')
axs[0].scatter(range(num_of_samples),x, label='Observations')
axs[0].scatter(range(num_of_samples),np.array(predictions).reshape(100,2)[:,0], label='Kalman Filter Predictions')
axs[0].scatter(range(num_of_samples),np.array(corrections).reshape(100,2)[:,0], label= 'Kalman Filter Corrections')
axs[0].legend()
axs[0].grid(True)
axs[1].scatter(range(num_of_samples),z[1,:], label='Ground True - Velocities')
axs[1].scatter(range(num_of_samples),np.array(predictions).reshape(100,2)[:,1], label='Kalman Filter Predictions Velocities')
axs[1].scatter(range(num_of_samples),np.array(corrections).reshape(100,2)[:,1], label= 'Kalman Filter Corrections Velocities')
axs[1].legend()
axs[1].grid(True)
fig.tight_layout()
plt.show()