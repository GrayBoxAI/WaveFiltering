# live (animated) demo of LQR (linear-quadratic regulator) with the lifted LDS

import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import hilbert
from sklearn import linear_model
import matplotlib
import time

# State Space

def get_filters(n, k):
	H = hilbert(n)
	vals,vecs = np.linalg.eig(H)
	idx = np.abs(vals).argsort()[::-1]
	return np.real( vecs[:, idx[:k]] )

def simulate(sys, u, h0=None, noisy=True):
	A,B,C,D = sys
	u = u.reshape(u.shape[0],-1)
	num_steps = u.shape[1]

	hidden = np.zeros((A.shape[0], num_steps+1))
	output = np.zeros((D.shape[0], num_steps))

	if h0 is not None:
		hidden[:,0] = h0

	for t in range(num_steps):
		if noisy:
			h_noise = 0.01*np.random.randn(A.shape[0])
			o_noise = 0.01*np.random.randn(*D.shape)
		else:
			h_noise = 0
			o_noise = 0
		hidden[:, t+1] = A.dot(hidden[:,t]) + B.dot(u[:,t]) + h_noise
		output[:, t] = C.dot(hidden[:, t]) + D + o_noise

	return hidden, output, range(num_steps)

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
 
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K, X, eigVals


if __name__ == "__main__":

	A = np.array([[0.99, 0], [0, 0.99]])
	#B = np.array([[1, 1], [1, 1]])
	B = np.array([[1],[2]])
	C = np.array([[1, 1]])
	D = np.array([[0]])

	sys = (A, B, C, D)
	u = np.random.randn(1,10000).reshape(1,10000)
	hidden, output, t = simulate(sys, u, h0=np.array([1, 1]), noisy=False)

	print(hidden)
	print(output)
	print(len(t))

	horizon = 200
	num_feat = 50
	filters = get_filters(horizon, num_feat)

	reg = linear_model.Ridge(alpha=0.001)
	feats = []
	targets = []
	for i in range(horizon, len(t)-1000):
		#z = np.append(u[:,i-horizon:i].dot(filters).reshape(-1,1), [[1]], axis=0)
		z = u[:,i-horizon:i].dot(filters).reshape(-1,1)
		feats.append(z.dot(z.T).reshape(-1,))
		targets.append((0-output[:,i])**2)
		#OGD.fit(feat, output[:,i])
	feats = np.stack(feats, axis=1)
	targets = np.stack(targets, axis=1)

	reg.fit(feats.T, targets.T)

	output_hat = []
	for i in range(horizon, len(t)):
		#z = np.append(u[:,i-horizon:i].dot(filters).reshape(-1,1), [[1]], axis=0)
		z = u[:,i-horizon:i].dot(filters).reshape(-1,1)
		feat = z.dot(z.T).reshape(-1,1).T
		output_hat.append(reg.predict(feat)[0])

	print(len(output_hat))

	#Q = reg.coef_.reshape(11,11)
	Q = reg.coef_.reshape(num_feat, num_feat)

	U, s, V = np.linalg.svd(Q)
	print(s)
	s[np.abs(s) < 1] = 0
	print(s)

	Q = U.dot(np.diag(s)).dot(U.T)


	AA = np.zeros((horizon, horizon))
	AA[:horizon-1, 1:] = np.eye(horizon-1)
	#AA[horizon, horizon] = 1
	
	BB = np.zeros((horizon,1))
	BB[horizon-1,0] = 1

	QQ = filters.dot(Q).dot(filters.T)	
	RR = 100*np.eye(1)
	
	K, _,_, = dlqr(AA,BB,QQ,RR)


	dummy_hidden, dummy_output, _ = simulate(sys, u[:, :horizon], h0=np.array([1, 1]), noisy=False)
	controled_output = np.zeros(output.shape)
	controled_output[0, :horizon] = dummy_output

	uu = np.zeros((1,10000))
	uu[:,0:horizon] = u[:,0:horizon]

	last_hidden = dummy_hidden[:, -1]
	for i in range(horizon, len(t)):
		uu[:,i] = -K.dot(uu[:,i-horizon:i].T) 
		hidden, output_new, _ = simulate(sys, uu[:,i], h0=last_hidden, noisy=False)
		last_hidden = hidden[:, -1]
		controled_output[:,i] = output_new[-1]
		#z = np.append(u[:,i-horizon:i].dot(filters).reshape(-1,1), [[1]], axis=0)
		#output_hat.append(reg.predict(feat)[0])

	free_output = np.zeros(output.shape)
	free_output[0, :horizon] = dummy_output
	last_hidden = dummy_hidden[:, -1]

	for i in range(horizon, len(t)):
		hidden, output_new, _ = simulate(sys, np.array([[0]]), h0=last_hidden, noisy=False)
		last_hidden = hidden[:, -1]
		free_output[:,i] = output_new[-1]

	lqr_output = np.zeros(output.shape)
	lqr_output[0, :horizon] = dummy_output
	last_hidden = dummy_hidden[:, -1]

	K_lqr,_,_, = dlqr(A,B,C.T.dot(C),RR)
	u_lqr = np.zeros((1,10000))
	u_lqr[:,0:horizon] = u[:,0:horizon]

	for i in range(horizon, len(t)):
		u_lqr[:,i] = -K_lqr.dot(last_hidden)
		hidden, output_new, _ = simulate(sys, u_lqr[:,i], h0=last_hidden, noisy=False)
		last_hidden = hidden[:, -1]
		lqr_output[:,i] = output_new[-1]

	"""
	for i in range(horizon, len(t)):

		z = np.append(u[:,i-horizon:i].dot(filters).reshape(-1,1), [[1]], axis=0)
		feat = z.dot(z.T).reshape(-1,)
		output_hat.append(reg.predict(feat)[0])
	"""

	x_roll = np.zeros(500)
	y_roll = np.zeros(500)
	x_roll_lqrpp = np.zeros(500)
	y_roll_lqrpp = np.zeros(500)
	x_ctl = 0

	ht = np.array([[0],[0]])
	ht_lqrpp = np.array([[0],[0]])

	# matplotlib.use('GTKAgg')

	plt.subplot(2,1,1)
	lxl, = plt.plot(x_roll_lqrpp, c='blue', label='LQR++ (ours)')
	lx, = plt.plot(x_roll, c='red', lw=0.5, label='release')
	plt.legend(loc='upper left')
	

	plt.subplot(2,1,2)
	lyl, = plt.plot(y_roll_lqrpp, c='blue')
	ly, = plt.plot(y_roll, c='red', lw=0.5)
	

	t_demo = 0
	while True:
		# plt.clf()
		plt.ion()

		plt.subplot(2,1,1)

		t_demo += 1
		plt.ylabel('Input (x)')

		plt.xlim([0,500])
		plt.ylim([-1,1])

		x_roll[:-1] = x_roll[1:]
		if t_demo % 200 <= 10:
			x_roll[-1] = np.random.randn()
		else:
			x_roll[-1] = 0

		ht = A.dot(ht) + B.dot(x_roll[-1])
		y_roll[:-1] = y_roll[1:]
		y_roll[-1] = C.dot(ht)

		# plt.plot(x_roll, c='red')
		lx.set_ydata(x_roll)

		x_horizon = x_roll_lqrpp[-horizon:]
		x_ctl = -K.dot(x_horizon)

		x_roll_lqrpp[:-1] = x_roll_lqrpp[1:]

		if t_demo % 200 <= 10:
			x_roll_lqrpp[-1] = x_roll[-1]
		else:
			x_roll_lqrpp[-1] = x_ctl

		ht_lqrpp = A.dot(ht_lqrpp) + B.dot(x_roll_lqrpp[-1])
		y_roll_lqrpp[:-1] = y_roll_lqrpp[1:]
		y_roll_lqrpp[-1] = C.dot(ht_lqrpp)
		
		# plt.plot(x_roll_lqrpp, c='blue')
		lxl.set_ydata(x_roll_lqrpp)

		# plt.plot(t[horizon:horizon+500], np.abs(controled_output[0,horizon:horizon+500]),'g:',linewidth=1,label='lqr++(ours)')
		# plt.plot(t[horizon:horizon+500], np.abs(free_output[0,horizon:horizon+500]),'b:',linewidth=1,label='free')
		# plt.plot(t[horizon:horizon+500], np.abs(lqr_output[0,horizon:horizon+500]),'r:',linewidth=1,label='lqr')

		plt.subplot(2,1,2)

		plt.ylabel('Response (y)')

		plt.xlim([0,500])
		plt.ylim([-20,20])

		# plt.plot(y_roll, c='red')
		ly.set_ydata(y_roll)

		plt.xlabel('Time - ' + str(t_demo))

		# plt.plot(y_roll_lqrpp, c='blue')
		lyl.set_ydata(y_roll_lqrpp)

		# plt.draw()
		plt.show()

		plt.pause(0.01)
		# time.sleep(0.05)
