# ARMA in word vector space
import numpy as np
from sklearn import linear_model

class AutoregressiveModel:
	def __init__(self, dimension=None, window=3, vecs='../corpus/brown/vecs.npy'):
		if dimension == None:
			self.dimension = 100000 # just guess something, lol
		else:
			self.dimension = dimension
		self.vecs = np.load(vecs)
		print self.vecs.shape
		self.vecdim = len(self.vecs[0])
		self.vecs = np.vstack( [self.vecs, np.zeros(self.vecdim)] )
		self.window = window
		self.state = [-1] * self.window

	def fit(self, seq):
		# seq = seq[:100000]

		Vecs = np.array( [self.vecs[w] for w in seq] )

		T = len(seq) - self.window
		X = []
		for t in range(self.window, len(seq)):
			X.append( Vecs[t-self.window:t, :].flat )
			if t % 100000 == 0:
				print 'featurize', t

		X = np.vstack(X)

		print 'now running linreg...'
		self.clf = linear_model.Ridge(alpha=100)
		self.clf.fit(X, Vecs[self.window:])
		print 'linreg done'

	def predict(self):
		feat = np.hstack( [self.vecs[w] for w in self.state] )
		vec = self.clf.predict(feat.reshape(1,-1))
		# vec = self.vecs[ self.state[-1] ]
		vec /= np.linalg.norm(vec) + 0.001

		out = self.vecs.dot(vec.T)
		# print(out)
		out = np.exp(out * 10)
		return out / np.sum(out)

	def step(self, w):
		self.state = self.state[1:] + [w]

if __name__ == '__main__':
	from ModelTester import ModelTester
	ModelTester(AutoregressiveModel(3521, 4, '../corpus/penn/vecs.npy')).train().test(ignore=10)
