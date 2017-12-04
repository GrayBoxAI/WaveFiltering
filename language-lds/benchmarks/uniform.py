# ultra-minimal toy example (to demonstrate API): just guess the uniform dist
import numpy as np

class GuessUniform:
	def __init__(self, dimension=None):
		if dimension == None:
			self.dimension = 100000 # just guess something, lol
		else:
			self.dimension = dimension
		self.unif = np.ones(self.dimension + 1) / (self.dimension + 1)

	def fit(self, seq):
		'''Train the language model. In: sequence of word indices.'''
		# lol
		pass

	def predict(self):
		'''Test the language model. Out: a probability vector, indexed in the
		same space as the input word indices.'''
		return self.unif

	def step(self, w):
		'''Advance by one state for testing. In: one word index.'''
		pass

if __name__ == '__main__':
	from ModelTester import ModelTester
	ModelTester(GuessUniform(3521)).train().test()
