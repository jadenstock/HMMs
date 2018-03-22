import numpy as np

class HMM():
	def __init__(self, transition_probs, emission_probs):
		self.transition_probs = transition_probs #2-dim
		self.emission_probs = emission_probs #2-dim 

	def likelihood(self, seq, log=True):
		"""simple implementation of forward algorithm"""
		alpha = [dict([(s,0) for s in self.transition_probs]) for i in xrange(len(seq))]
		alpha[0]  = dict([(i, np.log(self.emission_probs[i][seq[0]])) for i in self.emission_probs])
		for i in xrange(1, len(seq)):
			for j in self.transition_probs:
				alpha[i][j] = np.log(sum([np.exp(alpha[i-1][k])*self.transition_probs[k][j]*self.emission_probs[j][seq[i]] for k in self.transition_probs]))
		return sum([np.exp(p) for p in alpha[-1].values()]) #return the sum of the last row

	def viterbi(self, seq):
		"""implementation of the viterbi algorithm for inference"""
		return 0

	def baum_weltch(self, seq):
		"""implementation of the Baum-Weltch or Forward-Backward algorithm for inference of the transition probabilities"""
		return 0

#if __name__ == '__main__':
#	coin = HMM({1:{1:.8, 2:.2}, 2:{1:.5, 2:.5}}, {1:{1:.5, 2:.5}, 2:{1:.9, 2:.1}})
#	print(coin.likelihood([1,2,1,2,2,1,2,2,1,1,2,1,2,2,1,2,1,2,1,1,1]))