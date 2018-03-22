import numpy as np

class HMM():
	"""
	This class represents a simplem HMM.
	"""
	def __init__(self, transition_probs, emission_probs):
		# transition probs is a two level dictionary such that transition_probs[i][j] is 
		# the probability of transitioning from state i to state j.
		self.transition_probs = transition_probs
		# emission probs is a two level dictionary such that emission_probs[s][o] is 
		# the probability of emitting output o in state i.
		self.emission_probs = emission_probs

		#store the log of the transition probs to avoid recomputation
		self._log_transition_probs = transition_probs
		for i in transition_probs:
			for j in transition_probs[i]:
				self._log_transition_probs[i][j] = np.log(transition_probs[i][j])

		#store the log of the emission probs to avoid recomputation
		self._log_emission_probs = emission_probs
		for i in emission_probs:
			for o in emission_probs[i]:
				self._log_emission_probs[i][o] = np.log(emission_probs[i][o])

		if not self._valid_hmm(): raise ValueError("this is not a valid HMM")

	def likelihood(self, seq, log=True):
		"""simple implementation of forward algorithm.
		Does calculations in log space to avoid underflow issues."""
		alpha = [dict([(s, 0) for s in self.transition_probs]) for _ in xrange(len(seq))]
		alpha[0]  = dict([(i, self._log_emission_probs[i][seq[0]]) for i in self.emission_probs])
		for i in xrange(1, len(seq)):
			for j in self.transition_probs:
				alpha[i][j] = self._log_sum_exp([
					alpha[i-1][k] +
					self._log_transition_probs[k][j] +
					self._log_emission_probs[j][seq[i]] for k in self.transition_probs])

		if log:
			return self._log_sum_exp(alpha[-1].values())
		else:
			return sum([np.exp(p) for p in alpha[-1].values()]) #return the sum of the last row

	def viterbi(self, seq):
		"""implementation of the viterbi algorithm for inference"""
		return 0

	def baum_welch(self, seq):
		"""implementation of the Baum-Weltch or Forward-Backward algorithm for inference of the transition probabilities"""
		return 0

	def _log_sum_exp(self, arr):
		"""
		use the log-sum-exp trick to comput the log of a sum of values. The log-sum-exp trick is the
		fact that log(sum_{i}exp(x_{i})) = a + log(sum_{i}exp(x_{i}-a)) for any a, in particular for max[x_{i}].
		This trick can avoid some underflow issues by avoiding computing exp(x_{i}) for all x_{i}, which may retun 0.

		INPUT: an itterable holding number values
		OUTPUT: log(sum([exp(x) for x in arr]))
		"""
		a = np.max(arr)
		return (a + np.log(np.sum([np.exp(x-a) for x in arr])))
		
	def _valid_hmm(self):
		if (self.transition_probs != None) and (self.emission_probs != None):
			return (set(self.transition_probs.keys()) == set(self.emission_probs.keys()))
		else:
			return True