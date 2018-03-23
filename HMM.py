import numpy as np

class HMM():
	"""
	This class represents a simplem HMM.
	"""
	def __init__(self, initial_probabilities, transition_probs, emission_probs):
		# transition probs is a two level dictionary such that transition_probs[i][j] is 
		# the probability of transitioning from state i to state j.
		self.transition_probs = transition_probs
		# emission probs is a two level dictionary such that emission_probs[s][o] is 
		# the probability of emitting output o in state i.
		self.emission_probs = emission_probs

		# a dictionary such that initial_probabilities[i] is the probability of starting in state i
		self.initial_probabilities = initial_probabilities

		#store the log of the transition probs to avoid recomputation
		self._log_transition_probs = self._compute_log_probs(transition_probs)

		#store the log of the emission probs to avoid recomputation
		self._log_emission_probs = self._compute_log_probs(emission_probs)

		if not self._valid_hmm(): raise ValueError("this is not a valid HMM")


	def likelihood(self, seq, log=True):
		"""
		simple implementation of forward algorithm.
		Does calculations in log space to avoid underflow issues.
		
		INPUT: A sequence of observables (key in the emission_probs dict).
		OPTIONS: the log flag is set by default to be True. If the flag is false then this returns
			the regular probability of seeing the output sequence which may be very small.
		OUTPUT: the log  probability or probability of this sequence of observables in the HMM.
		"""
		alpha = [dict([(s, 0) for s in self.transition_probs]) for _ in xrange(len(seq))]
		alpha[0]  = dict([(i, np.log(self.initial_probabilities[i]) + self._log_emission_probs[i][seq[0]]) for i in self.emission_probs])
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


	def viterbi(self, seq, likelihood=None):
		"""
		implementation of the viterbi algorithm for inference.
		INPUT: a sequence of opservables (values in the emission_prob dict).
		OPTIONAL: you may pass in a likelihood list which will be filled with the probability
			of the most likely hidden state sequence
		OUTPUT: The sequence of hidden states that is most likely given then output sequence.
		"""
		T = [dict([(s, 0) for s in self.transition_probs]) for _ in xrange(len(seq))]
		T[0] = dict([(i, np.log(self.initial_probabilities[i]) + self._log_emission_probs[i][seq[0]]) for i in self.emission_probs])
		T_bp = [dict([(s, 0) for s in self.transition_probs]) for _ in xrange(len(seq))]
		T_bp[0] = dict([(i, i) for i in self.emission_probs])

		for i in xrange(1, len(seq)):
			for j in self.transition_probs:

				T_bp[i][j] = max(self.transition_probs,
					key = lambda k : T[i-1][k] + self._log_transition_probs[k][j] + self._log_emission_probs[j][seq[i]])

				prev_state = T_bp[i][j]
				T[i][j] = T[i-1][prev_state] + self._log_transition_probs[prev_state][j] + self._log_emission_probs[j][seq[i]]

		if likelihood != None:
			likelihood.append(np.exp(max(T[-1].values())))

		# construct the most likely sequence
		state_sequence = [None for i in xrange(len(seq))]
		state_sequence[-1] = max(self.transition_probs,
			key = lambda k : T[-1][k])

		for i in xrange(len(seq)-1):
			i = len(seq)-2-i #go in reverse order
			state_sequence[i] = T_bp[i+1][state_sequence[i+1]]
		
		return state_sequence

	def baum_welch(self, seq, iterations=1):
		"""
		implementation of the Baum-Weltch or Forward-Backward algorithm for inference of the transition probabilities.
		Computes the transition and emission probabilities which maxmimize the likelihood of seeing this observation sequence.
		INPUT: and observation which is a sequence of output variables
		EFFECT: Updates self.initial_probabilities, self.transition_probabilities, and self.emission_probabilities
		OUTPUT: NONE
		"""
		for _ in xrange(iterations):
			self._baum_welch_iteration(seq)

	def _baum_welch_iteration(self, seq):
		""" perform a single round of baum_welch algorithm"""

		#compute the forward probabilities
		forward_probabilities = [dict([(s,0) for s in self.transition_probs]) for _ in xrange(len(seq))]
		forward_probabilities[0] = dict([(s, np.log(self.initial_probabilities[s]) + self._log_emission_probs[s][seq[0]]) for s in self.transition_probs])

		for i in xrange(1, len(seq)):
			for j in self.transition_probs:
				forward_probabilities[i][j] = self._log_sum_exp([
					forward_probabilities[i-1][k] +
					self._log_transition_probs[k][j] +
					#is this seq[i] right?
					self._log_emission_probs[j][seq[i]] for k in self.transition_probs])

		#compute the backward probabilities
		backward_probabilities = [dict([(s, 0.0) for s in self.transition_probs]) for _ in xrange(len(seq))]
		for i in xrange(len(seq) - 1):
			i = len(seq) - 2 - i #move backwards
			for j in self.transition_probs:
				backward_probabilities[i][j] = self._log_sum_exp([
					backward_probabilities[i+1][k] +
					self._log_transition_probs[j][k] + 
					#should this be seq[i+1] for some reason?
					self._log_emission_probs[j][seq[i]] for k in self.transition_probs])

		# compute the probability of being in a given state
		gamma = [dict([(s,0) for s in self.transition_probs]) for t in xrange(len(seq))]
		for t in xrange(len(seq)):
			norm = sum([np.exp(forward_probabilities[t][k]+backward_probabilities[t][k]) for k in self.transition_probs])
			for j in self.transition_probs:
				gamma[t][j] = np.exp(forward_probabilities[t][j]+backward_probabilities[t][j])/norm

		# compute the probability of transitioning between states
		zeta = [dict([(s, dict([(s,0) for s in self.transition_probs])) for s in self.transition_probs]) for i in xrange(len(seq) - 1)]
		for t in xrange(len(seq) - 1):
			norm = sum([sum([
				np.exp(forward_probabilities[t][i] + self._log_transition_probs[i][j] 
					+ backward_probabilities[t][j] + self._log_emission_probs[j][seq[t+1]])
			for j in self.transition_probs]) for i in self.transition_probs])
			
			for i in self.transition_probs:
				for j in self.transition_probs:
					zeta[t][i][j] = np.exp(forward_probabilities[t][i] + self._log_transition_probs[i][j] 
					+ backward_probabilities[t][j] + self._log_emission_probs[j][seq[t+1]]) / norm

		#finally update our probabilites
		self.initial_probabilities = dict([(s, gamma[1][s]) for s in self.transition_probs])
		
		for i in self.transition_probs:
			for j in self.transition_probs[i]:
				self.transition_probs[i][j] = sum([zeta[t][i][j] for t in xrange(len(seq) - 1)])/sum([gamma[t][i] for t in xrange(len(seq) - 1)])

		for i in self.emission_probs:
			for o in self.emission_probs[i]:
				self.emission_probs[i][o] = sum([(seq[t] == o)*gamma[t][i] for t in xrange(len(seq))])/sum([gamma[t][i] for t in xrange(len(seq))])

		print("A:{}".format(self.transition_probs))
		print("B:{}".format(self.emission_probs))
		self._log_transition_probs = self._compute_log_probs(self.transition_probs)
		self._log_emission_probs = self._compute_log_probs(self.emission_probs)

	def _log_sum_exp(self, arr):
		"""
		use the log-sum-exp trick to comput the log of a sum of values. The log-sum-exp trick is the
		fact that log(sum_{i}exp(x_{i})) = a + log(sum_{i}exp(x_{i}-a)) for any a, in particular for max[x_{i}].
		This trick can avoid some underflow issues by avoiding computing exp(x_{i}) for all x_{i}, which may retun 0.

		INPUT: an itterable holding number values
		OUTPUT: log(sum([exp(x) for x in arr]))
		"""
		a = np.max(arr)
		print(arr)
		return (a + np.log(np.sum([np.exp(x-a) for x in arr])))

	def _compute_log_probs(self, prob_matrix):
		"""just takes the log of each element in the matrix"""
		log_prob_matrix = {}
		for i in prob_matrix:
			log_prob_matrix[i] = {}
			for j in prob_matrix[i]:
				log_prob_matrix[i][j] = np.log(prob_matrix[i][j])
		return log_prob_matrix
	
	def _valid_hmm(self):
		"""Checks to make sure the HMM provided is valid."""
		if (self.transition_probs != None) and (self.emission_probs != None):
			valid = True
			valid &= (set(self.transition_probs.keys()) == set(self.emission_probs.keys()))
			for k in self.transition_probs:
				#only consider fully connected HMMs for now
				valid &= set(self.transition_probs) == set(self.transition_probs[k])
			return valid
		else:
			return True
