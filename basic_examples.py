from HMM import *
import numpy as np

if __name__ == "__main__":

	### example from wikipedia: https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
	health_transition = {
		'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
		'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
	}
	health_emission = {
		'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
		'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
	}
	health_initial = {'Healthy': 0.6, 'Fever': 0.4}

	Health = HMM(health_initial, health_transition, health_emission)
	likelihood = []

	#print("the most likely hidden state sequence: {}".format(
	#	Health.viterbi(['normal', 'cold', 'dizzy'], likelihood=likelihood)))
	#print("That sequence has a probability {}".format(likelihood[0]))
	
	### example from wikipedia for the Baum-Welch algorithm: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Example
	chicken_transition = {
		1 : {1 : 0.5, 2 : 0.5},
		2 : {1 : 0.3, 2 : 0.7}
	}
	chicken_emission = {
		1 : {'N' : 0.3, 'E' : 0.7},
		2 : {'N' : 0.8, 'E' : 0.2}
	}
	chicken_initial = {1:0.2, 2:0.8}

	Chicken = HMM(chicken_initial, chicken_transition, chicken_emission)
	Chicken.baum_welch('NN')