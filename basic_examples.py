from HMM import *
import numpy as np

if __name__ == "__main__":
	coin_transition = {1:{1:.8, 2:.2}, 2:{1:.5, 2:.5}}
	coin_emission = {1:{1:.5, 2:.5}, 2:{1:.9, 2:.1}}
	coin_initial = {1:.5, 2:.5}
	coin = HMM(coin_transition, coin_emission, coin_initial)

	#print(coin.likelihood([1,2,1,2,2,1,2,2,1,1,2,1,2,2,1,2,1,2,1,1,1]))
	#print(coin.viterbi([1,2,1,2,2,1,2,2,1,1,2,1,2,2,1,2,1,2,1,1,1]))

	### example from wikipedia
	health_transition = {
		'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
		'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
	}
	health_emission = {
		'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
		'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
	}
	health_initial = {'Healthy': 0.6, 'Fever': 0.4}

	Health = HMM(health_transition, health_emission, health_initial)

	prob = []
	print(Health.viterbi(['normal', 'cold', 'dizzy'], likelihood=prob))
	print(np.exp(prob[0]))