import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(data, save_name):
	print(len(data))
	mean = []
	best = []
	for i in range(100, len(data)):
		cur_mean = np.mean(np.array(data[i-100:i]))
		mean.append(cur_mean)
		if not best or best[-1] < cur_mean:
			best.append(cur_mean)
		else:
			best.append(best[-1])

	plt.plot(range(100, len(data)), mean, color = 'red', label = 'EpRewMean')
	plt.plot(range(100, len(data)), best, color = 'blue', label = 'EpRewMeanBest')
	plt.xlabel("Episode Number")
	plt.ylabel("Episode Reward")
	plt.legend()
	plt.savefig(save_name)
	plt.figure()


with open("874e145c-f423-49f0-8baa-7d118d7bc2f5.pkl", 'rb') as f:
	vanilla = pickle.load(f)
plot(vanilla, "vanilla_DQN.png")

with open("932235fb-5da6-44f8-808b-9fe051bc0bf3.pkl", 'rb') as f:
	double = pickle.load(f)
plot(double, "double_DQN.png")
