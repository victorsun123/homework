import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(data, name, colors):
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

	plt.plot(range(100, len(data)), mean, color = colors[0], label = 'EpRewMean_' + name)
	plt.plot(range(100, len(data)), best, color = colors[1], label = 'EpRewMeanBest_' + name)



with open("874e145c-f423-49f0-8baa-7d118d7bc2f5.pkl", 'rb') as f:
	vanilla = pickle.load(f)
plot(vanilla, "vanilla_DQN", ["red", "blue"])

with open("932235fb-5da6-44f8-808b-9fe051bc0bf3.pkl", 'rb') as f:
	double = pickle.load(f)
plot(double, "double_DQN", ["magenta", "cyan"])

plt.xlabel("Episode Number")
plt.ylabel("Episode Reward")
plt.legend()
plt.savefig("vanilla_v_double.png")
plt.figure()