import matplotlib.pyplot as plt 

iteration = [1,2,3,4,5,6,7,8]
dagger_reward_mean= [499.61,1013.17,1814.99,7031.43,7660.73,9123.67,9667.56,10327.11]
dagger_reward_sd = [178.60,422,944.57,2750.69,2819.39,2730.62,1660.87,59.24]

plt.plot(iteration,dagger_reward_mean, label ='DAgger')
plt.axhline(y=10395.79, color='k', label='Expert Policy')
plt.axhline(y=8301.86, color='g', label='Behaviorial Cloning')
plt.errorbar(iteration, dagger_reward_mean, yerr=dagger_reward_sd , fmt='o', color ='r')
plt.xlabel('DAgger Iteration')
plt.ylabel('Mean Reward')
plt.title('Humanoid Reward using DAgger')
plt.legend()
plt.savefig('dagger.png')
plt.figure()



num_rollouts = [10,25,50,75,100]
cloning_mean = [539.01, 732.54, 2415.42, 1348.24, 7088]
cloning_sd = [125.35,276.81,2415.43,659.09,3382.36]

plt.plot(num_rollouts,cloning_mean)
plt.errorbar(num_rollouts,cloning_mean, yerr=cloning_sd , fmt='o', color ='r')
plt.xlabel('Rollouts used for Training')
plt.ylabel('Mean Reward')
plt.title("Humanoid Reward vs Number of Training Demonstrations")
plt.legend()
plt.savefig('cloning.png')

