#compare humanoid behavior cloning relative to expert vs ant behavior cloning relative to expert. 
# python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --save --num_rollouts 100  > humanoid_expert.out
# python behavioral_cloning.py expert_data/Humanoid-v2_100.pkl Humanoid-v2 --save --num_rollouts 100 --num_epochs 50  > runs/humanoid_cloning.out

# python run_expert.py experts/Ant-v2.pkl Ant-v2 --save --num_rollouts 100  > ant_expert.out
# python behavioral_cloning.py expert_data/Ant-v2_100.pkl Ant-v2 --save --num_rollouts 100 --num_epochs 50  > runs/humanoid_cloning.out






#compare number of rollouts used in training data for behavior cloning on Humanoid as changing hyperparameter
# python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --save --num_rollouts 10 > runs/humanoid_expert_10.out
# python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --save --num_rollouts 25  > runs/humnaoid_expert_25.out
# python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --save --num_rollouts 50  > runs/humanoid_expert_50.out
# python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --save --num_rollouts 75  > runs/humanoid_expert_75.out



python behavior_cloning.py expert_data/Humanoid-v2_10.pkl Humanoid-v2 --num_rollouts 20 --num_epochs 50  > runs/humanoid_cloning_10.out
python behavior_cloning.py expert_data/Humanoid-v2_25.pkl Humanoid-v2 --num_rollouts 20 --num_epochs 50  > runs/humnaoid_cloning_25.out
python behavior_cloning.py expert_data/Humanoid-v2_50.pkl Humanoid-v2 --num_rollouts 20 --num_epochs 50  > runs/humanoid_cloning_50.out
python behavior_cloning.py expert_data/Humanoid-v2_75.pkl Humanoid-v2 --num_rollouts 20 --num_epochs 50  > runs/humanoid_cloning_75.out
python behavior_cloning.py expert_data/Humanoid-v2_100.pkl Humanoid-v2 --num_rollouts 20 --num_epochs 50  > runs/humanoid_cloning_100.out

#run dagger for 10 iterations with each iteration updating with 20 rollouts of data on Humanoid
#python dagger.py experts/Humanoid-v2.pkl expert_data/Humanoid-v2.pkl Humanoid-v2 --dagger_iterations 10 --num_rollouts 20 --num_epochs 50 > runs/humanoid_dagger.out

