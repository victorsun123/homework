# #!/usr/bin/env python

# """
# Code to load an expert policy and generate roll-out data for behavioral cloning.
# Example usage:
#     python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
#             --num_rollouts 20

# Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
# """

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from model import Model



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rollout_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--log_dir', type=str)
    args = parser.parse_args()

    print('loading rollout data')
    with open(args.rollout_data, 'rb') as f:
        data = pickle.loads(f.read())
    observation_data = np.array(data['observations'])
    action_data = np.array(data['actions'])

    print('training supervised model')
    import gym
    env = gym.make(args.envname)
    cloning_policy = Model(env)
    cloning_policy.train(observation_data,action_data,args.num_epochs)

    print('running policy')
    with tf.Session():
        tf_util.initialize()
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = cloning_policy.predict(obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()


   