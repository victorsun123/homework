import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class Model:

    def __init__(self, env):
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        self.input = tf.placeholder(dtype=tf.float32, shape =[None,input_dim])
        self.output = tf.placeholder(dtype=tf.float32, shape =[None, output_dim])

        self.layer1 = tf.layers.dense(inputs = self.input, units = 128,activation = tf.nn.relu)
        self.layer2 = tf.layers.dense(inputs = self.layer1, units = 128,activation = tf.nn.relu)
        self.layer3 = tf.layers.dense(inputs = self.layer2, units = 128,activation = tf.nn.relu)
        self.output_pred = tf.layers.dense(inputs = self.layer3, units = output_dim)
        self.mse = tf.losses.mean_squared_error(self.output_pred,self.output)
        self.opt = tf.train.AdamOptimizer().minimize(self.mse)
        self.sess = tf.Session()


    def train(self, observation_data, action_data, epochs, batch_size=32):
        # run training
        num_samples = observation_data.shape[0]
        train_size = int(.9 * num_samples)
        observation_data, action_data = shuffle(observation_data, action_data)
        action_data = np.reshape(action_data,[action_data.shape[0],-1])

        x_train = observation_data[:train_size]
        x_val = observation_data[train_size:]
        y_train = action_data[:train_size]
        y_val = action_data[train_size:]
        

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(epochs):

            for i in range(0, train_size, batch_size):
                end = min(i+batch_size,train_size)
                x_batch = x_train[i:end]
                y_batch = y_train[i:end]

                # run the optimizer and get the mse
                self.sess.run(self.opt, feed_dict={self.input: x_batch, self.output: y_batch})
            
            mse_train = self.sess.run(self.mse, feed_dict={self.input: x_batch, self.output: y_batch})
            mse_val = self.sess.run(self.mse, feed_dict={self.input: x_val, self.output: y_val})
            print('Epoch: {0:d} Training loss: {1:.3f}'.format(epoch+1, mse_train))
            print('Epoch: {0:d} Validation loss: {1:.3f}'.format(epoch+1, mse_val))

    def predict(self, observation):
        observation = np.expand_dims(observation,axis=0)
        action = self.sess.run(self.output_pred, feed_dict={self.input: observation})
        return action