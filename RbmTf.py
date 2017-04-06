# Reference https://github.com/Cospel/rbm-ae-tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time

class RBM:
    def __init__(self, numVis, numHid, names, learningRate=0.5, weightDecay=0.0001, momentum=0.9):
        with tf.device('/gpu:0'):
            self.numVis = numVis
            self.numHid = numHid
            self.names = names
            self.learningRate = learningRate
            self.weightDecay = weightDecay
            self.momentum = momentum

            # Weight for save and restore Tensorflow
            self.weights = self.initialWeight()

            # placeholders
            self.x = tf.placeholder(tf.float32, [None, self.numVis])
            self.w = tf.Variable(tf.random_normal([self.numVis, self.numHid], mean=0.0, stddev=0.01, dtype=tf.float32))
            self.dW = tf.Variable(tf.zeros([self.numVis, self.numHid], tf.float32))
            self.vb = tf.Variable(tf.zeros([self.numVis], tf.float32))
            self.dVb = tf.Variable(tf.zeros([self.numVis], tf.float32))
            self.hb = tf.Variable(tf.zeros([self.numHid], tf.float32))
            self.dHb = tf.Variable(tf.zeros([self.numHid], tf.float32))

            # Initial Variables
            # The weights are initialized zero-mean Gaussian with a standard deviation of about 0.01.
            # self.newW = np.zeros([self.numVis, self.numHid], np.float32)
            # self.newVb = np.zeros([self.numVis], np.float32)
            # self.newHb = np.zeros([self.numHid], np.float32)
            # self.outW = np.random.normal(0.0, 0.01, [self.numVis, self.numHid])
            # self.outVb = np.zeros([self.numHid], np.float32)
            # self.outHb = np.zeros([self.numHid], np.float32)
            self.wShow = np.zeros([self.numVis, self.numHid], np.float32)

            # Gibbs sample = 1.
            # RBM is generative model, who tries to encode in weights the understanding of data.
            # RBMs typically learn better models if more steps of alternating Gibbs sampling are used.
            # 1. set visible state to training sample(x) and compute hidden state(h0) of data
            #    then we have binary units of hidden state computed. It is very important to make these
            #    hidden states binary, rather than using the probabilities themselves. (see Hinton paper)
            self.h0Prob = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hb)
            self.h0 = self.sampleProb(self.h0Prob)
            # 2. compute new visible state of reconstruction based on computed hidden state reconstruction.
            #    However, it is common to use the probability, instead of sampling a binary value.
            #    So this can be binary or probability(so i choose to not use sampled probability)
            self.v1 = tf.nn.sigmoid(tf.matmul(self.h0Prob, tf.transpose(self.w)) + self.vb)
            # 3. compute new hidden state of reconstruction based on computed visible reconstruction
            #    When hidden units are being driven by reconstructions, always use probabilities without sampling.
            self.h1 = tf.nn.sigmoid(tf.matmul(self.v1, self.w) + self.hb)

            # Compute gradients
            self.wPositiveGrad = tf.matmul(tf.transpose(self.x), self.h0)
            self.wNegativeGrad = tf.matmul(tf.transpose(self.v1), self.h1)

            # Stochastic steepest ascent because we need to maximize log likelihood of p(visible)
            # dlog(p)/dlog(w) = (visible * hidden)_data - (visible * hidden)_reconstruction
            # self.updateW = self.w + self.learningRate * (self.w_positive_grad - self.w_negative_grad) / tf.to_float(tf.shape(self.x)[0])
            # self.updateVb = self.vb + self.learningRate * tf.reduce_mean(self.x - self.v1, 0)
            # self.updateHb = self.hb + self.learningRate * tf.reduce_mean(self.h0prob - self.h1, 0)
            self.dW = self.momentum * self.dW + self.learningRate * ((self.wPositiveGrad - self.wNegativeGrad) / tf.to_float(tf.shape(self.x)[0]) - self.weightDecay * self.w)
            self.dVb = (self.learningRate * tf.reduce_mean(self.x - self.v1, 0)) + (self.momentum * self.dVb * self.learningRate / tf.to_float(tf.shape(self.x)[0]))
            self.dHb = (self.learningRate * tf.reduce_mean(self.h0Prob - self.h1, 0)) + (self.momentum * self.dHb * self.learningRate / tf.to_float(tf.shape(self.x)[0]))

            # Update weight
            self.w += self.dW
            self.vb += self.dVb
            self.hb += self.dHb

            # sampling functions
            self.hSample = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hb)
            self.vSample = tf.nn.sigmoid(tf.matmul(self.hSample, tf.transpose(self.w)) + self.vb)

            # cost
            self.errSum = tf.reduce_mean(tf.square(self.x - self.vSample))

            # Initial session
            self.init = tf.initialize_all_variables()
            # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            self.sess = tf.Session()
            self.sess.run(self.init)

    def initialWeight(self):
        # These weights are only for storing and loading model for Tensorflow Saver.
        allWeights = dict()
        allWeights['w'] = tf.Variable(tf.random_normal([self.numVis, self.numHid], stddev=0.01, dtype=tf.float32),name=self.names[0])
        allWeights['vb'] = tf.Variable(tf.zeros([self.numVis], dtype=tf.float32), name=self.names[1])
        allWeights['hb'] = tf.Variable(tf.random_uniform([self.numHid], dtype=tf.float32), name=self.names[2])
        return allWeights

    def transform(self, x):
        return self.sess.run(self.hSample, feed_dict={self.x: x})

    def sampleProb(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def fit(self, x, epoch):
        # Train
        for _ in range(epoch):
            tmpError, self.wShow = self.sess.run([self.errSum, self.w], feed_dict={self.x: x})
            print self.wShow

if __name__ == '__main__':

    # Train Data
    data = np.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])

    rbm = RBM (6, 2, ["w1", "vb1", "hb1"])
    rbm.fit(data, 1)