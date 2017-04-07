# Reference https://github.com/Cospel/rbm-ae-tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class RBM:
    def __init__(self, numVis, numHid, names, path, typeRun="/cpu:0", learningRate=0.5, weightDecay=0.0001, momentum=0.9):
        with tf.device(typeRun):
            self.numVis = numVis
            self.numHid = numHid
            self.names = names
            self.learningRate = learningRate
            self.weightDecay = weightDecay
            self.momentum = momentum

            # Weight for save and restore Tensorflow
            self.weights = self.initialWeight()

            # Placeholder and Variables
            self.x = tf.placeholder(tf.float32, [None, self.numVis])
            self.w = tf.Variable(tf.random_normal([self.numVis, self.numHid], mean=0.0, stddev=0.01, dtype=tf.float32))
            self.vb = tf.Variable(tf.zeros([self.numVis], tf.float32))
            self.hb = tf.Variable(tf.zeros([self.numHid], tf.float32))

            # For update
            self.dW = tf.Variable(tf.zeros([self.numVis, self.numHid], tf.float32))
            self.dVb = tf.Variable(tf.zeros([self.numVis], tf.float32))
            self.dHb = tf.Variable(tf.zeros([self.numHid], tf.float32))

            # For save model
            self.saveDW = tf.Variable(tf.zeros([self.numVis, self.numHid], tf.float32))
            self.saveDVb = tf.Variable(tf.zeros([self.numVis], tf.float32))
            self.saveDHb = tf.Variable(tf.zeros([self.numHid], tf.float32))

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
            # self.dW = self.w + self.learningRate * (self.wPositiveGrad - self.wNegativeGrad) / tf.to_float(tf.shape(self.x)[0])
            # self.dVb = self.vb + self.learningRate * tf.reduce_mean(self.x - self.v1, 0)
            # self.dHb = self.hb + self.learningRate * tf.reduce_mean(self.h0Prob - self.h1, 0)
            self.dW = self.momentum * self.saveDW + self.learningRate * ((self.wPositiveGrad - self.wNegativeGrad) / tf.to_float(tf.shape(self.x)[0]) - self.weightDecay * self.w)
            self.dVb = (self.learningRate * tf.reduce_mean(self.x - self.v1, 0)) + (self.momentum * self.saveDVb * self.learningRate / tf.to_float(tf.shape(self.x)[0]))
            self.dHb = (self.learningRate * tf.reduce_mean(self.h0Prob - self.h1, 0)) + (self.momentum * self.saveDHb * self.learningRate / tf.to_float(tf.shape(self.x)[0]))

            # Update weight
            self.updateAll = [self.w.assign_add(self.dW), self.vb.assign_add(self.dVb), self.hb.assign_add(self.dHb),
                              self.saveDW.assign(self.dW), self.saveDVb.assign(self.dVb), self.saveDHb.assign(self.dHb)]
            # self.w += self.dW
            # self.vb += self.dVb
            # self.hb += self.dHb

            # sampling functions
            self.hSample = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hb)
            self.vSample = tf.nn.sigmoid(tf.matmul(self.hSample, tf.transpose(self.w)) + self.vb)

            # Cost
            self.errSum = tf.reduce_mean(tf.square(self.x - self.vSample))
            tf.summary.scalar("Error Sum", self.errSum)

            # Initial session
            self.init = tf.global_variables_initializer()
            # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            # self.sess = tf.Session()
            self.sess.run(self.init)

            # For Tensorboard
            self.merge = tf.summary.merge_all()
            self.write = tf.summary.FileWriter(path, self.sess.graph)

    def initialWeight(self):
        # These weights are only for storing and loading model for Tensorflow Saver.
        allWeights = dict()
        allWeights["w"] = tf.Variable(tf.random_normal([self.numVis, self.numHid], stddev=0.01, dtype=tf.float32),name=self.names[0])
        allWeights["vb"] = tf.Variable(tf.zeros([self.numVis], dtype=tf.float32), name=self.names[1])
        allWeights["hb"] = tf.Variable(tf.zeros([self.numHid], dtype=tf.float32), name=self.names[2])
        allWeights["dW"] = tf.Variable(tf.zeros([self.numVis, self.numHid], dtype=tf.float32), name=self.names[3])
        allWeights["dVb"] = tf.Variable(tf.zeros([self.numVis], dtype=tf.float32), name=self.names[4])
        allWeights["dHb"] = tf.Variable(tf.zeros([self.numHid], dtype=tf.float32), name=self.names[5])
        return allWeights

    def transform(self, x):
        return self.sess.run(self.hSample, feed_dict={self.x: x})

    def sampleProb(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def sampleThreshold(self, probs, cutoff):
        return tf.nn.relu(tf.sign(probs - cutoff))

    def fit(self, x, epoch):
        for _ in range(epoch):
            summary, tmp1, tmp2 = self.sess.run([self.merge, self.errSum, self.updateAll], feed_dict={self.x: x})
            self.write.add_summary(summary, _)

    def predictH(self, x):
        return self.sess.run(self.hSample, {self.x: x})

    def predictV(self, x):
        return self.sess.run(self.vSample, {self.x: x})

    def predictVWithThreshold(self, x, cutoff):
        vSampleThreshold = self.sampleThreshold(self.vSample, cutoff)
        return self.sess.run(vSampleThreshold, {self.x: x})

    def getWeight(self):
        return self.sess.run(self.w)

    def restoreWeights(self, path):
        saver = tf.train.Saver({self.names[0]: self.weights["w"],
                                self.names[1]: self.weights["vb"],
                                self.names[2]: self.weights["hb"],
                                self.names[3]: self.weights["dW"],
                                self.names[4]: self.weights["dVb"],
                                self.names[5]: self.weights["dHb"]})
        saver.restore(self.sess, path)
        self.sess.run([self.w.assign(self.weights["w"]), self.vb.assign(self.weights["vb"]), self.hb.assign(self.weights["hb"]),
                       self.saveDW.assign(self.weights["dW"]), self.saveDVb.assign(self.weights["dVb"]), self.saveDHb.assign(self.weights["dHb"])])


    def saveWeights(self, path):
        self.sess.run([self.weights["w"].assign(self.w), self.weights["vb"].assign(self.vb), self.weights["hb"].assign(self.hb),
                       self.weights["dW"].assign(self.saveDW), self.weights["dVb"].assign(self.saveDVb), self.weights["dHb"].assign(self.saveDHb)])
        saver = tf.train.Saver({self.names[0]: self.weights["w"],
                                self.names[1]: self.weights["vb"],
                                self.names[2]: self.weights["hb"],
                                self.names[3]: self.weights["dW"],
                                self.names[4]: self.weights["dVb"],
                                self.names[5]: self.weights["dHb"]})
        saver.save(self.sess, path)

if __name__ == '__main__':

    # Train Data
    data = np.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])

    rbm = RBM (6, 2, ["w", "vb", "hb", "dW", "dVb", "dHb"], "./logs")
    np.set_printoptions(suppress=True)
    rbm.fit(data, 100)
    rbm.saveWeights("./model.ckpt")
    # rbm.restoreWeights("./model.ckpt")
    print rbm.predictH(np.array([[0, 0, 0, 1, 1, 0]]))
    print rbm.predictV(np.array([[0, 0, 0, 1, 1, 0]]))

    # [[0.00000313  0.99999726]]
    # [[0.0000861   0.00022533  0.99443108  0.99975079  0.9840337   0.00442834]]