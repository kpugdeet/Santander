import tensorflow as tf
import time

class RBM:
    def __init__(self):
        with tf.device('/gpu:0'):
            # Initial all variable and equation graph
            self.x = tf.placeholder(tf.float32, [None, 3])
            self.y = tf.placeholder(tf.float32, [None, 2])
            self.W = tf.Variable(tf.random_normal([3, 2]), trainable=True)
            self.y_ = tf.sigmoid(tf.matmul(self.x, self.W))
            self.costFunction = tf.reduce_sum(tf.scalar_mul(0.5, tf.square(tf.subtract(self.y, self.y_))))
            self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.costFunction)
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            # Initial session
            self.init = tf.global_variables_initializer()
            # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            self.sess = tf.Session()
            self.sess.run(self.init)

    def fit(self, x, y):
        # Train
        for _ in range(1000):
            self.sess.run(self.optimizer, feed_dict={self.x: x, self.y: y})
            self.sess.run(self.costFunction, feed_dict={self.x: x, self.y: y})

        # Test the model
        print("Accuracy: {0}".format(self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y})))
        tf.summary.FileWriter("./logs", self.sess.graph)

    def predict(self, x):
        return self.sess.run(tf.argmax(self.y_, 1), feed_dict={self.x: x})

if __name__ == '__main__':

    # Train Data
    trainX = [[0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0],
              [0.0, 1.0, 1.0],
              [1.0, 0.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 0.0],
              [1.0, 1.0, 1.0]]
    trainY = [[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [0.0, 1.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 0.0]]

    testX = [[0.7, 1.0, 0.2],
              [0.5, 0.3, 1.0]]

    rbm = RBM()
    start_time = time.time()
    rbm.fit(trainX, trainY)
    predict = rbm.predict(testX)
    print predict
    print("--- %s seconds ---" % (time.time() - start_time))

