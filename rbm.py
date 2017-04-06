from __future__ import division
import numpy as np
from scipy import spatial
import os.path

# Constant data path
PARAMS_DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + "/params/"

class RBM:
    def __init__(self,):
        print ("Initialize RBM")
        self.batches = []
        self.words = 0

    # sigmoid function:
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    # SoftMax function:
    def softmax(self, x):
        numerator = np.exp(x)
        denominator = numerator.sum(axis=1)
        denominator = denominator.reshape((x.shape[0], 1))
        softmax = numerator / denominator
        return softmax

    # Calculate and return Negative hidden states and probs
    def negativeProb(self, vis, hid, D):
        neg_vis = np.dot(hid, self.weights.T) + self.vbias
        softmax_value = self.softmax(neg_vis)
        neg_vis *= 0
        for i in xrange(len(vis)):
            neg_vis[i] = self.np_rng.multinomial(D[i], softmax_value[i], size=1)
        D = np.sum(neg_vis, axis=1)

        perplexity = np.nansum(vis * np.log(softmax_value))

        neg_hid_prob = self.sigmoid(np.dot(neg_vis, self.weights) + np.outer(D, self.hbias))
        return neg_vis, neg_hid_prob, D, perplexity

    # Train RSM model
    def trainRBM(self, max_epochs=15, batch_size=10, step=1, weight_cost=0.0002, momentum=0.9):
        data = self.input
        num_of_train = len(data)
        current_batch = batch_size
        while (current_batch + batch_size <= num_of_train):
            self.batches.append(current_batch)
            current_batch += batch_size
        self.batches.append(num_of_train)
        for epoch in range(max_epochs):
            # Divide in to minibatch
            total_batch = len(self.batches)
            start_batch = 0
            reconstruction_error = 0
            perplexity = 0

            # Loop for each batch
            for batch_index in range(total_batch):
                # Get the data for each batch
                pos_vis = data[start_batch:self.batches[batch_index]]
                batch_size = len(pos_vis)
                start_batch = self.batches[batch_index]
                D = np.sum(pos_vis, axis=1)
                if epoch == 0:
                    self.words += np.sum(
                        pos_vis)  # Calculate the number of words in order to calculate the perplexity.
                # Caculate positive probs and Expectation for Sigma(ViHj) data
                pos_hid_prob = self.sigmoid(np.dot(pos_vis, self.weights) + np.outer(D, self.hbias))

                # If probabilities are higher than randomly generated, the states are 1
                randoms = self.np_rng.rand(batch_size, self.num_hidden)
                pos_hidden_states = np.array(randoms < pos_hid_prob, dtype=int)

                neg_vis = pos_vis
                neg_hid_prob = pos_hidden_states
                # Calculate negative probs and Expecatation for Sigma(ViHj) recon with k = 1,....
                for i in range(step):
                    neg_vis, neg_hid_prob, D, p = self.negativeProb(neg_vis, pos_hid_prob, D)
                    if i == 0:
                        perplexity += p

                # Update weight
                pos_products = np.dot(pos_vis.T, pos_hid_prob)
                pos_visible_bias_activation = np.sum(pos_vis, axis=0)
                pos_hidden_bias_activation = np.sum(pos_hid_prob, axis=0)
                neg_products = np.dot(neg_vis.T, neg_hid_prob)
                neg_visibe_bias_activation = np.sum(neg_vis, axis=0)
                neg_hidden_bias_activation = np.sum(neg_hid_prob, axis=0)

                # Update the weights and biases
                self.delta_weights = momentum * self.delta_weights + self.learning_rate * ((pos_products - neg_products) / batch_size - weight_cost * self.weights)
                self.delta_vbias = (momentum * self.delta_vbias + (
                    pos_visible_bias_activation - neg_visibe_bias_activation)) * (self.learning_rate / batch_size)
                self.delta_hbias = (momentum * self.delta_hbias + (
                    pos_hidden_bias_activation - neg_hidden_bias_activation)) * (self.learning_rate / batch_size)
                self.weights += self.delta_weights
                self.vbias += self.delta_vbias
                self.hbias += self.delta_hbias

                reconstruction_error += np.square(pos_vis - neg_vis).sum()
            perplexity = np.exp(-perplexity / self.words)
            print('Epoch: {}, Error={}, Perplexity={}'.format(epoch, reconstruction_error, perplexity))

    def get_hidden_pro(self, visible):
        hidden_pro = self.sigmoid(np.dot(visible, self.weights) + np.outer(np.sum(visible,axis=1), self.hbias))
        return hidden_pro

    def save_RBM_weights(self):
        filename = PARAMS_DATA_PATH +"weights_added_biases.dat"
        with open(filename, "wb") as file:
            np.savez(file=file, weights=self.weights, hbias=self.hbias, vbias=self.vbias)

    def load_RBM_weights(self):
        filename = PARAMS_DATA_PATH +"weights_added_biases.dat"
        data=np.load(filename)
        weights=data["weights"]
        hbias=data["hbias"]
        vbias=data["vbias"]
        return weights,hbias,vbias

    def get_train_output(self, data):
        output=self.get_hidden_pro(data)
        topic_distri=[]
        for f in output:
            topic_distri.append(",".join(['{:.5f}'.format(x) for x in f]))
        return topic_distri

    def train(self, items_des):
        self.input=self.processer.get_bag_words_matirx(items_des, max_vocaulary=10000)

        self.num_hidden = 20
        self.num_visible = len(self.input[0])
        self.learning_rate = 0.01
        max_epochs = 50
        batch_size = 100
        step = 1

        # Create Random generator
        self.np_rng = np.random.RandomState(1234)

        # Initial Weights
        mu, sigma = 0, np.sqrt(0.01)
        self.weights = self.np_rng.normal(mu, sigma, (
            self.num_visible, self.num_hidden))
        # Inital hidden Bias
        self.hbias = np.zeros(self.num_hidden)
        # Inital visible Bias
        self.vbias = np.zeros(self.num_visible)

        self.delta_weights = np.zeros((self.num_visible, self.num_hidden))
        self.delta_hbias = np.zeros(self.num_hidden)
        self.delta_vbias = np.zeros(self.num_visible)

        self.trainRBM(max_epochs=max_epochs, batch_size=batch_size, step=step)
        topic_distri=self.get_train_output(self.input)
        self.save_RBM_weights()
        return topic_distri

    def calRanking(self, meta_card, user_pref):
        try:
            self.weights,self.hbias,vbias=self.load_RBM_weights()
        except:
            print("Can't find weights and biases in directory!!!")
        recommend_result = {}
        new_data_matrix=self.processer.get_bag_words_matrix_by_vocabulary(meta_card)
        new_topic_distri=self.get_hidden_pro(new_data_matrix)
        distances=[]
        for t1 in new_topic_distri:
            d=0
            for t2 in user_pref:
                d+=spatial.distance.euclidean(t1,t2)
            distances.append(d)
        ind=np.argsort(distances)
        for idx,i in enumerate(ind):
            recommend_result[str(idx)]=meta_card[i]
        return recommend_result

    def calDistri(self, text):
        try:
            self.weights,self.hbias,vbias=self.load_RBM_weights()
        except:
            print("Can't find weights and biases in directory!!!")
        new_data_matrix=self.processer.get_bag_words_matrix_by_vocabulary([text])
        topic_distri=self.get_train_output(new_data_matrix)
        return topic_distri


if __name__ == '__main__':
    rbm = RBM()
    items_ID = ["123", "125", "127"]
    items_des = ["I have a pen", "I have a apple", "umm apple pen"]
    meta_card=["pen ccc","a a a I should fly","a good apple"]
    user_pref=[[0.2,0.2,0.3,0.3],[0.1,0,0.4,0.5]]
    text="have a good night umm"
    topic_distri=rbm.train(items_des)
    print(topic_distri)
    recommend_result=rbm.calRanking(meta_card,user_pref)
    print(recommend_result)
    # topic_distri2=rbm.calDistri(text)
    # print(topic_distri2)
