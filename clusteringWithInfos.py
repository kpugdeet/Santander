import os
# Mute TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import numpy as np
import pickle
import csv

# Read csv file
if os.path.isfile("userAttrKmeans.pkl") and os.path.isfile("userKmeans.pkl"):
    userAttr = pickle.load(open("userAttrKmeans.pkl", "rb"))
    user = pickle.load(open("userKmeans.pkl", "rb"))
else:
    user = dict()
    userAttr = []
    with open("./data/train_ver2.csv") as csvFile:
        readCSV = csv.DictReader(csvFile, delimiter=',')
        count = 0
        for i, row in enumerate(readCSV):
            checkID = row['ncodpers'].strip()
            if checkID not in user:
                if row['age'].strip() != "NA":
                    user[checkID] = count

                    age = row['age'].strip()
                    resident = 1 if row['indresi'] == "S" else 0
                    foreign = 1 if row['indext'] == "S" else 0
                    segment = np.zeros(3)
                    if row['segmento'][:2] != '':
                        segment[int(row['segmento'][:2])-1] = 1
                    sex = 1 if row['sexo'] == "V" else 0

                    tmpData = [age, resident, foreign, segment, sex]
                    userAttr.append(np.hstack(tmpData))
                    count += 1
            # if i > 1000:
            #     break

        userAttr = np.array(userAttr).astype(float)
        # pickle.dump(userAttr, open("userAttrKmeans.pkl", "wb"))
        # pickle.dump(user, open("userKmeans.pkl", "wb"))
        print (userAttr.shape)

# Standardize data of all column
userAttrScaled = preprocessing.scale(userAttr)

# print "Start Kmeans"
# numberLoop = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000])
# for n_clusters in numberLoop:
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(userAttr)
#     silhouette_avg = silhouette_score(userAttr, kmeans.labels_, sample_size=10000)
#     print('For n_clusters = {0} The average silhouette_score is : {1}'.format(n_clusters, silhouette_avg))

kmeans = KMeans(n_clusters=200, random_state=0).fit(userAttr)
silhouette_avg = silhouette_score(userAttr, kmeans.labels_, sample_size=10000)
print('For n_clusters = {0} The average silhouette_score is : {1}'.format(200, silhouette_avg))

f = open("./data/userKLabel.csv","w")
for key, value in user.iteritems():
    f.write("{0},{1}\n".format(key,kmeans.labels_[value]))

# Create logs directory
if tf.gfile.Exists('./logs' + '/kmeans'):
    tf.gfile.DeleteRecursively('./logs' + '/kmeans')
    tf.gfile.MkDir('./logs' + '/kmeans')
tf.gfile.MakeDirs('./logs'  + '/kmeans')

# Generate Meta data file
with open('./logs' + '/kmeans/metadata.tsv', 'w') as f:
    for label in kmeans.labels_[:10000]:
        f.write('{}\n'.format(label))

# Generate Embedding
sess = tf.InteractiveSession()

# Input set for Embedded TensorBoard visualization
# Performed with cpu to conserve memory and processing power
with tf.device("/cpu:0"):
    embedding = tf.Variable(userAttr[:10000], trainable=False, name='embedding')

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter('./logs' + '/kmeans', sess.graph)

# Add embedding tensorboard visualization. Need tensorflow version
# >= 0.12.0RC0
config = projector.ProjectorConfig()
embed= config.embeddings.add()
embed.tensor_name = 'embedding:0'
embed.metadata_path = os.path.join('./logs' + '/kmeans/metadata.tsv')
# embed.sprite.image_path = os.path.join('./mnist' + '/mnist_10k_sprite.png')

# Specify the width and height of a single thumbnail.
# embed.sprite.single_image_dim.extend([28, 28])

projector.visualize_embeddings(writer, config)

saver.save(sess, os.path.join('./logs', 'kmeans/a_model.ckpt'), global_step=10000)




