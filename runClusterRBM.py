# import tensorflow as tf
# g = tf.Graph()
# with g.as_default():
# 	x = tf.Variable(tf.random_normal([3]))
#
# s1 = tf.Session(graph=g)
# s2 = tf.Session(graph=g)
# s1.run(x.initializer)
# s2.run(x.initializer)
#
# print s1.run(x)
# print s2.run(x)
#
#
# y = tf.Variable(tf.random_normal([3]))
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess1 = tf.Session()
# sess.run(init)
# sess1.run(init)
#
# print sess.run(y)
# print sess1.run(y)


import csv
import numpy as np
import pickle
import os
from RbmTFSession import RBM
import time
import tensorflow as tf
import sys

# System Argument
fromDate = "2015-01-28"
toDate = "2016-04-28"
targetDate = "2016-05-28"
evalDate = "2016-06-28"
# fromDate = sys.argv[1]
# toDate = sys.argv[2]
# targetDate = sys.argv[3]
# evalDate = sys.argv[4]
threshold = 0.00
limit = 7
select = "/cpu:0"
maxLabel = 200

# Print option for numpy
np.set_printoptions(suppress=True)

# User preference and its index
userID = [dict() for x in range(maxLabel)]
user = [[] for x in range(maxLabel)]

# User preference and its index
userIDPref = dict()
userPref = []

# User preference ad its index for checking to predict
userIDTarget = dict()
userTarget = []

# User preference ad its index for evaluation
userIDEval = dict()
userEval = []

# User Label
userLabel = dict()

# Read Label for each user
with open("./data/userKLabel.csv") as csvFile:
    readCSV = csv.reader(csvFile, delimiter=',')
    for i, row in enumerate(readCSV):
        userLabel[row[0]] = int(row[1])

# Read csv file
if os.path.isfile("user.pkl") and os.path.isfile("userID.pkl"):
    user = pickle.load(open("user.pkl", "rb"))
    userID = pickle.load(open("userID.pkl", "rb"))
else:
    with open("./data/train_ver2.csv") as csvFile:
        readCSV = csv.reader(csvFile, delimiter=',')
        next(readCSV, None)
        count = np.zeros(maxLabel)
        countPref = 0
        countTarget = 0
        countEvaluation = 0
        for i, row in enumerate(readCSV):
            if fromDate <= row[0] <= toDate:
                checkID = row[1].strip()
                # if checkID in userID:
                #     index = userID[checkID]
                #     for _, x in enumerate(row[24:]):
                #         if user[index][_] == 0 and x != "NA":
                #             user[index][_] = x
                # else:
                if checkID in userLabel:
                    user[userLabel[checkID]].append(['0' if x == "NA" else x for x in row[24:]])
                    userID[userLabel[checkID]][checkID] = count[userLabel[checkID]]
                    count[userLabel[checkID]] += 1

                # For cumulative preference
                if checkID in userIDPref:
                    index = userIDPref[checkID]
                    for _, x in enumerate(row[24:]):
                        if userPref[index][_] == 0 and x != "NA":
                            userPref[index][_] = x
                else:
                    userPref.append(['0' if x == "NA" else x for x in row[24:]])
                    userIDPref[checkID] = countPref
                    countPref += 1
            if row[0] == targetDate:
                checkID = row[1].strip()
                userIDTarget[checkID] = countTarget
                userTarget.append(['0' if x == "NA" else x for x in row[24:]])
                countTarget += 1
            if row[0] == evalDate:
                checkID = row[1].strip()
                userIDEval[checkID] = countEvaluation
                userEval.append(['0' if x == "NA" else x for x in row[24:]])
                countEvaluation += 1

            # if i > 100:
            #     break

        user = [np.array(x).astype(np.float) for x in user]
        userPref = np.array(userPref).astype(np.float)
        userTarget = np.array(userTarget).astype(np.float)
        userEval = np.array(userEval).astype(np.float)
        # pickle.dump(user, open("user.pkl", "wb"))
        # pickle.dump(userID, open("userID.pkl", "wb"))


# Check the input Data is correct
for x in user:
    print (x.shape, len(x))
print (userPref.shape, len(userIDPref))
print (userTarget.shape, len(userIDTarget))
print (userEval.shape, len(userIDEval))


# Training RBM
rbm = RBM (24, 12, ["w", "vb", "hb", "dW", "dVb", "dHb"], "./logs", select)
init = tf.global_variables_initializer()
listRBM = dict()
for i, eachUser in enumerate(user):
    if eachUser.shape[0] != 0:
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        startTime = time.time()
        rbm.fit(eachUser, 100, sess)
        listRBM[i] = sess
        print ("Time used: {0:15} {1}".format(time.time()-startTime, i))


# Read Test data
test = [[] for x in range(maxLabel)]
testID = [[] for x in range(maxLabel)]
with open("./data/test_ver2.csv") as csvFile:
    readCSV = csv.reader(csvFile, delimiter=',')
    next(readCSV, None)
    count = 0
    for i, row in enumerate(readCSV):
        if row[1].strip() in userIDPref and row[1].strip() in userIDTarget:
            # test.append(userTarget[userIDTarget[row[1].strip()]])
            test[userLabel[row[1].strip()]].append(np.logical_or(userTarget[userIDTarget[row[1].strip()]], userPref[userIDPref[row[1].strip()]]).astype(np.float))
            testID[userLabel[row[1].strip()]].append(row[1].strip())
        elif row[1].strip() in userIDTarget:
            test[userLabel[row[1].strip()]].append(userTarget[userIDTarget[row[1].strip()]])
            testID[userLabel[row[1].strip()]].append(row[1].strip())
    test = [np.array(x).astype(np.float) for x in test]
    for x in test:
        print (x.shape, len(x))


# Do recommendation
submission = dict()
itemName = ["ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
            "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
            "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1",
            "ind_tjcr_fin_ult1", "ind_valo_fin_ult1", "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1"]
f = open('submission.csv', 'w')
f.write("ncodpers,added_products\n")
for z, testData in enumerate(test):
    countEmpty = 0
    for _, pre in enumerate(rbm.predictV(np.array(testData), listRBM[z])):
        userIndex = userIDTarget[testID[z][_]]
        f.write("{0},".format(testID[z][_]))
        submission[testID[z][_]] = ""
        sortPre = sorted(range(len(pre)), key=lambda k: pre[k], reverse=True)
        outList = []
        subList = []
        for index in sortPre:
            if userTarget[userIndex][index] == 0 and pre[index] >= threshold and len(outList) < limit:
                outList.append(itemName[index])
                subList.append(index)
        if len(outList) != 0:
            f.write(" ".join(map(str, outList)))
            submission[testID[z][_]] = subList
        else:
            countEmpty += 1
        f.write("\n")
    print (z, countEmpty)


# # Evaluation
# countCorrect = []
# for key, value in submission.iteritems():
#     if key in userIDEval:
#         evalIndex = userIDEval[key]
#         targetIndex = userIDTarget[key]
#         relevantNodes = 0
#         predictNodes = len(value)
#         tmpAP = 0.0
#         # Loop for counting relevant nodes
#         for _, status in enumerate(userEval[evalIndex]):
#             if status == 1 and userTarget[targetIndex][_] == 0:
#                 relevantNodes += 1
#         # Calculate AP@Limit
#         correctIndex = 1.0
#         for _, index in enumerate(value):
#             if userEval[evalIndex][index] == 1:
#                 tmpAP += (correctIndex/(_+1))
#                 correctIndex += 1
#         # Append the AP to list to calculate MAP
#         if min(relevantNodes, predictNodes) == 0:
#             countCorrect.append(0.0)
#         else:
#             tmpAP /= min(relevantNodes, predictNodes)
#             countCorrect.append(tmpAP)
#
# countCorrect = np.array(countCorrect)
# print (fromDate, toDate, targetDate, evalDate, np.mean(countCorrect), len(testID))
# # print np.mean(countCorrect)

