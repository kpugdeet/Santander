import csv
import numpy as np
import pickle
import os
from RbmTf import RBM
import time

# System Argument
fromDate = "2016-02-28"
toDate = "2016-04-28"
targetDate = "2016-04-28"
evalDate = "2016-05-28"
threshold = 0.00
limit = 7
select = "/cpu:0"

# Print option for numpy
np.set_printoptions(suppress=True)

# User preference and its index
userID = dict()
user = []

# User preference ad its index for checking to predict
userIDTarget = dict()
userTarget = []

# User preference ad its index for evaluation
userIDEval = dict()
userEval = []

# Read csv file
if os.path.isfile("user.pkl") and os.path.isfile("userID.pkl"):
    user = pickle.load(open("user.pkl", "rb"))
    userID = pickle.load(open("userID.pkl", "rb"))
else:
    with open("./data/train_ver2.csv") as csvFile:
        readCSV = csv.reader(csvFile, delimiter=',')
        next(readCSV, None)
        count = 0
        countTarget = 0
        countEvaluation = 0
        for i, row in enumerate(readCSV):
            if fromDate <= row[0] <= toDate:
                checkID = row[1].strip()
                if checkID in userID:
                    index = userID[checkID]
                    for _, x in enumerate(row[24:]):
                        if user[index][_] == 0 and x != "NA":
                            user[index][_] = x
                else:
                    userID[checkID] = count
                    user.append(['0' if x == "NA" else x for x in row[24:]])
                    count += 1
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

        user = np.array(user).astype(np.float)
        userTarget = np.array(userTarget).astype(np.float)
        userEval = np.array(userEval).astype(np.float)
        # pickle.dump(user, open("user.pkl", "wb"))
        # pickle.dump(userID, open("userID.pkl", "wb"))

# Check the input Data is correct
print (user.shape, len(userID))
print (userTarget.shape, len(userIDTarget))
print (userEval.shape, len(userIDEval))
label, indices = np.unique(user,return_inverse=True)
count = np.bincount(indices)
print (label, count)
print (float(count[0]/float(count[1])))
print (user.shape[0]*user.shape[1]) == np.sum(count)

# Training RBM
rbm = RBM (user.shape[1], user.shape[1]/2, ["w", "vb", "hb", "dW", "dVb", "dHb"], "./logs", select)
startTime = time.time()
rbm.fit(user, 100)
rbm.saveWeights("./model.ckpt")
# rbm.restoreWeights("./model.ckpt")
print ("Time used: {0}".format(time.time()-startTime))

# Read Test data
test = []
testID = []
with open("./data/test_ver2.csv") as csvFile:
    readCSV = csv.reader(csvFile, delimiter=',')
    next(readCSV, None)
    count = 0
    for i, row in enumerate(readCSV):
        if row[1].strip() in userIDTarget:
            test.append(userTarget[userIDTarget[row[1].strip()]])
            testID.append(row[1].strip())
    test = np.array(test).astype(np.float)
    print (test.shape, len(testID))

# Do recommendation
submission = dict()
itemName = ["ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
            "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
            "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1",
            "ind_tjcr_fin_ult1", "ind_valo_fin_ult1", "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1"]
f = open('submission.csv', 'w')
f.write("ncodpers,added_products\n")
countEmpty = 0
for _, pre in enumerate(rbm.predictV(np.array(test))):
    userIndex = userIDTarget[testID[_]]
    f.write("{0},".format(testID[_]))
    submission[testID[_]] = ""
    sortPre = sorted(range(len(pre)), key=lambda k: pre[k], reverse=True)
    outList = []
    subList = []
    for index in sortPre:
        if userTarget[userIndex][index] == 0 and pre[index] >= threshold and len(outList) < limit:
            outList.append(itemName[index])
            subList.append(index)
    if len(outList) != 0:
        f.write(" ".join(map(str, outList)))
        submission[testID[_]] = subList
    else:
        countEmpty += 1
    f.write("\n")
print countEmpty
# target = 0
# print rbm.predictH(np.array([user[target]]))
# print rbm.predictV(np.array([user[target]]))
# print user[target]

# Evaluation
countCorrect = []
for key, value in submission.iteritems():
    evalIndex = userIDEval[key]
    targetIndex = userIDTarget[key]
    relevantNodes = 0
    predictNodes = len(value)
    tmpAP = 0.0
    # Loop for counting relevant nodes
    for _, status in enumerate(userEval[evalIndex]):
        if status == 1 and userTarget[targetIndex][_] == 0:
            relevantNodes += 1
    # Calculate AP@Limit
    correctIndex = 1.0
    for _, index in enumerate(value):
        if userEval[evalIndex][index] == 1:
            tmpAP += (correctIndex/(_+1))
            correctIndex += 1
    # Append the AP to list to calculate MAP
    if min(relevantNodes, predictNodes) == 0:
        countCorrect.append(0.0)
    else:
        tmpAP /= min(relevantNodes, predictNodes)
        countCorrect.append(tmpAP)

print countCorrect[0]
countCorrect = np.array(countCorrect)
print np.mean(countCorrect)

