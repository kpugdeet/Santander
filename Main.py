import csv
import numpy as np
import pickle
import os
import sys
from RbmTf import RBM
import time

np.set_printoptions(suppress=True)

# User preference and its index
userID = dict()
user = []

# User preference ad its index for checking to predict
userIDTarget = dict()
userTarget = []

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
        for i, row in enumerate(readCSV):
            if row[0] >= sys.argv[1]:
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
            if row[0] == "2016-05-28":
                checkID = row[1].strip()
                userIDTarget[checkID] = countTarget
                userTarget.append(['0' if x == "NA" else x for x in row[24:]])
                countTarget += 1

        user = np.array(user).astype(np.float)
        userTarget = np.array(userTarget).astype(np.float)
        # pickle.dump(user, open("user.pkl", "wb"))
        # pickle.dump(userID, open("userID.pkl", "wb"))

# Check the input Data is correct
print (user.shape, len(userID))
print (userTarget.shape, len(userIDTarget))
label, indices = np.unique(user,return_inverse=True)
count = np.bincount(indices)
print (label, count)
print (float(count[0]/float(count[1])))
print (user.shape[0]*user.shape[1]) == np.sum(count)

# Training RBM
rbm = RBM (user.shape[1], user.shape[1]/2, ["w", "vb", "hb", "dW", "dVb", "dHb"], "./logs")
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
        test.append(userTarget[userIDTarget[row[1].strip()]])
        testID.append(row[1].strip())

# Do recommendation
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
    sortPre = sorted(range(len(pre)), key=lambda k: pre[k], reverse=True)
    outList = []
    for index in sortPre:
        if userTarget[userIndex][index] == 0 and pre[index] >= 0.01 and len(outList) < 7:
            outList.append(itemName[index])
    if len(outList) != 0:
        f.write(" ".join(map(str, outList)))
    else:
        countEmpty += 1
    f.write("\n")

print countEmpty
# target = 0
# print rbm.predictH(np.array([user[target]]))
# print rbm.predictV(np.array([user[target]]))
# print user[target]
