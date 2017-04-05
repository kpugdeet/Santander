import csv
import numpy as np
import pickle
import os

# User preference and its index
userID = {}
user = []

# Read csv file
if os.path.isfile("user.pkl") and os.path.isfile("userID.pkl"):
    user = pickle.load(open("user.pkl", "rb"))
    userID = pickle.load(open("userID.pkl", "rb"))
else:
    with open("./data/train_ver2.csv") as csvFile:
        readCSV = csv.reader(csvFile, delimiter=',')
        next(readCSV, None)
        count = 0
        # globalCount = 0
        # prev = ""
        # countEach = 0
        for i, row in enumerate(readCSV):
            if row[0] == "2016-05-28":
                userID[row[1]] = count
                user.append(['0' if x == 'NA' else x for x in row[24:]])
                count += 1
            # globalCount += 1
            # if row[0] != prev:
            #     print countEach
            #     countEach = 0
            #     print row[0]
            #     prev = row[0]
            # countEach += 1
            # if i >= 999999:
            #     break
        user = np.array(user).astype(np.float)
        pickle.dump(user, open("user.pkl", "wb"))
        pickle.dump(userID, open("userID.pkl", "wb"))

print (user.shape, len(userID))
