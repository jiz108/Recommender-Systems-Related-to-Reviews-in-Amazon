#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gzip
from collections import defaultdict 
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords 
import string 
nltk.download('punkt')
def readGz(f):
    for l in gzip.open(f): 
        yield eval(l)


# In[3]:


#Problem 1
import random
count = 0
users = set()
businesses = set()
training_set = []
validation_set = [] 
all_pairs_set = set()
for l in readGz("/Users/jiawei/Desktop/assignment1/train.json.gz"):
    user = l['reviewerID']
    business = l['itemID']
    users.add(user)
    businesses.add(business)
    all_pairs_set.add((user, business))
    count += 1
    if count <= 100000:
        training_set.append((user, business))
    if count >100000 and count <= 200000:
        validation_set.append((user, business)) 

        
    
users = list(users) 
businesses = list(businesses) 

i=0
while i < 100000:
    u = random.sample(users, 1)
    b = random.sample(businesses, 1) 
    if (u[0],b[0]) in all_pairs_set:
        continue
    i+=1
    validation_set.append((u[0],b[0]))
print(len(validation_set))
print(len(users))
print(len(businesses))


# In[4]:


def would_purchase_baseline(threshold):
    businessCount = defaultdict(int)
    totalPurchases = 0

    for l in training_set:
        user = l[0]
        business = l[1]
        businessCount[business] += 1
        totalPurchases += 1

    mostPopular = [(businessCount[x], x) for x in businessCount]
    mostPopular.sort()
    mostPopular.reverse()

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalPurchases/threshold: break
    
    count = 0
    correct = 0
    for a, b in validation_set:
        count += 1
        if b in return1:
            if count <= 100000: 
                correct += 1
        else:
            if count > 100000:
                correct += 1 

    return (correct/200000.0)


# In[5]:


accuracy = would_purchase_baseline(2)
print('Accuracy = ', accuracy)


# In[6]:


#Problem 2
maxAccuracy = 0
maxThreshold = 0

for thr in range(1,100):
    x = would_purchase_baseline(thr) 
    if x > maxAccuracy:
        maxThreshold = thr
        maxAccuracy = x
print('Thresh-hold at ', maxThreshold, 'with maximum accuracy = ', maxAccuracy)


# In[7]:


#Problem 3

userCategories = defaultdict(set) 
itemCategory = defaultdict(int) 
for l in readGz("/Users/jiawei/Desktop/assignment1/train.json.gz"):
    user =  l['reviewerID']
    item = l['itemID']
    cat = l['categoryID']
    userCategories[user].add(cat)
    itemCategory[item] = cat


# In[ ]:


#Problem 4
predictions = open("/Users/jiawei/Desktop/assignment1/predictions_Purchase.txt", 'w')
for l in open("/Users/jiawei/Desktop/assignment1/pairs_Purchase.txt"):
    if l.startswith("reviewerID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    
    if (i in itemCategory and u in userCategories and itemCategory[i] in userCategories[u]):
        predictions.write(u + '-' + i + ",1\n")
    else:
        predictions.write(u + '-' + i + ",0\n")

predictions.close()


# In[ ]:


##### Assignment 1 #####


# In[ ]:


#Build up the dicts 
userCategories = defaultdict(set) 
itemCategory = defaultdict(int) 
userItem = defaultdict(set)
item_user = defaultdict(set)

#for checking popular
businessCount = defaultdict(int)
totalPurchases = 0

#Get categories of user's and item's
userCatWord = defaultdict(list)
itemCatWord = defaultdict(list)

for l in readGz("/Users/jiawei/Desktop/assignment1/train.json.gz"):
    user =  l['reviewerID']
    item = l['itemID']
    cat = l['categoryID']
    categories = l['categories']
    business = l['itemID']
    businessCount[business] += 1
    totalPurchases += 1
    
    #add categories
    userCatWord[user].append(categories)
    itemCatWord[item].append(categories)
    
    userCategories[user].add(cat)
    itemCategory[item] = cat
    userItem[user].add(item)
    item_user[item].add(user)


# In[ ]:


#Check if this item's categories are inside the cagetories of this user's buy items 
def predictByCat(user, item):
    #get the categories of this item
    a = itemCatWord[item]
    
    #check it has "categories"
    length = len(a)
    if(length == 0):
        return False

    #get the item last category
    last = a[len(a)-1]
    last = last[len(last)-1]

    for c in userCatWord[user]:
        for sub_c in c:
            if(last == sub_c):
                return True
    return False


# In[ ]:


#Jaccard by all users and the categoryID
import operator
def jaccard(user, item):
    simList = []
    sim = 0
    for u in users:
        #get the Categories this user buy
        a = userCategories[u] 
        b = userCategories[user]        
        c = a.intersection(b)        
        sim = (float(len(c))/float(len(a)+len(b)-len(c)))      
        simList.append((sim,u))
    simList.sort(key = operator.itemgetter(0), reverse = True)
    simList = simList[:1600]
    for i in simList:
        if (item in userItem[i[1]]):
            return True
    return False


# In[ ]:


#Jaccard by users who buy this item and compare the items they buy 
def Jaccard(user, item):
    sim = 0
    optimal_user = []
    for u in item_user[item]:
        #get the items this user buy
        a = userItem[u] 
        b = userItem[user]        
        c = a.intersection(b)        
        sim = (float(len(c))/float(len(a)+len(b)-len(c)))       
        if(sim > 0.02):
            optimal_user.append(u)
    for i in range (len(optimal_user)):
        if(item in userItem[optimal_user[i]]):
            return True
    return False


# In[ ]:


#check the item is popular or not, if it is popular then put it inside return1
mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()
    
return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPurchases*0.45: break


# In[ ]:


#Predict and write the file
predictions = open("/Users/jiawei/Desktop/assignment1/predictions_Purchase.txt", 'w')
for l in open("/Users/jiawei/Desktop/assignment1/pairs_Purchase.txt"):

    if l.startswith("reviewerID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
   

        
    if (predictByCat(u,i) or (i in return1) ):
        predictions.write(u + '-' + i + ",1\n")
    else:
        predictions.write(u + '-' + i + ",0\n")

predictions.close()


# In[ ]:


#######################################################


# In[ ]:


#Problem 5
cat_training_set = []
cat_validation_set = []
count = 0
for l in readGz("/Users/jiawei/Desktop/assignment1/train.json.gz"): 
    count += 1
    if 'categoryID' not in l: 
        continue
    if count <= 100000: 
        cat_training_set.append(l)
    if count >100000 and count <= 200000: 
        cat_validation_set.append(l)


# In[ ]:


catPopularity = defaultdict(int)
userCatCounts = defaultdict(lambda: defaultdict(int)) 
userPrediction = defaultdict(int)
for l in cat_training_set:
    user,item,cat = l['reviewerID'],l['itemID'],l['categoryID'] 
    catPopularity[cat] += 1
    userCatCounts[user][cat] += 1
    
for u in userCatCounts: 
    mxv = 0
    mxc = 0
    for i in range(5):
        if userCatCounts[u][i] > mxv: 
            mxv = userCatCounts[u][i] 
            mxc = i
    userPrediction[u] = mxc
    
correct = 0
for l in cat_validation_set:
    user,cat = l['reviewerID'],l['categoryID'] 
    if userPrediction[user] == cat:
        correct += 1
        

print('Accuracy = ', correct/len(cat_validation_set))


# In[ ]:


#Problem 6
# Get top 500 words
stop = [] #stopwords.words('english') 
punctuation = list(string.punctuation) 
wordCount = defaultdict(int)
for l in cat_training_set:
    r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation]) 
    words = [i for i in r.split() if i not in stop]
    l['reviewWords'] = words
    for w in words:
        wordCount[w] += 1
mostPopularWords = [(wordCount[x], x) for x in wordCount] 
mostPopularWords.sort()
mostPopularWords.reverse()
mostPopularWords = [x[1] for x in mostPopularWords[:500]] 
mostPopularWords_ = set(mostPopularWords) 
print(mostPopularWords[:10])


# In[ ]:


freq = defaultdict(int)
totalFreq = 0
freqCat = defaultdict(lambda: defaultdict(int)) 
totalFreqCat = defaultdict(int)
for l in cat_training_set:
    r = l['reviewWords'] 
    cat = l['categoryID'] 
    for w in r:
        if w in mostPopularWords_: 
            freq[w] += 1
            totalFreq += 1
            freqCat[cat][w] += 1
            totalFreqCat[cat] += 1
            
for w in freq:
    freq[w] = freq[w] * 1.0 / totalFreq 
    for cat in range(5):
        freqCat[cat][w] = 1.0 * freqCat[cat][w] / totalFreqCat[cat]
for cat in range(5): 
    lst = []
    for w in mostPopularWords: 
        lst.append((freqCat[cat][w] - freq[w], w))
    lst.sort()
    lst.reverse()
    print('Category: ', cat, ' top words: ', [x[1] for x in lst[:10]])


# In[ ]:


#Problem 7
from sklearn import svm 
wordId = defaultdict(int) 
k=0
for w in mostPopularWords:
    wordId[w] = k
    k += 1
    
def feature(datum):
    feat = [0]*len(mostPopularWords)
    r = ''.join([c for c in datum['reviewText'].lower() if not c in punctuation]) 
    for w in r.split():
        if w in mostPopularWords_: 
            feat[wordId[w]] += 1
    feat.append(1) #offset
    return feat


X_train = []
Y_train = []
X_valid = []
Y_valid = []
for l in cat_training_set[:10000]:
    X_train.append(feature(l)) 
    if l['categoryID'] == 0:
        Y_train.append(0) 
    else:
        Y_train.append(1)
for l in cat_validation_set[:10000]: 
    X_valid.append(feature(l)) 
    Y_valid.append(l['categoryID'])


# In[ ]:


for c in [0.01, 0.1]:#, 1, 10]:
    clf = svm.SVC(C=c, kernel='linear') 
    clf.fit(X_train, Y_train) 
    predictions = clf.predict(X_valid) 
    correct_tp = 0
    correct = 0
    for i in range(len(Y_valid)):
        if predictions[i] == 0 and Y_valid[i] == 0: 
            correct_tp += 1
            correct += 1
        if predictions[i] == 1 and Y_valid[i] != 0:
            correct += 1
    print('validation accuracy (TP only) for c = ', c, ' = ', correct_tp*1.0/len(X_valid))
    print('validation accuracy  for c = ', c, ' = ', correct*1.0/len(X_valid))


# In[ ]:


#Problem 8
def feature(datum):
    feat = [0]*len(mostPopularWords)
    r = ''.join([c for c in datum['reviewText'].lower() if not c in punctuation]) 
    for w in r.split():
        if w in mostPopularWords_: 
            feat[wordId[w]] += 1
        feat.append(1) #offset 
        return feat
    
X_train = []
Y_train = {}
X_valid = []
Y_valid = []

for cat in range(5):
    Y_train[cat] = []
    for l in cat_training_set[:10000]:
        if cat == 0: 
            X_train.append(feature(l))
        if l['categoryID'] == cat: 
            Y_train[cat].append(0)
        else: 
            Y_train[cat].append(1)
for l in cat_validation_set[:10000]: 
    X_valid.append(feature(l)) 
    Y_valid.append(l['categoryID'])


# In[ ]:


for c in [0.01, 0.1, 1, 10]: 
    clf = {}
    for cat in range(5):
        clf[cat] = svm.SVC(C=c, kernel='linear') 
        clf[cat].fit(X_train, Y_train[cat])
    correct = 0
    for i in range(len(Y_valid)):
        lst = []
        for cat in range(5):
            lst.append(clf[cat].decision_function([X_valid[i]])[0]) 
        for cat in range(5):
            if lst[cat] == min(lst): 
                prediction = cat
                break                
        if predictions[i] == Y_valid[i]:
            correct += 1
    print('validation accuracy for c = ', c, ' = ', correct*1.0/len(X_valid))


# In[ ]:


##### Assignment 1 #####


# In[ ]:


import string
cat_training_set = [l for l in readGz("/Users/jiawei/Desktop/assignment1/train.json.gz") if 'categoryID' in l]

punctuation = set(string.punctuation) 
wordCount = defaultdict(int)
wordCount_cate = defaultdict(int)
userCateID = defaultdict(list)
countCatWord = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]

for l in cat_training_set:
    r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation]) 
    cat = l['categoryID']
    u = l['reviewerID']
    userCateID[u] += [cat]
    
    for w in r.split():
        wordCount[w] += 1
        countCatWord[cat][w] += 1

mostPopularWords = [(wordCount[w],w) for w in wordCount]
mostPopularWords.sort()
mostPopularWords.reverse()
mostPopularWords = [p[1] for p in mostPopularWords[:2000]]


# In[ ]:


from sklearn import svm
import numpy as np

def feature(l):
    feat = []
    r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation]) 
    u = l['reviewerID']
    r = r.split()
    for w in mostPopularWords:
        feat.append(int(w in r))
    feat += [int(i in userCateID[u]) for i in range(5)]
    return feat

X_train = [feature(l) for l in cat_training_set]
Y_train = []
for i in range(5):
    Y_train.append([int(l['categoryID']) == i for l in cat_training_set])
    
data = [l for l in readGz("/Users/jiawei/Desktop/assignment1/test_Category.json.gz")]
X_test = [feature(l) for l in data]


# In[ ]:


#Predict
predictions = open("/Users/jiawei/Desktop/assignment1/predictions_Category.txt", 'w')
predictions.write("reviewerID-reviewHash,category\n")

clf = []
y_pred = []
scores = []
X_validatoin = [feature(l) for l in data]

for cat in range(5):
    clf.append(svm.LinearSVC(C = 0.01))
    clf[cat].fit(X_train, Y_train[cat])
    scores.append(clf[cat].decision_function(X_validatoin))
    
for k in range(len(X_validatoin)):
    max_score = max([scores[i][k] for i in range(5)])
    for i in range(5):
        if scores[i][k] == max_score:
            y_pred.append(i)
            break
            
for i in range(len(data)):
    predictions.write(data[i]['reviewerID'] + '-' + data[i]['reviewHash'] + "," + str(y_pred[i]) + "\n")

predictions.close()


# In[ ]:




