from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

train=fetch_20newsgroups( subset="train", remove=('headers','footers','quotes'))
test=fetch_20newsgroups( subset="test", remove=('headers','footers','quotes'))

vectorizer=CountVectorizer(stop_words="english")

xtrain=vectorizer.fit_transform(train.data)
xtest=vectorizer.transform(test.data)

ytrain=train.target
ytest=test.target

model1=MultinomialNB()
model1.fit(xtrain,ytrain)
ypred1=model1.predict(xtest)

model2=BernoulliNB()
model2.fit(xtrain,ytrain)
ypred2=model2.predict(xtest)

a1=accuracy_score(ytest,ypred1)
f11=f1_score(ytest,ypred1,average="macro")

a2=accuracy_score(ytest,ypred2)
f12=f1_score(ytest,ypred2,average="macro")

print("multinominal nb\n")
print ("accuracy ")
print(a1)
print("\n")
print("f1 ")
print(f11)

print("\nbernoulli nb\n")
print ("accuracy ")
print(a2)
print("\n")
print("f1 ")
print(f12)
