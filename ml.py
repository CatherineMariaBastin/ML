#sixth question

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split

data=pd.read_csv("diabetes.csv")

data=data.dropna()

X=data.drop(columns=["Outcome"])
y=data["Outcome"]

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)

model.fit(xtrain,ytrain)
pred=model.predict(xtest)

print("Without Scaling\n")
print("Accuracy: ",accuracy_score(ytest,pred))
print("Classification Report\n ",classification_report(ytest,pred))

scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)

model.fit(xtrain,ytrain)
pred=model.predict(xtest)

print("With Scaling\n")
print("Accuracy: ",accuracy_score(ytest,pred))
print("Classification Report\n ",classification_report(ytest,pred))


#---------------------------------------------
#seventh question 7

from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

train=fetch_20newsgroups(subset="train",remove=('headers','footers','quotes'))
test=fetch_20newsgroups(subset="test",remove=('headers','footers','quotes'))

vectorizer=CountVectorizer(stop_words='english')

xtrain=vectorizer.fit_transform(train.data)
xtest=vectorizer.transform(test.data)

ytrain=train.target
ytest=test.target

model=MultinomialNB()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print("Accuracy: ",accuracy_score(ytest,pred))
print("F1_Score: ",f1_score(ytest,pred,average='weighted'))

model=BernoulliNB()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print("Accuracy: ",accuracy_score(ytest,pred))
print("F1_Score: ",f1_score(ytest,pred,average='weighted'))

#---------------------------------------------------
#eighth question 8
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

train=pd.read_csv('fashion-mnist_train.csv')
test=pd.read_csv('fashion-mnist_test.csv')

xtrain=train.drop('label',axis=1)
xtest=test.drop('label',axis=1)
ytrain=train['label']
ytest=test['label']

xtrain=xtrain[:10000]/255.0
ytrain=ytrain[:10000]
xtest=xtest[:2000]/255.0
ytest=ytest[:2000]

for k in [1,3,5,7]:
  model=KNeighborsClassifier(n_neighbors=k)
  model.fit(xtrain,ytrain)
  pred=model.predict(xtest)
  print("Accuracy: ",accuracy_score(ytest,pred))
  print("Classification Report\n ",classification_report(ytest,pred))

#----------------------------------------------------
#9th question

from sklearn.tree import DecisionTreeClassifier,plot_tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split

df=pd.read_csv('OnlineRetail.csv',encoding='latin1')
df=df.dropna(subset=['CustomerID'])
df=df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df=df[(df['Quantity']>0) & (df['UnitPrice']>0)]

df['CustomerID']=df['CustomerID'].astype(int)
df['Amount']=df['Quantity']*df['UnitPrice']

customer=df.groupby('CustomerID').agg(
    TotalAmount=('Amount','sum'),
    Frequency=('InvoiceNo','nunique'),
    AvgQuantity=('Quantity','mean'),
    AvgPrice=('Amount','mean')
)

customer['segment']=pd.qcut(customer['TotalAmount'].rank(method='first'),q=3,labels=['low','medium','high'])

x=customer[['Frequency','AvgQuantity','AvgPrice']]
y=customer['segment']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=42)

model.fit(xtrain,ytrain)
pred=model.predict(xtest)

print("accuracy score: ",accuracy_score(ytest,pred))
print(classification_report(ytest,pred,zero_division=0))

plt.figure(figsize=(54,24))
plot_tree(model,feature_names=X.columns,class_names=model.classes_,filled=True,rounded=True)
plt.title('Decision Tree')
plt.tight_layout()
plt.show()

importance=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
print("\n Feature Importances\n")
print(importance)

#---------------------------------------------
#10th

import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

data=pd.read_csv('adult.csv')
df=data.dropna()

x=df.drop('income',axis=1)
y=(df['income']=='>50K').astype(int)

cat=x.select_dtypes(include=['object','category']).columns
num=x.select_dtypes(exclude=['object','category']).columns

pre=ColumnTransformer([
    ("cat",OneHotEncoder(handle_unknown='ignore'),cat),
    ('num',StandardScaler(),num)
])

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

xtrain=pre.fit_transform(xtrain)
xtest=pre.transform(xtest)

models={
    'Logistic regression':LogisticRegression(max_iter=1000),
    'Decision tree':DecisionTreeClassifier(random_state=42)
}

for name,model in models.items():
  model.fit(xtrain,ytrain)
  pred=model.predict(xtest)

  print("\n", name)
  print("Accuracy:", accuracy_score(ytest, pred))
  print("Classification report:", classification_report(ytest, pred))

#11

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris=load_iris()
x=iris.data[:,:2]
y=(iris.target==0).astype(int)

scaler=StandardScaler()
x=scaler.fit_transform(x)

model=SVC(kernel='linear')
model.fit(x,y)

plt.scatter(x[:,0],x[:,1],c=y,cmap='coolwarm')

w=model.coef_[0]
b=model.intercept_[0]

X=x[:,0]
boundary=-(w[0]*X+b)/w[1]
margin1=-(w[0]*X+b-1)/w[1]
margin2=-(w[0]*X+b+1)/w[1]

plt.plot(X,boundary,'k-')
plt.plot(X,margin1,'k--')
plt.plot(X,margin2,'k--')
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.title("Linear SVM - Iris Classification")
plt.show()

#12


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv(r'C:\Users\ASUS\Downloads\archive\fashion-mnist_train.csv')
test_df = pd.read_csv(r'C:\Users\ASUS\Downloads\archive\fashion-mnist_test.csv')

X_train = train_df.drop('label', axis=1) / 255.0
y_train = train_df['label']

X_test = test_df.drop('label', axis=1) / 255.0
y_test = test_df['label']

models = {
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=3),
    "RBF": SVC(kernel="rbf")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))



#13


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = {
    "One hidden layer - 10 neurons": (10,),
    "One hidden layer - 50 neurons": (50,),
    "Two hidden layers": (50, 30),
    "Three hidden layers": (100, 50, 25)
}

for name, layers in architectures.items():
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))


#14


import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

activations = ["logistic", "relu", "tanh"]

for act in activations:
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation=act,
        max_iter=20,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    pred = model.predict(X_test)

    print("\nActivation:", act)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Training time:", round(end - start, 2), "seconds")
    print("Iterations:", model.n_iter_)

    plt.plot(model.loss_curve_, label=act)

plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Convergence for Different Activation Functions")
plt.legend()
plt.show()

#15


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

x = data.data[:5000]
y = data.target[:5000].astype(int)

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [50, 100]

results = {}

for lr in learning_rates:
    for batch in batch_sizes:
        for ep in epochs:
            model = MLPClassifier(
                hidden_layer_sizes=(100,),
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=ep,
                random_state=42
            )

            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)

            acc = accuracy_score(ytest, ypred)
            key = f"LR={lr}, Batch={batch}, Epochs={ep}"
            results[key] = acc
            print(key, "Accuracy =", round(acc, 4))



plt.figure(figsize=(12, 5))
plt.bar(results.keys(), results.values())
plt.title("Fashion MNIST Hyperparameter Comparison")
plt.xlabel("Hyperparameter Settings")
plt.ylabel("Accuracy")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()