
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
