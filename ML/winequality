
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
