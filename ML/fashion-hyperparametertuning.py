
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
