
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
