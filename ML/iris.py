from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, :2]
y = (iris.target == 0).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = SVC(kernel="linear")
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")

w = model.coef_[0]
b = model.intercept_[0]

x = X[:, 0]
boundary = -(w[0] * x + b) / w[1]
margin1 = -(w[0] * x + b - 1) / w[1]
margin2 = -(w[0] * x + b + 1) / w[1]

plt.plot(x, boundary, "k-")
plt.plot(x, margin1, "k--")
plt.plot(x, margin2, "k--")

plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.title("Linear SVM - Iris Classification")
plt.show()
