
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = pd.read_excel("Online Retail.xlsx")

data = data.dropna(subset=['CustomerID'])
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
data = data[~data['InvoiceNo'].astype(str).str.startswith('C')]

data['Amount'] = data['Quantity'] * data['UnitPrice']

customer = data.groupby('CustomerID').agg(
    TotalSpend=('Amount', 'sum'),
    TotalQuantity=('Quantity', 'sum'),
    NumInvoices=('InvoiceNo', 'nunique'),
    AvgOrderValue=('Amount', 'mean')
).reset_index()

customer['Segment'] = pd.qcut(
    customer['TotalSpend'], q=3,
    labels=['Low', 'Medium', 'High'],
    duplicates='drop'
)

X = customer[['TotalQuantity', 'NumInvoices', 'AvgOrderValue']]
y = customer['Segment']

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=42
)
model.fit(xtrain, ytrain)

pred = model.predict(xtest)
print("Accuracy:", accuracy_score(ytest, pred))

plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns,
          class_names=model.classes_,
          filled=True, rounded=True)
plt.show()

importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind='bar', title='Feature Importance')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
