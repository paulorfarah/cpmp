from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Build a synthetic dataset
# X, y = make_classification(
#     n_samples=1000, n_features=5, n_informative=4, n_redundant=1, n_classes=4
# )

X, y = make_multilabel_classification(n_features=20, n_classes=2)
print(y)

# print(X.data)

# Train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1121218
)

# Fit/predict
etc = ExtraTreesClassifier()
_ = etc.fit(X_train, y_train)
y_pred = etc.predict(X_test)

# parameters = {
#         'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=200)],
#         'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
#     }
# c = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, n_jobs=-1)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 5))
# y_test.values.argmax(axis=1), predictions.argmax(axis=1))


cm = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
print(cm)
cmp = ConfusionMatrixDisplay(
    confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1)),
    # display_labels=["class_1", "class_2", "class_3", "class_4"],
)

cmp.plot(ax=ax)
plt.show()