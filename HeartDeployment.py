import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("heart.csv")

print(df.head())
print(df.info())
print("Shape before dropping NA:", df.shape)

df = df.dropna()
print("Shape after dropping NA:", df.shape)
print("Null values present?", df.isnull().values.any())

# Target class distribution
df["target"].value_counts().plot(kind='bar', color=["salmon", "lightblue"])
plt.title("Target Class Distribution")
plt.show()

# Feature and target selection
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier.fit(X_train, y_train)

y_pred_svm = svm_classifier.predict(X_test)

print("\n===== SVM Classification Report =====")
print(classification_report(y_test, y_pred_svm))
print("SVM Accuracy on training set: {:.2f}".format(svm_classifier.score(X_train, y_train)))
print("SVM Accuracy on test set: {:.2f}".format(svm_classifier.score(X_test, y_test)))

# Save SVM model
pickle.dump(svm_classifier, open('heart-SVM-model.pkl', 'wb'))
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_pred_lr = linear_regressor.predict(X_test)

# Convert continuous predictions to class labels (0 or 1)
y_pred_lr_class = np.where(y_pred_lr >= 0.5, 1, 0)

print("\n===== Linear Regression (as classifier) Report =====")
print(classification_report(y_test, y_pred_lr_class))
print("Linear Regression Accuracy on training set: {:.2f}".format(linear_regressor.score(X_train, y_train)))
print("Linear Regression Accuracy on test set: {:.2f}".format(linear_regressor.score(X_test, y_test)))

# Also check regression metrics (optional)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Save Linear Regression model
pickle.dump(linear_regressor, open('heart-LR-model.pkl', 'wb'))
