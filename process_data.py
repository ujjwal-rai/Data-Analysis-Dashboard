import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Paths and task type
dataset_path, features_path, task = sys.argv[1], sys.argv[2], sys.argv[3]

# Load datasets
dataset = pd.read_csv(dataset_path)
features_desc = pd.read_csv(features_path)

# Clean and prepare data
dataset = dataset.dropna()  
dataset = pd.get_dummies(dataset, drop_first=True)  # Convert strings to numeric if necessary

# Separate target variable (last column) and features
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run model based on selected task
if task == 'regression':
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    print("Regression Model Mean Squared Error:", error)

elif task == 'classification':
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

# Save model
with open('model/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Plot feature relationships and save
plt.figure(figsize=(10, 6))
pd.plotting.scatter_matrix(X, alpha=0.2, figsize=(9, 9), diagonal='hist')
plt.savefig("model/features_plot.png")
