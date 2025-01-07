# //  ████████      ██          ██          ████████ 
# //  ██            ██          ██          ██       
# //  ██  ████      ██          ██          ██  ████
# //  ██            ██          ██          ██       
# //  ████████      ████████    ████████    ████████ 

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Simple EDA
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print("Sample Data:")
print(df.head())

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define parameter grid (modify as needed)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 2, 4, 6],
    'min_samples_split': [2, 5, 10]
}

# Create model
rf = RandomForestClassifier(random_state=42)

# StratifiedKFold to keep class balance across folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

# Fit on training
grid_search.fit(X_train, y_train)

# Best params found
print("Best Params:", grid_search.best_params_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", acc)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
