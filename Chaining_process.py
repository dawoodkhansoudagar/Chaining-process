# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Sample dataset (Loan Approval type)
data = {
    'Age': [25, 35, 45, 32, 23, 52, 43],
    'Income': [30000, 50000, 70000, 48000, 32000, 90000, 60000],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'Loan_Status': [0, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Split features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Separate numeric and categorical columns
numeric_features = ['Age', 'Income']
categorical_features = ['Gender']

# Numeric pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Final Pipeline (Chaining Process)
model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression())
])

# Train model
model_pipeline.fit(X_train, y_train)

# Prediction
predictions = model_pipeline.predict(X_test)

# Output results
print("Predictions:", predictions)

# Accuracy
accuracy = model_pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)

input("Press Enter to exit...")