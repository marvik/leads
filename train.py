import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
leads = pd.read_csv("Leads.csv")

# --- Data Preparation and Cleaning ---

# Handle 'Select' values by replacing them with NaN
leads.replace('Select', np.nan, inplace=True)

# Drop columns with a high percentage of missing values (more than 30%)
missing_percentage = leads.isnull().sum() / len(leads) * 100
cols_to_drop = missing_percentage[missing_percentage > 30].index
leads.drop(cols_to_drop, axis=1, inplace=True)

# Impute missing values in numerical columns with the median
numerical_cols = leads.select_dtypes(include=np.number).columns
for col in numerical_cols:
    leads[col] = leads[col].fillna(leads[col].median())

# Impute missing values in categorical columns with the mode
categorical_cols = leads.select_dtypes(include=object).columns
for col in categorical_cols:
    leads[col] = leads[col].fillna(leads[col].mode()[0])

# --- Feature Engineering and Encoding ---

# Encode categorical variables
leads_encoded = pd.get_dummies(leads, columns=categorical_cols, drop_first=True)

# --- Model Training ---

# Separate features and target variable
X = leads_encoded.drop('Converted', axis=1)
y = leads_encoded['Converted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# --- Save the Model ---

# Save the trained model to a pickle file
with open('model.bin', 'wb') as f_out:
    pickle.dump(model, f_out)

print("Model training complete and saved as 'model.bin'")
