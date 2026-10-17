import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("train.csv")

print("Original Dataset:")
print(df.head())

# Step 2: Check dataset information
print("\nDataset Information:")
df.info()

# Step 3: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Step 5: Remove duplicate rows
df = df.drop_duplicates()

# Step 6: Handle missing values
# Fill missing Age values with the median
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

# Fill missing Fare values with the median
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Step 7: Convert data types if required
if "Survived" in df.columns:
    df["Survived"] = df["Survived"].astype(int)

# Step 8: Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Step 9: Save the cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

# Step 10: Display final information
print("\nCleaned Dataset:")
print(df.head())

print("\nFinal Dataset Shape:", df.shape)

print("\nData cleaning completed successfully!")
print("Cleaned dataset saved as cleaned_dataset.csv")