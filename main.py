import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/refined_file.csv")
# df = df[df["price"] != 0]
# df = df[df["vehicle_age"] != 0]

# basic overview
# print("shape of dataset", df.shape) #rows x columns 
print("Columns names in dataset", df.columns)
# print("data types", df.dtypes)

# preview data
print("\nfirst 5 rows", df.head())

# print("\n last 5 rows", df.tail())

# check missing values

# print("Missing values per column", df.isnull().sum())

# duplicates values

# print("number of duplicate rows", df.duplicated().sum())

# summary stats for numeric columns
# print("summary stats:\n", df.describe())

# summary stats for object column
# print("summary stats:\n", df.describe(include='object'))

# unique values per column

# for col in df.columns:
#     print(f"{col} - Unique values {df[col].nunique()}")

# check correaltion for numeric columns

# print("Correlation Matrix \n:", df.select_dtypes(include=[np.number]).corr())

# print(df['title'].head(20))
# print(df['title'].dtype)


# =======================================================================================

zero_counts = (df == 0).sum()

column_with_zeros = zero_counts[zero_counts > 0]

print("\n Columns with zero values")
print(column_with_zeros)

# ===========================================================================

# sns.histplot(df['price'], kde=True)
# plt.title('Car Price Distribution')
# plt.show()

# sns.boxplot(x='fuel_type', y='price', data=df)
# plt.title('Fuel Type vs Price')
# plt.show()


# ============================================================================
# Label encoder
le = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df [col] = le.fit_transform(df[col])

print ("\n Data types after encoding")
print(df.dtypes.head())

# ============================================================================

# Train test split
X = df.drop('price', axis=1)
# target columns output
y = df['price']
 # split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split completed.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)    