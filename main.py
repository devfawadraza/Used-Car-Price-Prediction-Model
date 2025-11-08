import pandas as pd

df = pd.read_csv("data/file.csv")

# basic overview
# print("shape of dataset", df.shape) #rows x columns 
print("Columns name in dataset", df.columns)