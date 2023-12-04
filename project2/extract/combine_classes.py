import pandas as pd 


def read_data(X_path):  
    X = pd.read_csv(X_path)
    return X

X1 = read_data(X_path="data/out.csv")
X2 = read_data(X_path="data/out_3class.csv")

X1["y"][X2["y"] == 0] = 1
print(X1)
print(X1["id"].nunique())
X1.to_csv("data/out_combined.csv", index=False)