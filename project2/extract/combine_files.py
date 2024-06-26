import pandas as pd 


def read_data(X_train_path, y_train_path, X_test_path, extract_data, header=1):  
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:,1]

    train_ids, test_ids = pd.DataFrame(list(range(0, X_train.shape[0]))), \
            pd.DataFrame(list(range(0, X_test.shape[0])))
    
    return X_train, y_train, train_ids, X_test, test_ids

def combine_two(X_train1, X_train2, path: str):
    df_train = pd.concat([X_train1, X_train2], axis=1, join='inner')
    df_train = df_train.loc[:,~df_train.columns.duplicated()]
    df_train.to_csv(path, index=False)
    print(df_train.shape)
    print(df_train.columns)

X_train_path, y_train_path, X_test_path = "data/train_combined.csv", "data/y_train.csv", "data/test_combined.csv"
X_train1, y_train1, train_ids1, X_test1, test_ids1 = read_data(X_train_path, y_train_path, X_test_path, False)
# X_train1.drop([f"r{r}" for r in range(0, 160)], axis=1, inplace=True)
# X_test1.drop([f"r{r}" for r in range(0, 201)], axis=1, inplace=True)

# X_train1.drop([f"r{r}" for r in range(0, 6)], axis=1, inplace=True)
# X_test1.drop([f"r{r}" for r in range(0, 7)], axis=1, inplace=True)

print(X_train1.columns)
print(X_test1.columns)

X_train_path, y_train_path, X_test_path = "data/X_train_cnn_64_best.csv", "data/y_train.csv", "data/X_test_cnn_64_best.csv"
X_train2, y_train2, train_ids2, X_test2, test_ids2 = read_data(X_train_path, y_train_path, X_test_path, False)
# X_train2.drop([f"r{r}" for r in range(0, 160)], axis=1, inplace=True)
# X_test2.drop([f"r{r}" for r in range(0, 201)], axis=1, inplace=True)
print(X_train2.columns)
print(X_test2.columns)

combine_two(X_train1, X_train2, path="data/train_combined_cnn_64.csv")
combine_two(X_test1, X_test2, path="data/test_combined_cnn_64.csv")
range(0, 64)

