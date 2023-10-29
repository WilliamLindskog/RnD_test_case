import pandas as pd

df_train = pd.read_csv('./data/TrainOnMe.csv', index_col=0)
df_test = pd.read_csv('./data/PredictOnMe.csv', index_col=0)

# remove NaNs
df_train = df_train.dropna()

# check unique values in x4 column
print(len(df_train['x4'].unique()))

# drop x4 column
df_train = df_train.drop('x4', axis=1)
df_test = df_test.drop('x4', axis=1)

cat_code_dict = {}
# encode categorical features
for col in df_train.columns:
    if df_train[col].dtype == 'object':
        df_train[col] = df_train[col].astype('category')
        cat_code_dict[col] = {i: cat for i, cat in enumerate(df_train[col].cat.categories)}
        df_train[col] = df_train[col].cat.codes
        if col != 'y':
            df_test[col] = df_test[col].astype('category')
            df_test[col] = df_test[col].cat.codes
    else:
        # normalize numerical features
        df_train[col] = (df_train[col] - df_train[col].mean()) / df_train[col].std()

# save to csv
df_train.to_csv('./data/train_preprocessed.csv')
df_test.to_csv('./data/test_preprocessed.csv')

print(cat_code_dict)

# save cat_code_dict to json
import json
with open('./data/cat_code_dict.json', 'w') as fp:
    json.dump(cat_code_dict, fp)