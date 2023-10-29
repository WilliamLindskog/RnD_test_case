import pandas as pd
import xgboost as xgb

# import random forest from sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load data
df_train = pd.read_csv('./data/train_preprocessed.csv', index_col=0)
df_test = pd.read_csv('./data/test_preprocessed.csv', index_col=0)

# Split data
X_train = df_train.drop('y', axis=1)
y_train = df_train['y']

# Train model and print score
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))

model = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))

# predict on test data
y_pred = model.predict(df_test)

# Get cad_code_dict
import json
with open('./data/cat_code_dict.json', 'r') as fp:
    cat_code_dict = json.load(fp)

y_pred = [cat_code_dict['y'][str(i)] for i in y_pred]

# save to csv
pd.DataFrame(y_pred).to_csv('./data/predictions.csv')
