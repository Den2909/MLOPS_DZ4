import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ4/datasets')
data_csv_path_X_train = os.path.join(datasets_dir, 'data_X_train.csv')
data_csv_path_y_train = os.path.join(datasets_dir, 'data_y_train.csv')
X_train = pd.read_csv(data_csv_path_X_train)
y_train = pd.read_csv(data_csv_path_y_train)

gbdt_clf = XGBClassifier()
gbdt_clf.fit(X_train, y_train)


# Сохраняем обученную модель
model_output_path = os.path.expanduser('~/Projects/MLOPS_DZ4/models/model.pkl')
with open(model_output_path, "wb") as model_file:
    pickle.dump(gbdt_clf, model_file)

print("Модель сохранена по пути:", model_output_path)