import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# import mlflow
# import mlflow.sklearn


# Инициализация MLflow
# mlflow.set_tracking_uri("http://localhost:5000")  # Укажите адрес сервера MLflow
# mlflow.set_experiment("train_model")
# mlflow.start_run()
# Загружаем данные
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ4/datasets')
data_csv_path_X_train = os.path.join(datasets_dir, 'data_X_train.csv')
data_csv_path_y_train = os.path.join(datasets_dir, 'data_y_train.csv')
X_train = pd.read_csv(data_csv_path_X_train)
y_train = pd.read_csv(data_csv_path_y_train)

# Задаём значения параметров для перебора

parameters = {"n_estimators": [40, 50, 60, 70, 80, 90, 100],
              "max_depth": [1, 2, 3, 4, 5],
              "learning_rate": [0.1, 0.2, 0.3]}

# Инициализируем классификатор XGBClassifier

gbdt_clf = XGBClassifier()

# Инициализируем перебор параметров

gscv = GridSearchCV(gbdt_clf, parameters, n_jobs=-1, cv=5)

# Запускаем перебор параметров

gscv.fit(X_train, y_train)

# Выводим наилучшие значения параметров для текущей модели

print("Наилучшие значения параметров: {}".format(gscv.best_params_),
      "Лучший результат на кросс-валидации: {:.2f}%".format(gscv.best_score_ * 100),
      sep="\n")


# Сохраняем обученную модель
model_output_path = os.path.expanduser('~/Projects/MLOPS_DZ4/models/model.pkl')
with open(model_output_path, "wb") as model_file:
    pickle.dump(gscv, model_file)

# Логируем модель в MLflow
#with mlflow.start_run():
    #mlflow.sklearn.log_model(forecaster, "forecast_model")
    #mlflow.log_params({"seasonality": SEASON})

# Завершение MLflow run
#mlflow.end_run()
print("Модель сохранена по пути:", model_output_path)