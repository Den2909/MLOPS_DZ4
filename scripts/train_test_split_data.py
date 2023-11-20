import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


# Загружаем данные
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ4/datasets')
data_csv_path = os.path.join(datasets_dir, 'data_proc.csv')
df = pd.read_csv(data_csv_path)

# Разделяем датасет на признаки и таргет

columns_list = list(df)
columns_list.remove("churn")

X, y = df[columns_list].copy(), df["churn"].copy()

# Создаём тренировочную и тестовую части датасета

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Записываем train и test данные в файлы
data_csv_path_X_train = os.path.join(datasets_dir, 'data_X_train.csv')
data_csv_path_X_test = os.path.join(datasets_dir, 'data_X_test.csv')
data_csv_path_y_train = os.path.join(datasets_dir, 'data_y_train.csv')
data_csv_path_y_test = os.path.join(datasets_dir, 'data_y_test.csv')

X_train.to_csv(data_csv_path_X_train, index=False)
X_test.to_csv(data_csv_path_X_test, index=False)
y_train.to_csv(data_csv_path_y_train, index=False)
y_test.to_csv(data_csv_path_y_test, index=False)


print(f'Соотношение данных: Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')
print("Датасет X_train сохранен по пути:", data_csv_path_X_train)
print("Датасет X_test сохранен по пути:", data_csv_path_X_test)
print("Датасет y_train сохранен по пути:", data_csv_path_y_train)
print("Датасет y_test сохранен по пути:", data_csv_path_y_test)
