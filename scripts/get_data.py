import pandas as pd
import os

# URL для файла train.csv
url_train = 'https://drive.google.com/file/d/1fFzjZl11aUAkVf9_vO9AN2R9c9CdK9Hq/view?usp=sharing'
file_id_train = url_train.split('/')[-2]
url = 'https://drive.google.com/uc?id=' + file_id_train
# Указываем путь для сохранения файла
datasets_dir = os.path.expanduser('~/Projects/MLOPS_DZ4/datasets')
data_csv_path = os.path.join(datasets_dir, 'data.csv')

# Загружаем датасет
df = pd.read_csv(url)

# Сохранение данных
df.to_csv(data_csv_path, index=False)

print("Датасет успешно загружен и сохранен по пути:", data_csv_path)
