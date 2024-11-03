# base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import streamlit as st

# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline


# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from category_encoders import TargetEncoder

# for model learning
from sklearn.model_selection import train_test_split

#models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error



st.write("""# Ислледование: Цены на жилье - Передовые методы регрессии""")

st.write("""## Цель:
        Предсказать цену продажи каждого дома.
    Для каждого идентификатора в тестовом наборе вы должны предсказать значение переменной SalePrice""")


st.write("""##  Метрика:
        Оценивается по среднеквадратичнойошибке (RMLSE и R2)
    между логарифмом прогнозируемого значения и логарифмом наблюдаемой цены продажи""")


file = st.sidebar.file_uploader("Загрузите CSV-файл", type="csv")
if file is not None:
    test_df = pd.read_csv(file)
    st.write("Необработанные данные")
    st.write(test_df.head(5))
else:
    st.stop()


train_df = pd.read_csv('Data/train.csv')


nan_info = pd.DataFrame(data={'NaN_count': train_df.isna().sum(), 'data_type':train_df.dtypes})
nan_info = nan_info[nan_info['NaN_count']>0]
nan_info = nan_info.reset_index()

# Feature engineering
train_df['houseage'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['houseage'] = test_df['YrSold'] - test_df['YearBuilt']

train_df['houseremodelage'] = train_df['YrSold'] - train_df['YearRemodAdd']
test_df['houseremodelage'] = test_df['YrSold'] - test_df['YearRemodAdd']

train_df['totalsf'] = train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2']
test_df['totalsf'] = test_df['1stFlrSF'] + test_df['2ndFlrSF'] + test_df['BsmtFinSF1'] + test_df['BsmtFinSF2']

train_df['totalarea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
test_df['totalarea'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']

train_df['totalporchsf'] = train_df['OpenPorchSF'] + train_df['3SsnPorch'] + train_df['EnclosedPorch'] + train_df['ScreenPorch']
test_df['totalporchsf'] = test_df['OpenPorchSF'] + test_df['3SsnPorch'] + test_df['EnclosedPorch'] + test_df['ScreenPorch']

train_df['totalbaths'] = train_df['BsmtFullBath'] + train_df['FullBath'] + 0.5 * (train_df['BsmtHalfBath'] + train_df['HalfBath'])
test_df['totalbaths'] = test_df['BsmtFullBath'] + test_df['FullBath'] + 0.5 * (test_df['BsmtHalfBath'] + test_df['HalfBath'])



label = 'SalePrice'
X = train_df.drop(columns=[label])  # все признаки, кроме целевого
y = train_df[label]                 # целевая переменная

# Разделение на тренировочную и тестовую выборки
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)




drop_features = ['Alley', 'MasVnrType', 'MasVnrArea', 'Id', 'YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1',
                  'BsmtFinSF2', 'GrLivArea', 'TotalBsmtSF', 'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF',
                      '3SsnPorch', 'EnclosedPorch', 'ScreenPorch'] # Эти столбцы считаем не нужными и выкидываем
na_features = nan_info.loc[nan_info['data_type'] == 'object', 'index'].tolist()
na_features = [x for x in na_features if x not in ['Electrical', 'MasVnrType', 'Alley', 'Id']] 

# Начинаем создавать наш Pipeline
imputer = ColumnTransformer(
    transformers = [
        ('drop_features', 'drop', drop_features),
        ('num_imputer', SimpleImputer(strategy='median'), ['GarageYrBlt', 'LotFrontage']), 
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='No'), na_features),
        ('el_imputer', SimpleImputer(strategy='most_frequent'), ['Electrical'])
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough',
    force_int_remainder_cols=False
)    
filled_data = imputer.fit_transform(X_train)
filled_data_valid = imputer.transform(X_valid)


# test_df = imputer.transform(test_df)

# #Фильтруем выбросы
# lower_quantile = 0.01
# upper_quantile = 0.99

# # Проходим по всем столбцам и отфильтровываем строки
# for col in filled_data.columns:
#     if pd.api.types.is_numeric_dtype(filled_data[col]):
#         lower_threshold = filled_data[col].quantile(lower_quantile)
#         upper_threshold = filled_data[col].quantile(upper_quantile)
#         filled_data = filled_data[(filled_data[col] >= lower_threshold) & (filled_data[col] <= upper_threshold)]


# Проходим по всем столбцам и отфильтровываем строки для тестового файла       
# for col in test_df.columns:
#     if pd.api.types.is_numeric_dtype(test_df[col]):
#         lower_threshold = test_df[col].quantile(lower_quantile)
#         upper_threshold = test_df[col].quantile(upper_quantile)
#         test_df = test_df[(test_df[col] >= lower_threshold) & (test_df[col] <= upper_threshold)]


# # Определение целевой переменной и признаков
# label = 'SalePrice'
# X = filled_data.drop(columns=[label])  # все признаки, кроме целевого
# y = filled_data[label]                 # целевая переменная

# # Разделение на тренировочную и тестовую выборки
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Кодирование данных

category_col = filled_data.select_dtypes(include='object').drop('CentralAir', axis=1)
target_encoder_cat = np.array(category_col.columns)

encoder = ColumnTransformer(
    [

        ('central_air', OrdinalEncoder(), ['CentralAir']),
        ('target_encoder', TargetEncoder(), target_encoder_cat)

    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough',
    force_int_remainder_cols=False
)
encoded_data = encoder.fit_transform(filled_data, y_train)
encodded_data_valid = encoder.transform(filled_data_valid)


standard_scaler_columns = encoded_data.columns.to_list()

scaler = ColumnTransformer(
    [
        ('scaling_num_columns', StandardScaler(), standard_scaler_columns)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough',
    force_int_remainder_cols=False
)
processed_data = scaler.fit_transform(encoded_data)
processed_data_valid = scaler.transform(encodded_data_valid)

# Собираем пайплайн
cb = CatBoostRegressor()

preprocessor_cb = Pipeline(
    [
        ('imputer', imputer),
        ('encoder', encoder),
        ('scaler', scaler),
        ('model', cb)
    ]
)

rf = RandomForestRegressor()

preprocessor_rf = Pipeline(
    [
        ('imputer', imputer),
        ('encoder', encoder),
        ('scaler', scaler),
        ('model', rf)
    ]
)

l = DecisionTreeRegressor()

preprocessor_l = Pipeline(
    [
        ('imputer', imputer),
        ('encoder', encoder),
        ('scaler', scaler),
        ('model', l)
    ]
)

bg = BaggingRegressor()

preprocessor_bg = Pipeline(
    [
        ('imputer', imputer),
        ('encoder', encoder),
        ('scaler', scaler),
        ('model', bg)
    ]
)


preprocessor_cb.fit(X_train, np.log(y_train))

y_preds = preprocessor_cb.predict(X_valid)

RMLSE_cb = np.sqrt(np.mean((np.log(y_valid) - y_preds) ** 2))
RMLSE_KAGGLE_cb = 0.12511
R2_cb = 1 - (((y_valid - np.exp(y_preds))**2).sum() / ((y_valid - y_valid.mean())**2).sum())




preprocessor_rf.fit(X_train, np.log(y_train))

y_preds = preprocessor_rf.predict(X_valid)

RMLSE_rf = np.sqrt(np.mean((np.log(y_valid) - y_preds) ** 2))
RMLSE_KAGGLE_rf = 0.14308
R2_rf = 1 - (((y_valid - np.exp(y_preds))**2).sum() / ((y_valid - y_valid.mean())**2).sum())


preprocessor_bg.fit(X_train, np.log(y_train))

y_preds = preprocessor_bg.predict(X_valid)

RMLSE_bg = np.sqrt(np.mean((np.log(y_valid) - y_preds) ** 2))
RMLSE_KAGGLE_bg = 0.15146
R2_bg = 1 - (((y_valid - np.exp(y_preds))**2).sum() / ((y_valid - y_valid.mean())**2).sum())




preprocessor_l.fit(X_train, np.log(y_train))

y_preds = preprocessor_l.predict(X_valid)

RMLSE_l = np.sqrt(np.mean((np.log(y_valid) - y_preds) ** 2))
RMLSE_KAGGLE_l = 0.19523
R2_l = 1 - (((y_valid - np.exp(y_preds))**2).sum() / ((y_valid - y_valid.mean())**2).sum())



kaggle_index = test_df['Id']

kaggle_preds = preprocessor_cb.predict(test_df)
kaggle_normal = np.exp(kaggle_preds)
kaggle_pred_df = pd.DataFrame(kaggle_normal, columns=['SalePrice'])
concat_kaggle_df = pd.concat([kaggle_index, kaggle_pred_df], axis=1)

csv_data = concat_kaggle_df.to_csv(index=False).encode('utf-8')



button = st.button("Вывести результаты")
if button:
    st.write(concat_kaggle_df)
else:
    st.stop()

data = {
    'RMLSE': [RMLSE_cb, RMLSE_rf, RMLSE_bg, RMLSE_l],
    'RMLSE-Kaggle': [RMLSE_KAGGLE_cb, RMLSE_KAGGLE_rf, RMLSE_KAGGLE_bg, RMLSE_KAGGLE_l],
    'R2': [R2_cb, R2_rf, R2_bg, R2_l],
}
    
table = pd.DataFrame(data, index=['CatBoostRegressor', 'RandomForestRegressor', 'BaggingRegressor', 'DecisionTreeRegressor'])

st.write("### Таблица с метриками и регрессорами")

# Вывод таблицы в Streamlit
st.dataframe(table, width=600, height=300)

st.image('results.png')


button_download = st.sidebar.download_button("Сохранить результаты лучшей регрессии", data=csv_data, file_name='submission.csv', mime="text/csv")

