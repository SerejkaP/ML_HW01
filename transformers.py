from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np
import re


def mileage_to_kmpl(mileage: str) -> float:
    value, measure = mileage.split(' ')
    val = np.nan
    if measure == 'kmpl':
        val = float(value)
    else:
        val = float(value) * (1000 / 760)
    if val == 0:
        return np.nan
    return val


def to_torque_value(torque: str, value: str) -> float:
    is_kgm = re.findall(r'kgm', torque, flags=re.IGNORECASE)
    if is_kgm:
        return float(value) * 9.80665
    else:
        return float(value)


def to_rpm_value(torque: str, values: list[str]) -> float:
    # rpm считаю как среднее
    with_err = re.findall(r'\+\s*/\s*-', torque)
    values_count = len(values)
    if values_count < 1:
        return np.nan
    elif with_err:
        return float(values[0])
    else:
        rpms = 0
        for rpm in values:
            rpms = rpms + float(rpm)
        return rpms / values_count


def torque_to_values(df):
    torque_values = []
    rpm_values = []
    for row in df['torque']:
        torque_value = np.nan
        rpm_value = np.nan
        if isinstance(row, str):
            str_row = row.replace(',', '')
            values = re.findall(r'[\d.]+', str_row)
            torque_value = to_torque_value(str_row, values[0])
            rpm_value = to_rpm_value(str_row, values[1:])
        torque_values.append(torque_value)
        rpm_values.append(rpm_value)

    df['torque_value'] = torque_values
    df['rpm_value'] = rpm_values


class StrToNum(BaseEstimator, TransformerMixin):
    def set_output(self, transform):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Выделение числовых признаков
        X = X.copy()
        X['mileage'] = X['mileage'].map(lambda m: mileage_to_kmpl(m) if isinstance(m, str) else m)
        X['engine'] = X['engine'].map(lambda m: int(m.split(' ')[0]) if isinstance(m, str) else m)
        X['max_power'] = X['max_power'].map(
            lambda m: (np.nan if (len(m.split(' ')[0]) == 0 or m == '0') else float(m.split(' ')[0])) if isinstance(m,
                                                                                                                    str) else m)
        torque_to_values(X)
        X.drop('torque', axis=1, inplace=True)
        return X

    def predict(self, X):
        return self.transform(X)


def get_car_make(car_name: str) -> str:
    return car_name.split(' ')[0]


class CarNameToMaker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['car_maker'] = X['name'].map(lambda n: get_car_make(n))
        X.drop('name', axis=1, inplace=True)
        return X

    def predict(self, X):
        return self.transform(X)


class CategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = TargetEncoder(cols=['car_maker'])
        self.ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.ohe.set_output(transform='pandas')

    def fit(self, X, y=None):
        y = y.copy()
        self.encoder.fit(X['car_maker'], y)
        self.ohe.fit(X[['fuel', 'seller_type', 'transmission', 'owner', 'seats']], y)
        return self

    def transform(self, X):
        X = X.copy()
        X['car_maker_encoded'] = self.encoder.transform(X['car_maker'])['car_maker']
        X.drop('car_maker', axis=1, inplace=True)
        upd = self.ohe.transform(X[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
        X = pd.concat([X, upd], axis=1)
        X.drop(['fuel', 'seller_type', 'transmission', 'owner', 'seats'], axis=1, inplace=True)
        return X

    def predict(self, X):
        return self.transform(X)


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X[X.select_dtypes(include='number').columns])
        self.scaler.fit(X[X.select_dtypes(include='number').columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[X.select_dtypes(include='number').columns] = self.imputer.transform(
            X[X.select_dtypes(include='number').columns])
        X['engine'] = X['engine'].astype(int)
        X['seats'] = X['seats'].astype(int)
        X[X.select_dtypes(include='number').columns] = self.scaler.transform(
            X[X.select_dtypes(include='number').columns])
        return X

    def predict(self, X):
        return self.transform(X)
