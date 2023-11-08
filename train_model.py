import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Imports para regressão logística
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
# Import para gerar arquivo pkl
import joblib

df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
df.drop(columns="id")


def tratar_dados(df):
    df = df.drop(columns='id')
    df['age'] = df['age'].apply(lambda x: round(x))
    # substituindo os bmi's maiores que 50 e menores que 15 por não haver muitas
    # na prática pessoas com estes valores de IMC
    df['bmi'] = df['bmi'].apply(lambda bmi_value: bmi_value if 15 <
                                bmi_value < 50 else np.nan)
    # preenchendo o índice de massa corporal pela média da idade da entrada
    # para aqueles casos em que temos NaN
    mean_values = df.groupby('age')['bmi'].transform('mean')
    df['bmi'].fillna(mean_values, inplace=True)
    # transformando as colunas object em colunas int64 para ver a correlaçao
    encoded_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    label_encoder = LabelEncoder()
    for col in encoded_cols:
        df[col] = label_encoder.fit_transform(df[col])

    return df


df = tratar_dados(df)

class_majority = df[df['stroke'] == 0]
class_minority = df[df['stroke'] == 1]

# Fazer downsampling das classes majoritárias
class_majority_downsampled = resample(
    class_majority, replace=False, n_samples=len(class_minority), random_state=42)

# Combinar as classes majoritárias e minoritárias após o downsampling
data_downsampled = pd.concat([class_majority_downsampled, class_minority])

# Separar novamente as variáveis independentes (X) e a variável dependente (y) após o downsampling
X_downsampled = data_downsampled.drop(columns='stroke')
y_downsampled = data_downsampled['stroke']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_downsampled, y_downsampled, test_size=0.2, random_state=42)

# Padronizar as features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar um modelo de regressão logística com penalidade L2
model = LogisticRegression(penalty='l2', random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, './trained_logistic_regression_model.pkl')
