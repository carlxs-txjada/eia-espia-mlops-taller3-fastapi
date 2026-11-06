import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 1. Cargar datos
data = pd.read_csv('medical_insurance.csv')
X = data.drop('charges', axis=1)
y = data['charges']

# 2. Definición de características
categorical_features = ['sex', 'region', 'smoker']
numerical_features = [col for col in X.columns if col not in categorical_features]

# 3. Preprocesamiento: StandardScaler para numéricas, OneHotEncoder para categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# 4. Modelo con hiperparámetros
model = RandomForestRegressor(
    n_estimators=178,
    max_depth=23,
    min_samples_split=7,
    min_samples_leaf=1,
    random_state=42
)

# 5. Integración en Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# 6. Entrenamiento completo
pipeline.fit(X, y)

# 7. Serialización
joblib.dump(pipeline, 'pipeline.joblib')
print("Pipeline entrenado y guardado como 'pipeline.joblib'")

