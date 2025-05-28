import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Leer el CSV
df = pd.read_csv('citas_oftalmologia.csv')
df['fecha'] = pd.to_datetime(df['fecha'])
df.set_index('fecha', inplace=True)

# Serie temporal
serie = df['citas_oftalmologia']

# Visualizar datos históricos
plt.figure(figsize=(12, 6))
plt.plot(serie, label='Citas históricas')
plt.title('Citas Oftalmología - Histórico')
plt.xlabel('Fecha')
plt.ylabel('Número de citas')
plt.legend()
plt.show()

# Aplicar Holt-Winters (tendencia + estacionalidad semanal)
modelo = ExponentialSmoothing(serie, trend='add', seasonal='add', seasonal_periods=7)
modelo_fit = modelo.fit()

# Predecir los próximos 14 días
predicciones = modelo_fit.forecast(14)

# Visualizar resultados
plt.figure(figsize=(12, 6))
plt.plot(serie, label='Histórico')
plt.plot(predicciones, label='Predicción', linestyle='--')
plt.title('Predicción de Citas Oftalmología (Holt-Winters)')
plt.xlabel('Fecha')
plt.ylabel('Número de citas')
plt.legend()
plt.show()

# Mostrar predicciones
print(predicciones)
