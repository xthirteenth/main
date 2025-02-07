import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n_samples = 100

area = np.random.uniform(30, 150, n_samples)

rooms = np.random.randint(1, 6, n_samples)

price = 1000 * area + 5000 * rooms + np.random.normal(0, 10000, n_samples)

X = np.column_stack([area, rooms])
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Коэффициент модели:")
print(f"Влияние площади:{model.coef_[0]:.2f}")
print(f"Влияние кол-ва комнат:{model.coef_[1]:.2f}")
print(f"Константа (базовая стоимость):{model.intercept_:.2f}")
print(f"\nСредняя квадратичная ошибка:{mse:.2f}")
print(f"Коэффициент детерминации (R2):{r2:.2f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Фактические vs Предсказанные цены')
plt.xlabel('Фактические цены')
plt.ylabel('Предсказанные цены')