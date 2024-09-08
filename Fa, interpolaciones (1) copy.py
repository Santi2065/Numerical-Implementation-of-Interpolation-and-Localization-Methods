import numpy as np
from scipy.interpolate import lagrange, CubicSpline
import matplotlib.pyplot as plt

def fa(x):
    return - 0.4 * np.tanh(50*x) + 0.6

# Generar puntos de medición usando linspace
x_values = np.linspace(-1, 1, 15)
y_values = fa(x_values)

# Generar puntos para las interpolaciones
x_interp = np.linspace(-1, 1, 15)  # Más puntos para una mejor visualización

# Generar puntos de medición usando Chebyshev
x_values_cheb = np.polynomial.chebyshev.chebpts2(15)
y_values_cheb = fa(x_values_cheb)

# Coeficientes de la interpolación con Chebyshev
coefficients = np.polynomial.chebyshev.chebfit(x_values_cheb, y_values_cheb, 15)

# puntos densos para la interpolación con Chebyshev
x_dense = np.linspace(-1, 1, 200)
y_dense = fa(x_dense)


# Interpolación con puntos de Chebyshev
cheb_interpolation = np.polynomial.chebyshev.chebval(x_dense, coefficients)

# Graficar la función original
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_dense, label='Función original', linestyle='-', color='black')
plt.scatter(x_values_cheb, y_values_cheb, color='black', label='Puntos de Chebyshev')
plt.title('Función original e interpolación con Chebyshev')
plt.xlabel('x')
plt.ylabel('función(x)')
plt.legend()
plt.grid(True)
plt.show()

# Graficar la interpolación con puntos de Chebyshev
plt.figure(figsize=(10, 6))
plt.plot(x_dense, cheb_interpolation, label='Interpolación Chebyshev', linestyle='--', color='green')
plt.plot(x_dense, y_dense, label='Función original', linestyle='-', color='black')
plt.scatter(x_values_cheb, y_values_cheb, color='black', label='Puntos de Chebyshev')
plt.title('Interpolación Chebyshev')
plt.xlabel('x')
plt.ylabel('función(x)')
plt.legend()
plt.grid(True)
plt.show()

# Interpolación de Lagrange
lagrange_interpolation = lagrange(x_values, y_values)
y_interp_lagrange = lagrange_interpolation(x_dense)

# Interpolación con splines cúbicos
spline_interpolation = CubicSpline(x_values, y_values)
y_interp_spline = spline_interpolation(x_dense)

# Graficar la interpolación de Lagrange
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_interp_lagrange, label='Interpolación Lagrange', linestyle='--', color='blue')
plt.plot(x_dense, y_dense, label='Función original', linestyle='-', color='black')
plt.scatter(x_values, y_values, color='black', label='Puntos de medición')
plt.title('Interpolación Lagrange')
plt.xlabel('x')
plt.ylabel('función(x)')
plt.legend()
plt.grid(True)
plt.show()

# Graficar la interpolación con splines cúbicos
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_interp_spline, label='Interpolación Splines Cúbicos', linestyle='-.', color='red')
plt.plot(x_dense, y_dense, label='Función original', linestyle='-', color='black')
plt.scatter(x_values, y_values, color='black', label='Puntos de medición')
plt.title('Interpolación Splines Cúbicos')
plt.xlabel('x')
plt.ylabel('función(x)')
plt.legend()
plt.grid(True)
plt.show()

# Calcular los errores de la interpolación de Lagrange y los splines cúbicos
error_lagrange = np.abs(fa(x_dense) - y_interp_lagrange)
error_spline = np.abs(fa(x_dense) - y_interp_spline)

# Graficar los errores
plt.figure(figsize=(10, 6))
plt.plot(x_dense, error_lagrange, label='Error de Lagrange', linestyle='--', color='blue')
plt.plot(x_dense, error_spline, label='Error de Splines Cúbicos', linestyle='-.', color='red')
plt.title('Errores de Interpolación')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()