import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import math

def f1(x):
    return -0.4 * np.tanh(50*x) + 0.6

# Genera puntos de colocación de Chebyshev
def chebyshev_nodes(a, b, n):
    k = np.arange(1, n+1)
    x_cheb = np.cos((2*k - 1) / (2*n) * np.pi)  # Puntos de Chebyshev en [-1, 1]
    return 0.5*(b - a)*x_cheb + 0.5*(b + a)     # Escala a [a, b]

# Error de interpolación de Lagrange
def lagrange_error(x, x_nodes, f_values, deriv_max):
    n = len(x_nodes) - 1
    product_term = np.prod([(x - xi) for xi in x_nodes])
    error = (deriv_max / math.factorial(n + 1)) * product_term
    return np.abs(error)

# Datos
n_points = 15

# Generar puntos equiespaciados para la función f1
x_eq = np.linspace(-1, 1, n_points)
y_eq = f1(x_eq)

# Generar puntos de Chebyshev para la función f1
x_cheb = chebyshev_nodes(-1, 1, n_points)
y_cheb = f1(x_cheb)

# Interpolación cúbica para f1 usando puntos equiespaciados y Chebyshev
interp_f1_eq = lagrange(x_eq, y_eq)
interp_f1_cheb = lagrange(x_cheb, y_cheb)


# Ajustar los puntos densos para que estén dentro del rango de Chebyshev
x_dense = np.linspace(np.min(x_cheb), np.max(x_cheb), 400)
y_dense = f1(x_dense)
y_interp_eq = interp_f1_eq(x_dense)
y_interp_cheb = interp_f1_cheb(x_dense)

# Aproximación para la derivada máxima de la función f1
# Aquí se usa una aproximación de la derivada
deriv_max = np.max(np.abs(np.gradient(np.gradient(f1(x_dense)))))

# Calcular el error de interpolación (Lagrange) para ambos esquemas
error_eq = [lagrange_error(x, x_eq, y_eq, deriv_max) for x in x_dense]
error_cheb = [lagrange_error(x, x_cheb, y_cheb, deriv_max) for x in x_dense]

# Gráfico para f1 con puntos equiespaciados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_dense, y_dense, label='Función Original')
plt.plot(x_dense, y_interp_eq, '--', label='Interpolación Equiespaciada')
plt.scatter(x_eq, y_eq, c='red', label='Puntos Equiespaciados')
plt.title('Interpolación de f1(x) - Puntos Equiespaciados')
plt.legend()
plt.grid(True)

# Gráfico para f1 con puntos de Chebyshev
plt.subplot(1, 2, 2)
plt.plot(x_dense, y_dense, label='Función Original')
plt.plot(x_dense, y_interp_cheb, '--', label='Interpolación Chebyshev')
plt.scatter(x_cheb, y_cheb, c='green', label='Puntos Chebyshev')
plt.title('Interpolación de f1(x) - Puntos de Chebyshev')
plt.legend()
plt.grid(True)

plt.show()

# Gráfico de errores para ambos esquemas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_dense, error_eq, label='Error Interpolación Equiespaciada')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error de Interpolación - Puntos Equiespaciados')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_dense, error_cheb, label='Error Interpolación Chebyshev')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error de Interpolación - Puntos de Chebyshev')
plt.grid(True)
plt.legend()

plt.show()
