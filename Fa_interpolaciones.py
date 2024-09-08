import numpy as np
from scipy.interpolate import lagrange, CubicSpline, interp1d
import matplotlib.pyplot as plt

def fa(x):
    return - 0.4 * np.tanh(50*x) + 0.6

N_POINTS = 15

# Generar puntos de medición usando linspace
x_values = np.linspace(-1, 1, N_POINTS)
y_values = fa(x_values)

# Generar puntos para las interpolaciones
x_interp = np.linspace(-1, 1, N_POINTS)  # Más puntos para una mejor visualización

# Generar puntos de medición usando Chebyshev
x_values_cheb = np.polynomial.chebyshev.chebpts2(N_POINTS)
y_values_cheb = fa(x_values_cheb)

# Coeficientes de la interpolación con Chebyshev
coefficients = np.polynomial.chebyshev.chebfit(x_values_cheb, y_values_cheb, N_POINTS-1)

# puntos densos para la interpolación
x_dense = np.linspace(-1, 1, 1000)
y_dense = fa(x_dense)

# Interpolación lineal equiespaciada
linear_interpolation = interp1d(x_values, y_values, kind='linear')
y_interp_linear = linear_interpolation(x_dense)

# Interpolación de Lagrange equiespaciada
lagrange_interpolation = lagrange(x_values, y_values)
y_interp_lagrange = lagrange_interpolation(x_dense)

# Interpolación de lineal con puntos de Chebyshev
linear_interpolation_cheb = interp1d(x_values_cheb, y_values_cheb, kind='linear')
y_interp_linear_cheb = linear_interpolation_cheb(x_dense)

#interpolacion de lagrange con puntos de chebyshev
lagrange_interpolation_cheb = lagrange(x_values_cheb, y_values_cheb)
y_interp_lagrange_cheb = lagrange_interpolation_cheb(x_dense)

# Interpolación con splines cúbicos equiespaciados
spline_interpolation = CubicSpline(x_values, y_values)
y_interp_spline = spline_interpolation(x_dense)

# Interpolación con splines cúbicos Chebyshev
spline_interpolation_cheb = CubicSpline(x_values_cheb, y_values_cheb)
y_interp_spline_cheb = spline_interpolation_cheb(x_dense)

# Graficar las interpolaciónes con puntos equiespaciados
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_interp_linear, label='Interpolación Lineal', linestyle='--', color='green')
plt.plot(x_dense, y_interp_spline, label='Interpolación Splines Cúbicos', linestyle='--', color='red')
plt.plot(x_dense, y_interp_lagrange, label='Interpolación Lagrange', linestyle='--', color='blue')
plt.plot(x_dense, y_dense, label='Función original', linestyle=':', color='black')
plt.scatter(x_values, y_values, color='black', label='Puntos de medición')
plt.title('Interpolación con puntos equiespaciados')
plt.xlabel('x')
plt.ylabel('función(x)')
plt.legend()
plt.grid(True)
plt.show()

# Graficar las interpolaciónes con puntos de Chebyshev
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_interp_linear_cheb, label='Interpolación Lineal', linestyle='--', color='green')
plt.plot(x_dense, y_interp_spline_cheb, label='Interpolación Splines Cúbicos', linestyle='--', color='red')
plt.plot(x_dense, y_interp_lagrange_cheb, label='Interpolación Lagrange', linestyle='--', color='blue')
plt.plot(x_dense, y_dense, label='Función original', linestyle=':', color='black')
plt.scatter(x_values, y_values, color='black', label='Puntos de medición')
plt.title('Interpolación con puntos de Chebyshev')
plt.xlabel('x')
plt.ylabel('función(x)')
plt.legend()
plt.grid(True)
plt.show()

# Calcular los errores de la interpolación de Lagrange y los splines cúbicos con puntos equiespaciados
error_lineal = np.abs(fa(x_dense) - y_interp_linear)
error_lagrange = np.abs(fa(x_dense) - y_interp_lagrange)
error_spline = np.abs(fa(x_dense) - y_interp_spline)

# Graficar errores de interpolación con puntos equiespaciados
plt.figure(figsize=(10, 6))
plt.plot(x_dense, error_lineal, label='Error de Interpolación Lineal', linestyle='--', color='green')
plt.plot(x_dense, error_lagrange, label='Error de Lagrange', linestyle='--', color='blue')
plt.plot(x_dense, error_spline, label='Error de Splines Cúbicos', linestyle='-.', color='red')
plt.title('Errores de Interpolación con puntos equiespaciados')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

# Calcular los errores de la interpolación de Lagrange y los splines cúbicos con Chebyshev
error_lineal_cheb = np.abs(fa(x_dense) - y_interp_linear_cheb)
error_lagrange_cheb = np.abs(fa(x_dense) - y_interp_lagrange_cheb)
error_spline_cheb = np.abs(fa(x_dense) - y_interp_spline_cheb)

#graficar los errores de interpolación con Chebyshev
plt.figure(figsize=(10, 6))
plt.plot(x_dense, error_lineal_cheb, label='Error de Interpolación Lineal', linestyle='--', color='green')
plt.plot(x_dense, error_lagrange_cheb, label='Error de Lagrange', linestyle='--', color='blue')
plt.plot(x_dense, error_spline_cheb, label='Error de Splines Cúbicos', linestyle='-.', color='red')
plt.title('Errores de Interpolación con Chebyshev')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

# calcular el Error Cuadrático Medio (MSE) para ambas interpolaciones variando la cantidad de nodos tomados
n_nodes = np.arange(5, 100)
mse_lieal = []
mse_lagrange = []
mse_spline = []
mse_lineal_cheb = []
mse_lagrange_cheb = []
mse_spline_cheb = []

for n in n_nodes:
    # Interpolación con puntos equiespaciados
    x_values = np.linspace(-1, 1, n)
    y_values = fa(x_values)
    linear_interpolation = interp1d(x_values, y_values, kind='linear')
    lagrange_interpolation = lagrange(x_values, y_values)
    spline_interpolation = CubicSpline(x_values, y_values)
    y_interp_lagrange = lagrange_interpolation(x_dense)
    y_interp_spline = spline_interpolation(x_dense)
    y_interp_linear = linear_interpolation(x_dense)
    mse_lieal.append(np.mean((fa(x_dense) - y_interp_linear)**2))
    mse_lagrange.append(np.mean((fa(x_dense) - y_interp_lagrange)**2))
    mse_spline.append(np.mean((fa(x_dense) - y_interp_spline)**2))

    # Interpolación con puntos de Chebyshev
    x_values_cheb = np.polynomial.chebyshev.chebpts2(n)
    y_values_cheb = fa(x_values_cheb)
    linear_interpolation_cheb = interp1d(x_values_cheb, y_values_cheb, kind='linear')
    lagrange_interpolation_cheb = lagrange(x_values_cheb, y_values_cheb)
    spline_interpolation_cheb = CubicSpline(x_values_cheb, y_values_cheb)
    y_interp_lagrange_cheb = lagrange_interpolation_cheb(x_dense)
    y_interp_spline_cheb = spline_interpolation_cheb(x_dense)
    y_interp_linear_cheb = linear_interpolation_cheb(x_dense)
    mse_lineal_cheb.append(np.mean((fa(x_dense) - y_interp_linear_cheb)**2))
    mse_lagrange_cheb.append(np.mean((fa(x_dense) - y_interp_lagrange_cheb)**2))
    mse_spline_cheb.append(np.mean((fa(x_dense) - y_interp_spline_cheb)**2))
                           
# Graficar el MSE en función de la cantidad de nodos
plt.figure(figsize=(10, 6))
plt.plot(n_nodes, mse_lagrange, label='Lagrange Equiespaciados', linestyle='--', color='blue')
plt.plot(n_nodes, mse_lagrange_cheb, label='Lagrange Chebyshev', linestyle='--', color='black')
plt.title('MSE en función de la cantidad de nodos')
plt.yscale('log')
plt.xlabel('Cantidad de nodos')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el MSE en función de la cantidad de nodos
plt.figure(figsize=(10, 6))
plt.plot(n_nodes, mse_lieal, label='Interpolación Lineal Equiespaciados', linestyle='--', color='green')
plt.plot(n_nodes, mse_spline, label='Splines Cúbicos Equiespaciados', linestyle='-.', color='red')
plt.plot(n_nodes, mse_lineal_cheb, label='Interpolación Lineal Chebyshev', linestyle='--', color='brown')
plt.plot(n_nodes, mse_spline_cheb, label='Splines Cúbicos Chebyshev', linestyle='-.', color='orange')
plt.title('MSE en función de la cantidad de nodos')
plt.yscale('log')
plt.xlabel('Cantidad de nodos')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()


