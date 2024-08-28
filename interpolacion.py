# Estudie la performance de distintos esquemas de interpolación en las funciones
# f1(x) con x entre -1 y 1
# f2(x) con x1,x2 entre -1 y 1
# Hágalo primero tomando puntos de colocación equiespaciados. Luego proponga (al menos) una regla
# para elegir puntos no equiespaciados. Compare los resultados.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def f1(x):
    return -0.4 * np.tanh(50*x) + 0.6

def f2(x1, x2):
    term1 = 0.75 * np.exp(-((9*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4)
    term2 = 0.75 * np.exp(-((9*x1 + 1)**2)/49 - ((9*x2 + 1)**2)/10)
    term3 = 0.5 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    term4 = -0.2 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    return term1 + term2 + term3 + term4

# Generar puntos equiespaciados para la función f1
x_eq = np.linspace(-1, 1, 15)
y_eq = f1(x_eq)

# Interpolación para f1 usando interpolación lineal
interp_f1_eq = interp1d(x_eq, y_eq, kind='linear')

# grid para evaluar la interpolación en f1
x_dense = np.linspace(-1, 1, 400)
y_dense = f1(x_dense)
y_interp_eq = interp_f1_eq(x_dense)

# Gráfico para f1
plt.plot(x_dense, y_dense, label='Función Original')
plt.plot(x_dense, y_interp_eq, '--', label='Interpolación Equiespaciada')
plt.scatter(x_eq, y_eq, c='red', label='Puntos Equiespaciados')
plt.title('Interpolación de f1(x)')
plt.legend()
plt.grid(True)


plt.show()

