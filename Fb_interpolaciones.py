import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def fb(x):
    x1, x2 = x
    term1 = 0.75 * np.exp(-((9*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4)
    term2 = 0.75 * np.exp(-((9*x1 + 1)**2)/49 - ((9*x2 + 1)**2)/10)
    term3 = 0.5 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    term4 = -0.2 * np.exp(-((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
    return term1 + term2 + term3 - term4

def generate_dataset(n):
    x_vals = np.linspace(-1, 1, n)
    x_grid = np.meshgrid(x_vals, x_vals)
    dataSet = np.column_stack((x_grid[0].flatten(), x_grid[1].flatten()))
    return dataSet

def generate_interpolator_spline(dataSet, function_values):
    xi = generate_interp_points()
    return griddata(dataSet, function_values, xi, method='cubic')

def generate_interpolator_bilinear(dataSet, function_values):
    return griddata(dataSet, function_values, xi, method='linear')

def generate_interp_points():
    x_interp = np.linspace(-1, 1, 100)
    interp_points = np.meshgrid(x_interp, x_interp)
    return np.column_stack((interp_points[0].flatten(), interp_points[1].flatten()))

def plot_surface(ax, title, data):
    ax.plot_surface(data[0], data[1], data[2], cmap='plasma')
    ax.set_title(title)

def plot_wireframe(ax, title, data):
    ax.plot_wireframe(data[0], data[1], data[2], color='black', alpha=0.5)  # Wireframe en color negro
    ax.set_title(title)

n = 10
dataSet = generate_dataset(n)
function_values = [fb(x) for x in dataSet]
interpolator = generate_interpolator_spline(dataSet, function_values)

interp_points = generate_interp_points()
y_interp = interpolator

cheb_node = 10
cheb_nodes = np.cos((2 * np.arange(1, cheb_node + 1) - 1) * np.pi / (2 * cheb_node))

a, b = -1, 1
cheb_nodes_mapped = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes
x1_cheb, x2_cheb = cheb_nodes_mapped, cheb_nodes_mapped

newDataSet = dataSet.copy()
for x1, x2 in zip(x1_cheb, x2_cheb):
    newDataSet = np.append(newDataSet, [[x1, x2]], axis=0)

function_values_new = [fb(x) for x in newDataSet]

newDataSet_interpolator = generate_interpolator_spline(newDataSet, function_values_new)
y_cheb_interp = newDataSet_interpolator

y_real = [fb(x) for x in interp_points]

# Graficar interpolación spline cubico equiespaciada con el wireframe del ground truth
fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(111, projection='3d')

# Superficie de interpolación spline cubico equiespaciada
plot_surface(ax2, 'Interpolación Equiespaciada', 
             (interp_points[:, 0].reshape((100, 100)), 
              interp_points[:, 1].reshape((100, 100)), 
              y_interp.reshape((100, 100))))

# Wireframe del ground truth
plot_wireframe(ax2, 'Interpolación Equiespaciada', 
               (interp_points[:, 0].reshape((100, 100)), 
                interp_points[:, 1].reshape((100, 100)), 
                np.array(y_real).reshape((100, 100))))

plt.tight_layout()
plt.show()

# Graficar interpolación spline cubico Chebyshev con el wireframe del ground truth
fig = plt.figure(figsize=(10, 6))
ax3 = fig.add_subplot(111, projection='3d')

# Superficie de interpolación spline cubico Chebyshev
plot_surface(ax3, 'Interpolación Chebyshev', 
             (interp_points[:, 0].reshape((100, 100)), 
              interp_points[:, 1].reshape((100, 100)), 
              y_cheb_interp.reshape((100, 100))))

# Wireframe del ground truth
plot_wireframe(ax3, 'Interpolación Chebyshev', 
               (interp_points[:, 0].reshape((100, 100)), 
                interp_points[:, 1].reshape((100, 100)), 
                np.array(y_real).reshape((100, 100))))

plt.tight_layout()
plt.show()

# Calcular el Error Cuadrático Medio (MSE) para ambas interpolaciones
n_nodes = np.arange(5, 100)
mse_bilinear_equispaciada = []
mse_bilinear_chebyshev = []
mse_spline_equispaciada = []
mse_spline_chebyshev = []

for n in n_nodes:
    # Interpolación bilineal con puntos equiespaciados
    x_values = np.linspace(-1, 1, n)
    x1, x2 = np.meshgrid(x_values, x_values)
    dataSet_equi = np.column_stack((x1.flatten(), x2.flatten()))  # Puntos (x1, x2) equiespaciados
    y_values = [fb(x) for x in dataSet_equi]
    y_interp_equispaced = griddata(dataSet_equi, y_values, interp_points, method='linear')
    mse_bilinear_equispaciada.append(np.mean((np.array(y_real) - y_interp_equispaced)**2))

    # Interpolación bilineal con puntos de Chebyshev
    x_values_cheb = np.polynomial.chebyshev.chebpts2(n)
    x1_cheb, x2_cheb = np.meshgrid(x_values_cheb, x_values_cheb)
    dataSet_cheb = np.column_stack((x1_cheb.flatten(), x2_cheb.flatten()))  # Puntos (x1_cheb, x2_cheb)
    y_values_cheb = [fb(x) for x in dataSet_cheb]
    y_interp_cheb = griddata(dataSet_cheb, y_values_cheb, interp_points, method='linear')
    mse_bilinear_chebyshev.append(np.mean((np.array(y_real) - y_interp_cheb)**2))
                                     
    # Interpolación splines bicubico con puntos equiespaciados
    x_values = np.linspace(-1, 1, n)
    x1, x2 = np.meshgrid(x_values, x_values)
    dataSet_equi = np.column_stack((x1.flatten(), x2.flatten()))  # Puntos (x1, x2) equiespaciados
    y_values = [fb(x) for x in dataSet_equi]
    y_interp_equispaced = griddata(dataSet_equi, y_values, interp_points, method='cubic')
    mse_spline_equispaciada.append(np.mean((np.array(y_real) - y_interp_equispaced)**2))

    # Interpolación splines bicubico con puntos de Chebyshev
    x_values_cheb = np.polynomial.chebyshev.chebpts2(n)
    x1_cheb, x2_cheb = np.meshgrid(x_values_cheb, x_values_cheb)
    dataSet_cheb = np.column_stack((x1_cheb.flatten(), x2_cheb.flatten()))  # Puntos (x1_cheb, x2_cheb)
    y_values_cheb = [fb(x) for x in dataSet_cheb]
    y_interp_cheb = griddata(dataSet_cheb, y_values_cheb, interp_points, method='cubic')
    mse_spline_chebyshev.append(np.mean((np.array(y_real) - y_interp_cheb)**2))

# Graficar los errores de ambas interpolaciones
plt.figure(figsize=(10, 6))
plt.plot(n_nodes, mse_bilinear_equispaciada, label='Error de Interpolación Bilineal Equiespaciada', linestyle='solid', color='yellow')
plt.plot(n_nodes, mse_bilinear_chebyshev, label='Error de Interpolación Bilineal Chebyshev', linestyle='solid', color='red')
plt.plot(n_nodes, mse_spline_equispaciada, label='Error de Interpolación Equiespaciada', linestyle='solid', color='green')
plt.plot(n_nodes, mse_spline_chebyshev, label='Error de Interpolación Chebyshev', linestyle='solid', color='blue')
plt.title('Errores de Interpolación con puntos equiespaciados y Chebyshev')
plt.yscale('log')
plt.xlabel('Cantidad de nodos')
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.legend()
plt.grid(True)
plt.show()