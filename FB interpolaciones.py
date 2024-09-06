import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator


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

def generate_interpolator(dataSet, function_values):
    return LinearNDInterpolator(dataSet, function_values)

def evaluate_interpolator(interpolator, interp_points):
    return interpolator(interp_points)

def generate_interp_points():
    x_interp = np.linspace(-1, 1, 100)
    interp_points = np.meshgrid(x_interp, x_interp)
    return np.column_stack((interp_points[0].flatten(), interp_points[1].flatten()))

def plot_surface(ax, title, data):
    ax.plot_surface(data[0], data[1], data[2], cmap='plasma')
    ax.set_title(title)

def save_figure(fig, filename):
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def main():
    n = 15
    dataSet = generate_dataset(n)
    function_values = [fb(x) for x in dataSet]
    interpolator = generate_interpolator(dataSet, function_values)
    
    interp_points = generate_interp_points()
    y_interp = evaluate_interpolator(interpolator, interp_points)
    
    cheb_node = 40
    cheb_nodes = np.cos((2 * np.arange(1, cheb_node + 1) - 1) * np.pi / (2 * cheb_node))
    
    a, b = -1, 1
    cheb_nodes_mapped = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes
    x1_cheb, x2_cheb = cheb_nodes_mapped, cheb_nodes_mapped
    
    newDataSet = dataSet.copy()
    for x1, x2 in zip(x1_cheb, x2_cheb):
        newDataSet = np.append(newDataSet, [[x1, x2]], axis=0)
    
    function_values_new = [fb(x) for x in newDataSet]
    
    newDataSet_interpolator = generate_interpolator(newDataSet, function_values_new)
    y_cheb_interp = evaluate_interpolator(newDataSet_interpolator, interp_points)
    
    y_real = [fb(x) for x in interp_points]
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    plot_surface(ax1, 'Interpolacion por Lagrange equiespaciadamente', (interp_points[:,0].reshape((100, 100)), interp_points[:,1].reshape((100, 100)), y_interp.reshape((100, 100))))
    
    ax2 = fig.add_subplot(132, projection='3d')
    plot_surface(ax2, 'Interpolacion por Lagrange con nodos de Chebyshev', (interp_points[:,0].reshape((100, 100)), interp_points[:,1].reshape((100, 100)), y_cheb_interp.reshape((100, 100))))
    
    ax3 = fig.add_subplot(133, projection='3d')
    plot_surface(ax3, 'Ground truth', (interp_points[:,0].reshape((100, 100)), interp_points[:,1].reshape((100, 100)), np.array(y_real).reshape((100, 100))))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
