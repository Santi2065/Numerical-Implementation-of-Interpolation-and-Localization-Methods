import numpy as np
from scipy.linalg import inv
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

#parsea la ubicacion de los sensores [i, x_i(m), y_i(m), z_i(m)]
sensor_pos = []
with open('sensor_positions.txt', 'r') as file:
    i=0
    for line in file:
        if i==0:
            i+=1
            continue
        sensor_pos.append([float(x) for x in line.split(",")])
        i+=1

#parsea las distancias medidas [t(s), d1(m), d2(m), d3(m)]
distance = []
with open('measurements.txt', 'r') as file:
    i=0
    for line in file:
        if i==0:
            i+=1
            continue
        distance.append([float(x) for x in line.split(",")])
        i+=1

#parsea la ubicacion del objeto [t(s), x(m), y(m), z(m)] (ground truth)
trajectory = []
with open('trajectory.txt', 'r') as file:
    i=0
    for line in file:
        if i==0:
            i+=1
            continue
        trajectory.append([float(x) for x in line.split(",")])
        i+=1

def func(x, y, z, sensor_pos,distances,t):
    actual_time = 0
    for i in range(len(distances)):
        if distances[i][0] == t:
            actual_time = i
            break
    return np.array([
        (x - sensor_pos[0][1])**2 + (y - sensor_pos[0][2])**2 + (z - sensor_pos[0][3])**2 - distances[actual_time][1]**2,
        (x - sensor_pos[1][1])**2 + (y - sensor_pos[1][2])**2 + (z - sensor_pos[1][3])**2 - distances[actual_time][2]**2,
        (x - sensor_pos[2][1])**2 + (y - sensor_pos[2][2])**2 + (z - sensor_pos[2][3])**2 - distances[actual_time][3]**2
    ])

def jacobian(x, y, z, sensor_pos):
    return np.array([
        [2*(x - sensor_pos[0][1]), 2*(y - sensor_pos[0][2]), 2*(z - sensor_pos[0][3])],
        [2*(x - sensor_pos[1][1]), 2*(y - sensor_pos[1][2]), 2*(z - sensor_pos[1][3])],
        [2*(x - sensor_pos[2][1]), 2*(y - sensor_pos[2][2]), 2*(z - sensor_pos[2][3])]
    ])

def newton_method(x0, y0, z0, sensor_pos,distances,t=0.0, tol=1e-6, max_iter=10000):
    x, y, z = x0, y0, z0
    for _ in range(max_iter):
        f = func(x, y, z, sensor_pos,distances,t)
        J = jacobian(x, y, z, sensor_pos)
        
        # Resuelve el sistema J * delta = -f
        delta = inv(J).dot(-f)
        
        # Actualiza la estimación
        x, y, z = x + delta[0], y + delta[1], z + delta[2]
        
        # comprueba la convergencia
        if np.linalg.norm(delta) < tol:
            return x, y, z
    
    raise ValueError("El método de Newton no convergió en el número máximo de iteraciones")


# Estimación inicial
x0, y0, z0 = 0, 0, 0

#interpolacion de la trayectoria usando spline para 0.0 < t < 10
t = 0.0
trajectory_estimated = []
for i in range(100):
    x, y, z = newton_method(x0, y0, z0, sensor_pos,distance,t)
    trajectory_estimated.append([t, x, y, z])
    t += 0.5
    x0, y0, z0 = x, y, z

trajectory_estimated = np.array(trajectory_estimated)

# Ahora ajusta splines a las coordenadas estimadas
spline_x = CubicSpline(trajectory_estimated[:, 0], trajectory_estimated[:, 1])
spline_y = CubicSpline(trajectory_estimated[:, 0], trajectory_estimated[:, 2])
spline_z = CubicSpline(trajectory_estimated[:, 0], trajectory_estimated[:, 3])

# Evalua los spline en un rango más fino de tienpo, t0,tf,steps
t_fine = np.linspace(0.0, 10.0, 100)  # Ajusta según tus necesidades
x_fine = spline_x(t_fine)
y_fine = spline_y(t_fine)
z_fine = spline_z(t_fine)


# plotea la trayectoria estimada y la real en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([x[1] for x in trajectory], [x[2] for x in trajectory], [x[3] for x in trajectory], label='Real')
ax.plot(x_fine, y_fine, z_fine, label='Estimated',linestyle='--')
ax.legend()
plt.show()

#calcular error absoluto en funcion del tiempo
error = np.zeros((len(trajectory_estimated),2))
for i in range(len(trajectory_estimated)):
    error[i,0] = trajectory[i][0]
    error[i,1] = np.sqrt((trajectory[i][1]-x_fine[i])**2 + (trajectory[i][2]-y_fine[i])**2 + (trajectory[i][3]-z_fine[i])**2)



#plotea el error en funcion del tiempo
plt.plot(error[:,0],error[:,1])
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.title('Error vs Time')
plt.show()
