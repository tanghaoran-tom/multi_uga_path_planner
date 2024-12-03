import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.interpolate import interp1d

# 使用线性插值生成更多路径点
def interpolate_path(path, num_interp_points=100):
    x = path[:, 0]
    y = path[:, 1]
    z = path[:, 2]
    
    # 在每个轴上分别进行插值
    interp_x = np.linspace(x.min(), x.max(), num_interp_points)
    interp_y = np.linspace(y.min(), y.max(), num_interp_points)
    interp_z = np.linspace(z.min(), z.max(), num_interp_points)
    
    # 线性插值函数
    f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='linear')
    f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='linear')
    f_z = interp1d(np.linspace(0, 1, len(z)), z, kind='linear')
    
    # 生成插值后的路径
    interp_path = np.vstack((f_x(np.linspace(0, 1, num_interp_points)),
                             f_y(np.linspace(0, 1, num_interp_points)),
                             f_z(np.linspace(0, 1, num_interp_points)))).T
    return interp_path


# 1. 定义地形生成函数
def generate_terrain(x, y, peaks):
    z = np.zeros_like(x)
    for peak in peaks:
        xc, yc = peak["center"]
        x_decay, y_decay = peak["decay"]
        height = peak["height"]
        z += height * np.exp(-((x - xc) / x_decay)**2 - ((y - yc) / y_decay)**2)
    return z

# 2. 定义雷达威胁模型
def radar_threat(x, y, z, radars, R_min, R_max):
    threat = np.zeros_like(x)
    for radar in radars:
        xc, yc, zc = radar["center"]
        distance = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
        threat += np.where(distance <= R_max, np.exp(-distance / R_min), 0)
    return threat

# 碰撞检测函数，增加了对路径段的检测
def collision_detection(path, terrain):
    for i in range(len(path) - 1):
        x1, y1, z1 = path[i]
        x2, y2, z2 = path[i + 1]
        
        # 生成路径段（两个点之间的连线）
        num_points = 100  # 在路径段上取100个点进行检测
        x_vals = np.linspace(x1, x2, num_points)
        y_vals = np.linspace(y1, y2, num_points)
        z_vals = np.linspace(z1, z2, num_points)

        # 检查路径段上的每个点是否与地形发生碰撞
        for j in range(num_points):
            x, y, z = x_vals[j], y_vals[j], z_vals[j]
            xi, yi = int(np.clip(x, 0, terrain.shape[0] - 1)), int(np.clip(y, 0, terrain.shape[1] - 1))
            terrain_height = terrain[xi, yi]
            if z < terrain_height:  # 如果路径点位于地形之下，认为发生了碰撞
                return True  # 一旦检测到碰撞，返回True表示碰撞发生
    return False  # 如果没有碰撞，返回False

# 适应度函数
def fitness_function(path, terrain, radars, k1, k2, k3, R_min, R_max):
    # 检查路径是否与地形发生碰撞
    if collision_detection(path, terrain):
        return 999  # 如果路径发生碰撞，返回无穷大的代价

   # 使用插值生成更多路径点
    interpolated_path = interpolate_path(path)
    
    # 路径长度代价
    fL = np.sum(np.sqrt(np.sum(np.diff(interpolated_path, axis=0)**2, axis=1)))
    fL_max = 200  # 根据地图的最大可能路径长度估计
    fL_min = 0
    fL_normalized = (fL - fL_min) / (fL_max - fL_min)

    # 雷达威胁代价
    fW = np.sum(radar_threat(interpolated_path[:, 0], interpolated_path[:, 1], interpolated_path[:, 2], radars, R_min, R_max))
    fW_max = 100  # 根据雷达分布和路径点数量合理估计
    fW_min = 0
    fW_normalized = (fW - fW_min) / (fW_max - fW_min)

    # 偏航角代价
    yaw_diff = np.diff(np.arctan2(np.diff(interpolated_path[:, 1]), np.diff(interpolated_path[:, 0])))
    fA = np.sum(np.abs(yaw_diff))
    fA_max = np.pi  # 最大偏航角为 180 度
    fA_min = 0
    fA_normalized = (fA - fA_min) / (fA_max - fA_min)

    # 地形碰撞代价
    x_indices = np.clip(interpolated_path[:, 0].astype(int), 0, terrain.shape[0] - 1)
    y_indices = np.clip(interpolated_path[:, 1].astype(int), 0, terrain.shape[1] - 1)
    terrain_height = terrain[x_indices, y_indices]
    fC = np.sum(np.maximum(0, terrain_height - interpolated_path[:, 2]))
    fC_max = 50  # 合理估计最大碰撞代价
    fC_min = 0
    fC_normalized = (fC - fC_min) / (fC_max - fC_min)

    
    # 综合代价
    return k1 * fL_normalized + k2 * fW_normalized + k3 * fA_normalized

# 4. 粒子群算法
class PSO:
    def __init__(self, num_particles, num_points, bounds, fitness_func, max_iter, C1, C2, W_max, W_min, V_max):
        self.num_particles = num_particles
        self.num_points = num_points
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.max_iter = max_iter
        self.C1 = C1
        self.C2 = C2
        self.W_max = W_max
        self.W_min = W_min
        self.V_max = V_max
        self.initialize_particles()

    def initialize_particles(self):
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_particles, self.num_points, 3))
        self.positions[:, 0, :] = start  # 固定起点
        self.positions[:, -1, :] = end  # 固定终点
        self.velocities = np.random.uniform(-self.V_max, self.V_max, (self.num_particles, self.num_points, 3))
        self.pbest = self.positions.copy()
        self.pbest_scores = np.full(self.num_particles, np.inf)
        self.gbest = None
        self.gbest_score = np.inf

    def optimize(self):
        for t in range(self.max_iter):
            W = self.W_max - t * (self.W_max - self.W_min) / self.max_iter
            for i in range(self.num_particles):
                score = self.fitness_func(self.positions[i])
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest[i] = self.positions[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.positions[i].copy()

                r1, r2 = np.random.rand(2)
                self.velocities[i] = W * self.velocities[i] + \
                                     self.C1 * r1 * (self.pbest[i] - self.positions[i]) + \
                                     self.C2 * r2 * (self.gbest - self.positions[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.V_max, self.V_max)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])

                # 固定起点和终点
                self.positions[i, 0, :] = start
                self.positions[i, -1, :] = end
        return self.gbest


# 5. 绘制结果
# 5. 绘制结果
def plot_results(terrain, path, radars, start, end, num_interp_points=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(terrain.shape[0]), np.arange(terrain.shape[1]))
    ax.plot_surface(x, y, terrain, alpha=0.7, cmap='terrain')

    # 绘制雷达
    for radar in radars:
        xc, yc, zc = radar["center"]
        ax.scatter(xc, yc, zc, color='red', s=50, label="Radar")

    # 绘制起点和终点
    ax.scatter(*start, color='green', s=100, label="Start", marker="o")
    ax.scatter(*end, color='blue', s=100, label="End", marker="x")

    # 绘制原始路径
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', label="Original Path", linestyle='-', linewidth=2)

    # 使用插值生成平滑路径
    interpolated_path = interpolate_path(path, num_interp_points)
    ax.plot(interpolated_path[:, 0], interpolated_path[:, 1], interpolated_path[:, 2], color='orange', label="Interpolated Path", linestyle='--', linewidth=2)

    plt.legend()
    plt.show()

# 6. 主程序
if __name__ == "__main__":
    # 山峰数据
    peaks = [
    {"center": (20, 20), "decay": (7, 9), "height": 45},
    {"center": (30, 60), "decay": (5, 5), "height": 51},
    {"center": (50, 50), "decay": (9, 6), "height": 62},
    {"center": (70, 20), "decay": (3, 4), "height": 88},
    {"center": (85, 85), "decay": (4, 7), "height": 40},
    {"center": (60, 70), "decay": (4, 5), "height": 74},
    ]

    # 雷达数据
    radars = [
        {"center": (80, 30, 0), "radius": 30},  # 半径增大
        {"center": (30, 80, 0), "radius": 30},  # 半径增大
    ]

    # 其他参数
    bounds = np.array([[0, 100], [0, 100], [0, 100]])
    start = np.array([0, 0, 5])
    end = np.array([100, 100, 40])
    # 其他参数
    k1, k2, k3 = 0.5, 0.3, 0.2  # 航迹长度，雷达威胁，偏航角代价，地形碰撞代价

    # 创建地形
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    x, y = np.meshgrid(x, y)
    terrain = generate_terrain(x, y, peaks)

    # 粒子群优化
    fitness_func = lambda path: fitness_function(path, terrain, radars, k1, k2, k3, 20, 30)
    pso = PSO(num_particles=20, num_points=5, bounds=bounds, fitness_func=fitness_func, max_iter=100, C1=1.5, C2=1.5, W_max=0.9, W_min=0.4, V_max=10)
    optimal_path = pso.optimize()

    # 提取无人机路径点和地形高度
    x_indices = np.clip(optimal_path[:, 0].astype(int), 0, terrain.shape[0] - 1)
    y_indices = np.clip(optimal_path[:, 1].astype(int), 0, terrain.shape[1] - 1)
    terrain_heights = terrain[x_indices, y_indices]
    drone_heights = optimal_path[:, 2]

    # 绘制无人机高度和地形高度对比图
    plt.figure(figsize=(10, 6))
    plt.plot(drone_heights, label="Drone Height", marker="o", color="blue")
    plt.plot(terrain_heights, label="Terrain Height", marker="x", color="green")
    plt.fill_between(range(len(terrain_heights)), terrain_heights, drone_heights, 
                    where=(drone_heights < terrain_heights), color='red', alpha=0.3, label="Collision")
    plt.title("Drone Height vs Terrain Height")
    plt.xlabel("Path Point Index")
    plt.ylabel("Height")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 检测碰撞
    collisions = collision_detection(optimal_path, terrain)
    if collisions:
        print("Collision detected at the following points:")
        for x, y, z, terrain_height in collisions:
            print(f"Path point ({x:.2f}, {y:.2f}, {z:.2f}) is below terrain height {terrain_height:.2f}")
    else:
        print("No collisions detected.")

    # 绘制结果
    plot_results(terrain, optimal_path, radars, start, end)
    