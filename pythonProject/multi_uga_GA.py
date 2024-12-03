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

    # 综合代价
    return k1 * fL_normalized + k2 * fW_normalized + k3 * fA_normalized

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


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d


# 使用线性插值生成更多路径点
def interpolate_path(path, num_interp_points=100):
    x = path[:, 0]
    y = path[:, 1]
    z = path[:, 2]

    f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='linear')
    f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='linear')
    f_z = interp1d(np.linspace(0, 1, len(z)), z, kind='linear')

    interp_path = np.vstack((f_x(np.linspace(0, 1, num_interp_points)),
                             f_y(np.linspace(0, 1, num_interp_points)),
                             f_z(np.linspace(0, 1, num_interp_points)))).T
    return interp_path


# 地形生成函数
def generate_terrain(x, y, peaks):
    z = np.zeros_like(x)
    for peak in peaks:
        xc, yc = peak["center"]
        x_decay, y_decay = peak["decay"]
        height = peak["height"]
        z += height * np.exp(-((x - xc) / x_decay) ** 2 - ((y - yc) / y_decay) ** 2)
    return z


# 雷达威胁模型
def radar_threat(x, y, z, radars, R_min, R_max):
    threat = np.zeros_like(x)
    for radar in radars:
        xc, yc, zc = radar["center"]
        distance = np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
        threat += np.where(distance <= R_max, np.exp(-distance / R_min), 0)
    return threat


# 碰撞检测
def collision_detection(path, terrain):
    """优化的矢量化碰撞检测"""
    num_points = 100
    segment_points = np.linspace(0, 1, num_points)[:, None]
    all_segments = path[:-1, None, :] * (1 - segment_points) + path[1:, None, :] * segment_points
    
    x_vals = np.clip(all_segments[..., 0].ravel(), 0, terrain.shape[0] - 1).astype(int)
    y_vals = np.clip(all_segments[..., 1].ravel(), 0, terrain.shape[1] - 1).astype(int)
    z_vals = all_segments[..., 2].ravel()

    terrain_heights = terrain[x_vals, y_vals]
    if np.any(z_vals < terrain_heights):
        return True
    return False



# 适应度函数
def fitness_function(path, terrain, radars, k1, k2, k3, R_min, R_max):
    if collision_detection(path, terrain):
        return 1e6 

    interpolated_path = interpolate_path(path)
    fL = np.sum(np.sqrt(np.sum(np.diff(interpolated_path, axis=0) ** 2, axis=1)))
    fL_normalized = fL / 200
    fW = np.sum(
        radar_threat(interpolated_path[:, 0], interpolated_path[:, 1], interpolated_path[:, 2], radars, R_min, R_max))
    fW_normalized = fW / 100
    yaw_diff = np.diff(np.arctan2(np.diff(interpolated_path[:, 1]), np.diff(interpolated_path[:, 0])))
    fA = np.sum(np.abs(yaw_diff))
    fA_normalized = fA / np.pi
    sum = k1 * fL_normalized + k2 * fW_normalized + k3 * fA_normalized
    return sum


# 粒子群优化
class GA:
    def __init__(self, num_particles, num_points, bounds, fitness_func, max_iter, mutation_rate=0.1, crossover_rate=0.8):
        self.num_particles = num_particles
        self.num_points = num_points
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.initialize_population()

    def initialize_population(self):
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                             (self.num_particles, self.num_points, 3))
        self.fitness_scores = np.full(self.num_particles, np.inf)
        self.best_solution = None
        self.best_score = np.inf

    def optimize(self, start, end):
        for _ in range(self.max_iter):
            # 设置起点和终点
            for i in range(self.num_particles):
                self.population[i, 0, :] = start
                self.population[i, -1, :] = end
                score = self.fitness_func(self.population[i])
                self.fitness_scores[i] = score
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = self.population[i].copy()

            # 选择阶段（轮盘赌）
            total_fitness = np.sum(1 / (1 + self.fitness_scores))
            probabilities = (1 / (1 + self.fitness_scores)) / total_fitness
            selected_indices = np.random.choice(self.num_particles, size=self.num_particles, p=probabilities)
            selected_population = self.population[selected_indices]

            # 交叉阶段
            next_population = []
            for i in range(0, self.num_particles, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % self.num_particles]
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.num_points - 1)
                    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
                    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
                else:
                    child1, child2 = parent1, parent2
                next_population.extend([child1, child2])

            # 突变阶段
            next_population = np.array(next_population)
            mutation_mask = np.random.rand(*next_population.shape) < self.mutation_rate
            mutation_values = np.random.uniform(-1, 1, next_population.shape)
            next_population[mutation_mask] += mutation_values[mutation_mask]
            next_population = np.clip(next_population, self.bounds[:, 0], self.bounds[:, 1])

            self.population = next_population

        return self.best_solution


# 生成分散的起始点
def generate_distributed_points(num_points, x_range, y_range, z_range):
    points = []
    for _ in range(num_points):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)  # 起始点在空中
        points.append(np.array([x, y, z]))
    return points

# 生成集中分布的终点
def generate_clustered_points(num_points, center, radius, z_range):
    points = []
    for _ in range(num_points):
        x = np.random.normal(center[0], radius)
        y = np.random.normal(center[1], radius)
        z = np.random.uniform(*z_range)  # 终点在地面
        points.append(np.clip([x, y, z], [0, 0, z_range[0]], [100, 100, z_range[1]]))  # 限制在范围内
    return points


def generate_terrain_with_ocean(x, y, peaks, ocean_fraction=1/3):
    """
    根据指定参数生成地形，包含海洋和山峰。
    
    参数:
    - x, y: 网格坐标
    - peaks: 山峰的配置，每个山峰包括中心点(center)、衰减(decay)和高度(height)
    - ocean_fraction: 海洋占比 (0 到 1)
    
    返回:
    - terrain: 生成的地形数组
    """
    terrain = np.zeros_like(x)
    grid_size = x.shape[0]  # 假设 x 和 y 是方形网格

    # 创建海洋区域
    ocean_threshold = int(grid_size * ocean_fraction)
    terrain[:ocean_threshold, :] = 0  # 将前 ocean_fraction 的行设置为海洋

    # 生成陆地和山峰地形
    for peak in peaks:
        center_x, center_y = peak["center"]
        decay_x, decay_y = peak["decay"]
        height = peak["height"]
        
        # 计算每个山峰的高度影响
        peak_terrain = height * np.exp(
            -((x - center_x)**2 / (2 * decay_x**2) + (y - center_y)**2 / (2 * decay_y**2))
        )
        terrain = np.maximum(terrain, peak_terrain)  # 叠加山峰地形

    return terrain

# 辅助函数
def get_terrain_along_path(terrain, path):
    terrain_height = []
    max_x, max_y = terrain.shape
    for p in path:
        x = int(np.clip(p[0], 0, max_x - 1))  # 限制 X 坐标
        y = int(np.clip(p[1], 0, max_y - 1))  # 限制 Y 坐标
        terrain_height.append(terrain[x, y])
    return terrain_height


# 绘制所有无人机的高度与地形的对比折线图
def plot_combined(paths, terrain, radars, starts, ends, num_interp_points=100):
    # 创建一个图窗，包含两个子图
    fig = plt.figure(figsize=(16, 8))

    # 第一个子图：高度比较
    ax1 = fig.add_subplot(121)
    from matplotlib import colormaps
    colors = colormaps["tab10"]  # 使用Tab10配色方案

    for i, path in enumerate(paths):
        # 为每个无人机路径分配颜色
        color = colors(i)

        # 获取无人机路径和地形高度
        terrain_along_path = get_terrain_along_path(terrain, path)

        # 绘制无人机路径高度曲线
        ax1.plot(
            range(len(path)),
            path[:, 2],
            label=f"Drone {i + 1} Height",
            linestyle='-',
            marker='o',
            color=color
        )

        # 绘制地形高度曲线
        ax1.plot(
            range(len(terrain_along_path)),
            terrain_along_path,
            label=f"Terrain {i + 1} Height",
            linestyle='--',
            color=color
        )

    ax1.set_title("Drones Height vs Terrain")
    ax1.set_xlabel("Path Index")
    ax1.set_ylabel("Height")
    ax1.legend()
    ax1.grid()

    # 第二个子图：三维地形与路径
    ax2 = fig.add_subplot(122, projection='3d')

    # 地形表面
    x, y = np.meshgrid(np.arange(terrain.shape[0]), np.arange(terrain.shape[1]))
    ax2.plot_surface(x, y, terrain, alpha=0.7, cmap='terrain')

    # 绘制雷达覆盖范围（半圆球）
    for radar in radars:
        xc, yc, zc = radar["center"]
        radius = radar["radius"]
        u = np.linspace(0, np.pi, 50)  # 半球
        v = np.linspace(0, 2 * np.pi, 50)
        x_sphere = xc + radius * np.outer(np.sin(u), np.cos(v))
        y_sphere = yc + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = zc + radius * np.outer(np.cos(u), np.ones_like(v))
        ax2.plot_surface(x_sphere, y_sphere, z_sphere, color='red', alpha=0.3)

    for i, (start, end, path) in enumerate(zip(starts, ends, paths)):
        color = colors(i)

        # 绘制起点、终点
        ax2.scatter(*start, color=color, s=100, label=f"Start {i + 1}", marker="o")
        ax2.scatter(*end, color=color, s=100, label=f"End {i + 1}", marker="x")

        # 路径绘制
        ax2.plot(path[:, 0], path[:, 1], path[:, 2], color=color, label=f"Path {i + 1}", linewidth=2)

        # 插值平滑路径
        interpolated_path = interpolate_path(path, num_interp_points)

        # 计算路径在地形表面的投影
        surface_projection_z = np.array([
            terrain[
                np.clip(int(round(p[0])), 0, terrain.shape[0] - 1),  # 限制 x 坐标范围
                np.clip(int(round(p[1])), 0, terrain.shape[1] - 1)  # 限制 y 坐标范围
            ]
            for p in interpolated_path
        ])

        # 绘制路径在地形表面的投影
        ax2.plot(interpolated_path[:, 0], interpolated_path[:, 1], surface_projection_z,
                 color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_title("3D Terrain and Drone Paths")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_zlabel("Z-axis")
    ax2.legend()

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 增加复杂地形的 peaks 配置
    peaks = [
        {"center": (30, 30), "decay": (5, 8), "height": 40},  # 左侧山峰
        {"center": (70, 70), "decay": (8, 5), "height": 40},  # 右侧山峰
        {"center": (50, 50), "decay": (10, 10), "height": 80},  # 中心高山
        {"center": (30, 70), "decay": (7, 9), "height": 20},  # 额外小山峰
        {"center": (70, 30), "decay": (9, 7), "height": 20}  # 对称小山峰
    ]

    radars = [{"center": (80, 30, 0), "radius": 10}, {"center": (30, 80, 0), "radius": 10}]
    bounds = np.array([[0, 100], [0, 100], [0, 100]])

    # 参数
    num_drones = 10
    x_range_start = (0, 3)  # 起点X坐标范围
    y_range_start = (80, 100) # 起点Y坐标范围
    z_range_start = (20, 24)  # 起点Z坐标范围（空中）

    cluster_center = [80, 0]  # 终点集中点的中心
    cluster_radius = 20 # 终点集中点的半径
    z_range_end = (0, 2)  # 终点Z坐标范围（地面）

    # 生成起点和终点
    starts = generate_distributed_points(num_drones, x_range_start, y_range_start, z_range_start)
    ends = generate_clustered_points(num_drones, cluster_center, cluster_radius, z_range_end)

    k1, k2, k3 = 0.5, 0.3, 0.2

    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    x, y = np.meshgrid(x, y)
    terrain = generate_terrain(x, y, peaks)

    fitness_func = lambda path: fitness_function(path, terrain, radars, k1, k2, k3, 20, 30)
    paths = []
    print("处理中……")

    for start, end in zip(starts, endssd):
        optimizer = GA(num_particles=30, num_points=5, bounds=bounds, fitness_func=fitness_func,
                       max_iter=200, mutation_rate=0.1, crossover_rate=0.8)
        paths.append(optimizer.optimize(start, end))


    plot_combined(paths, terrain, radars, starts, ends)

