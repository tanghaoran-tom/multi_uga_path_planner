import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.interpolate import interp1d
import argparse
import numpy as np

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

# 检测路径是否与地形或禁飞区冲突
def collision_detection(path, terrain, no_fly_zones):
    num_points = 100
    segment_points = np.linspace(0, 1, num_points)[:, None]
    all_segments = path[:-1, None, :] * (1 - segment_points) + path[1:, None, :] * segment_points

    x_vals = np.clip(all_segments[..., 0].ravel(), 0, terrain.shape[0] - 1).astype(int)
    y_vals = np.clip(all_segments[..., 1].ravel(), 0, terrain.shape[1] - 1).astype(int)
    z_vals = all_segments[..., 2].ravel()

    # 地形高度检测
    terrain_heights = terrain[x_vals, y_vals]
    if np.any(z_vals < terrain_heights):
        return True

    # 禁飞区检测
    for zone in no_fly_zones:
        center = np.array(zone["center"][:2])
        radius = zone["radius"]
        height_min, height_max = zone["height_range"]

        positions_2d = all_segments[..., :2].reshape(-1, 2)
        distances = np.linalg.norm(positions_2d - center, axis=1)
        in_radius = distances <= radius
        in_height = (z_vals >= height_min) & (z_vals <= height_max)

        if np.any(in_radius & in_height):
            return True

    return False

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
# 创建地形的函数
def generate_terrain(x, y, peaks):
    terrain = np.zeros_like(x)
    for peak in peaks:
        cx, cy = peak["center"]
        dx, dy = peak["decay"]
        height = peak["height"]
        terrain += height * np.exp(-(((x - cx) / dx) ** 2 + ((y - cy) / dy) ** 2))

    return terrain


# 雷达威胁模型
def radar_threat(x, y, z, radars, R_min, R_max):
    threat = np.zeros_like(x)
    for radar in radars:
        xc, yc, zc = radar["center"]
        distance = np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
        threat += np.where(distance <= R_max, np.exp(-distance / R_min), 0)
    return threat


# 碰撞检测
# 检测路径是否与地形或禁飞区冲突
def collision_detection(path, terrain, no_fly_zones):
    num_points = 100
    segment_points = np.linspace(0, 1, num_points)[:, None]
    all_segments = path[:-1, None, :] * (1 - segment_points) + path[1:, None, :] * segment_points

    x_vals = np.clip(all_segments[..., 0].ravel(), 0, terrain.shape[0] - 1).astype(int)
    y_vals = np.clip(all_segments[..., 1].ravel(), 0, terrain.shape[1] - 1).astype(int)
    z_vals = all_segments[..., 2].ravel()

    # 地形高度检测
    terrain_heights = terrain[x_vals, y_vals]
    if np.any(z_vals < terrain_heights):
        return True

    # 禁飞区检测
    for zone in no_fly_zones:
        center = np.array(zone["center"][:2])
        radius = zone["radius"]
        height_min, height_max = zone["height_range"]

        positions_2d = all_segments[..., :2].reshape(-1, 2)
        distances = np.linalg.norm(positions_2d - center, axis=1)
        in_radius = distances <= radius
        in_height = (z_vals >= height_min) & (z_vals <= height_max)

        if np.any(in_radius & in_height):
            return True

    return False

def get_terrain_along_path(terrain, path):
    terrain_height = []
    max_x, max_y = terrain.shape
    for p in path:
        x = int(np.clip(p[0], 0, max_x - 1))  # 限制 X 坐标
        y = int(np.clip(p[1], 0, max_y - 1))  # 限制 Y 坐标
        terrain_height.append(terrain[x, y])
    return terrain_height

# 适应度函数
def fitness_function(path, terrain, radars, k1, k2, k3, R_min, R_max):
    if collision_detection(path, terrain,no_fly_zones):
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
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                           (self.num_particles, self.num_points, 3))
        self.velocities = np.random.uniform(-self.V_max, self.V_max, (self.num_particles, self.num_points, 3))
        self.pbest = self.positions.copy()
        self.pbest_scores = np.full(self.num_particles, np.inf)
        self.gbest = None
        self.gbest_score = np.inf

    def optimize(self, start, end):
        for t in range(self.max_iter):
            W = self.W_max - t * (self.W_max - self.W_min) / self.max_iter
            for i in range(self.num_particles):
                self.positions[i, 0, :] = start
                self.positions[i, -1, :] = end
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
        return self.gbest



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


# 绘制所有无人机的高度与地形的对比折线图
def plot_combined(paths, terrain, radars, no_fly_zones, starts, ends, num_interp_points=100):
    # 创建一个图窗，包含两个子图
    fig = plt.figure(figsize=(16, 8))

    # 第一个子图：综合地形高度折线图
    ax1 = fig.add_subplot(121)
    from matplotlib import colormaps
    colors = colormaps["tab10"]  # 使用Tab10配色方案

    for i, path in enumerate(paths):
        color = colors(i)
        combined_terrain_height = []

        for point in path:
            x, y, z_drone = point
            x_idx, y_idx = int(round(x)), int(round(y))
            x_idx = np.clip(x_idx, 0, terrain.shape[0] - 1)
            y_idx = np.clip(y_idx, 0, terrain.shape[1] - 1)

            z_terrain = terrain[x_idx, y_idx]
            z_combined = z_terrain

            for zone in no_fly_zones:
                xc, yc, _ = zone["center"]
                radius = zone["radius"]
                z_min, z_max = zone["height_range"]
                if (x - xc) ** 2 + (y - yc) ** 2 <= radius ** 2:
                    z_combined = max(z_combined, z_min)

            for radar in radars:
                xc, yc, zc = radar["center"]
                radius = radar["radius"]
                if (x - xc) ** 2 + (y - yc) ** 2 + (z_drone - zc) ** 2 <= radius ** 2:
                    z_radar_surface = zc + np.sqrt(radius ** 2 - (x - xc) ** 2 - (y - yc) ** 2)
                    z_combined = max(z_combined, z_radar_surface)

            combined_terrain_height.append(z_combined)

        # 设置折线粗细
        ax1.plot(range(len(path)), path[:, 2], label=f"Drone {i + 1} Height", linestyle='-', marker='o', color=color,
                 linewidth=2)
        ax1.plot(range(len(path)), combined_terrain_height, label=f"Combined Terrain {i + 1}", linestyle='--',
                 color=color, linewidth=1.5)

    ax1.set_title("Drones Height vs Combined Terrain")
    ax1.set_xlabel("Path Index")
    ax1.set_ylabel("Height")
    ax1.legend()
    ax1.grid()

    # 第二个子图：三维地形与路径
    ax2 = fig.add_subplot(122, projection='3d')

    # 更新的部分：仅山地为纯蓝色
    visibility_threshold = 1e-2  # 设定山地的最低可视海拔
    terrain_visible = np.where(terrain > visibility_threshold, terrain, np.nan)

    # 自定义颜色映射：低于阈值部分为透明，高于阈值部分为纯蓝色
    from matplotlib.colors import ListedColormap

    # 创建纯蓝色的自定义颜色映射
    blue_cmap = ListedColormap(['blue'])

    # 绘制山地表面（超过阈值的部分为纯蓝色）
    x, y = np.meshgrid(np.arange(terrain.shape[0]), np.arange(terrain.shape[1]))
    ax2.plot_surface(
        x, y, terrain_visible, cmap=blue_cmap, alpha=0.8, edgecolor='none', zorder=1
    )

    # 绘制雷达覆盖范围（半圆球），放在地形之上
    for radar in radars:
        xc, yc, zc = radar["center"]
        radius = radar["radius"]
        u = np.linspace(0, np.pi, 50)  # 半球
        v = np.linspace(0, 2 * np.pi, 50)
        x_sphere = xc + radius * np.outer(np.sin(u), np.cos(v))
        y_sphere = yc + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = zc + radius * np.outer(np.cos(u), np.ones_like(v))
        ax2.plot_surface(
            x_sphere, y_sphere, z_sphere,
            color='red', alpha=0.5, rstride=5, cstride=5, edgecolor='black', zorder=2
        )

    # 绘制禁飞区（圆柱体）
    for zone in no_fly_zones:
        xc, yc, z_min, z_max = zone["center"][0], zone["center"][1], zone["height_range"][0], zone["height_range"][1]
        radius = zone["radius"]
        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(z_min, z_max, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cylinder = xc + radius * np.cos(theta_grid)
        y_cylinder = yc + radius * np.sin(theta_grid)
        ax2.plot_surface(
            x_cylinder, y_cylinder, z_grid,
            color='orange', alpha=0.5, rstride=10, cstride=10
        )

    for i, (start, end, path) in enumerate(zip(starts, ends, paths)):
        color = colors(i)
        ax2.scatter(*start, color=color, s=100, label=f"Start {i + 1}", marker="o")
        ax2.scatter(*end, color=color, s=100, label=f"End {i + 1}", marker="x")
        ax2.plot(path[:, 0], path[:, 1], path[:, 2], color=color, label=f"Path {i + 1}", linewidth=1)

    ax2.set_title("3D Terrain and Drone Paths")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_zlabel("Z-axis")
    ax2.legend()

    plt.tight_layout()
    plt.show()



def is_valid_point(x, y, z, terrain, radars, no_fly_zones):
    # 检查是否在雷达覆盖范围内
    for radar in radars:
        xc, yc, zc = radar["center"]
        radius = radar["radius"]
        if (x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2 <= radius ** 2:
            return False

    # 检查是否在禁飞区内
    for zone in no_fly_zones:
        xc, yc, _ = zone["center"]
        radius = zone["radius"]
        z_min, z_max = zone["height_range"]
        if (x - xc) ** 2 + (y - yc) ** 2 <= radius ** 2 and z_min <= z <= z_max:
            return False

    # 检查是否在山峰上
    x_idx = np.clip(int(round(x)), 0, terrain.shape[0] - 1)  # 限制 x 索引范围
    y_idx = np.clip(int(round(y)), 0, terrain.shape[1] - 1)  # 限制 y 索引范围
    terrain_height = terrain[x_idx, y_idx]
    if z < terrain_height:
        return False

    return True


def generate_valid_endpoint(cluster_center, cluster_radius, z_range_end, terrain, radars, no_fly_zones, bounds):
    while True:
        # 在 cluster_center 附近随机生成点
        x = np.random.uniform(cluster_center[0] - cluster_radius, cluster_center[0] + cluster_radius)
        y = np.random.uniform(cluster_center[1] - cluster_radius, cluster_center[1] + cluster_radius)
        z = np.random.uniform(z_range_end[0], z_range_end[1])

        # 确保点在范围内
        if not (bounds[0, 0] <= x <= bounds[0, 1] and bounds[1, 0] <= y <= bounds[1, 1] and bounds[2, 0] <= z <= bounds[2, 1]):
            continue

        # 检查点是否合法
        if is_valid_point(x, y, z, terrain, radars, no_fly_zones):
            return [x, y, z]

# 定义场景
import random

def generate_random_peaks(num_peaks, bounds):
    return [
        {
            "center": (random.uniform(bounds[0][0], bounds[0][1]),
                       random.uniform(bounds[1][0], bounds[1][1])),
            "decay": (random.uniform(1, 1), random.uniform(1, 1)),
            "height": random.uniform(30, 100),
        }
        for _ in range(num_peaks)
    ]

def generate_random_no_fly_zones(num_zones, bounds):
    return [
        {
            "center": (random.uniform(bounds[0][0], bounds[0][1]),
                       random.uniform(bounds[1][0], bounds[1][1]),
                       0),
            "radius": random.uniform(2, 6),
            "height_range": (0, random.uniform(40, 80)),
        }
        for _ in range(num_zones)
    ]

def generate_random_radars(num_radars, bounds):
    return [
        {
            "center": (random.uniform(bounds[0][0], bounds[0][1]),
                       random.uniform(bounds[1][0], bounds[1][1]),
                       0),
            "radius": random.uniform(3, 8),
        }
        for _ in range(num_radars)
    ]

def get_scenarios():
    bounds = [[0, 100], [0, 100]]  # 地图范围
    return [
        {  # 场景 1：50 个山地
            "peaks": generate_random_peaks(50, bounds),
            "radars": generate_random_radars(2, bounds),
            "no_fly_zones": generate_random_no_fly_zones(2, bounds),
        },
        {  # 场景 2：50 个禁飞区
            "peaks": generate_random_peaks(2, bounds),
            "radars": generate_random_radars(2, bounds),
            "no_fly_zones": generate_random_no_fly_zones(50, bounds),
        },
        {  # 场景 3：50 个小雷达
            "peaks": generate_random_peaks(2, bounds),
            "radars": generate_random_radars(50, bounds),
            "no_fly_zones": generate_random_no_fly_zones(2, bounds),
        },
        {  # 场景 4：综合场景
            "peaks": generate_random_peaks(20, bounds),
            "radars": generate_random_radars(20, bounds),
            "no_fly_zones": generate_random_no_fly_zones(20, bounds),
        },
    ]

# 主程序
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="无人机路径规划场景模拟器")
    parser.add_argument("--num_drones", type=int, default=10, help="无人机数量")
    parser.add_argument("--scene", type=int, default=1, choices=range(1, 5), help="选择场景 (1-4)")

    args = parser.parse_args()
    num_drones = args.num_drones
    scene_index = args.scene - 1  # 场景编号转换为索引

    # 获取场景配置
    scenarios = get_scenarios()
    selected_scene = scenarios[scene_index]
    peaks = selected_scene["peaks"]
    radars = selected_scene["radars"]
    no_fly_zones = selected_scene["no_fly_zones"]

    # 生成地形
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    x, y = np.meshgrid(x, y)
    terrain = generate_terrain(x, y, peaks)

    # 生成起点和终点
    x_range_start = (0, 3)
    y_range_start = (80, 100)
    z_range_start = (20, 24)
    starts = generate_distributed_points(num_drones, x_range_start, y_range_start, z_range_start)

    ends = []
    y_values = np.linspace(0, 100, num_drones)
    z_range_end = (0, 2)

    for y in y_values:
        end_x = 100
        end_z = np.random.uniform(z_range_end[0], z_range_end[1])
        ends.append((end_x, y, end_z))

    # 定义适应度函数
    k1, k2, k3 = 0.5, 0.3, 0.2
    fitness_func = lambda path: fitness_function(path, terrain, radars, k1, k2, k3, 20, 30)

    # 路径规划
    paths = []
    print("处理中……")
    for start, end in zip(starts, ends):
        pso = PSO(
            num_particles=40,
            num_points=5,
            bounds=np.array([[0, 100], [0, 100], [0, 100]]),
            fitness_func=fitness_func,
            max_iter=200,
            C1=2.0,
            C2=2.0,
            W_max=0.7,
            W_min=0.3,
            V_max=10,
        )
        paths.append(pso.optimize(start, end))

    # 可视化结果
    plot_combined(paths, terrain, radars, no_fly_zones, starts, ends)
