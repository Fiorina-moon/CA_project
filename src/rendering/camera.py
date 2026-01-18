"""
相机控制
实现轨道相机（Orbit Camera）用于3D场景观察
"""
import math
import numpy as np
from src.utils.math_utils import Vector3


class Camera:
    """
    轨道相机
    
    使用球坐标系（距离、方位角、仰角）控制相机位置
    支持旋转、缩放等交互操作
    """
    
    def __init__(self, target: Vector3 = None, distance: float = 3.0, 
                 azimuth: float = 45.0, elevation: float = 30.0):
        """
        初始化相机
        
        Args:
            target: 观察目标点（世界坐标）
            distance: 距离目标的距离
            azimuth: 方位角（度，0度为+Y轴方向，逆时针为正）
            elevation: 仰角（度，0度为水平，向上为正）
        """
        self.target = target if target else Vector3(0, 0, 1)
        self.distance = distance
        self.azimuth = math.radians(azimuth)
        self.elevation = math.radians(elevation)
        
        # 投影参数
        self.fov = 45.0  # 视场角（度）
        self.near = 0.1  # 近裁剪面
        self.far = 100.0  # 远裁剪面
        
        # 约束参数
        self.min_distance = 0.5
        self.max_distance = 10.0
        self.elevation_limit = math.pi / 2 - 0.1  # 防止万向锁
    
    def get_position(self) -> Vector3:
        """
        计算相机位置（球坐标转笛卡尔坐标）
        
        公式：
            x = r * cos(φ) * sin(θ)
            y = r * cos(φ) * cos(θ)
            z = r * sin(φ)
        
        其中：
            r: 距离
            θ: 方位角（azimuth）
            φ: 仰角（elevation）
        
        Returns:
            相机在世界空间的位置
        """
        x = self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        z = self.distance * math.sin(self.elevation)
        
        return Vector3(
            self.target.x + x,
            self.target.y + y,
            self.target.z + z
        )
    
    def get_view_matrix(self) -> np.ndarray:
        """
        计算视图矩阵（世界空间 → 观察空间）
        
        Returns:
            4x4 视图变换矩阵
        """
        position = self.get_position()
        
        # 计算正交基向量
        forward = (self.target - position).normalize()
        right = Vector3.cross(forward, Vector3(0, 0, 1)).normalize()
        up = Vector3.cross(right, forward).normalize()
        
        # 构建视图矩阵（旋转 + 平移）
        view = np.eye(4, dtype=np.float32)
        
        # 旋转部分（基向量作为行）
        view[0, 0:3] = right.to_array()
        view[1, 0:3] = up.to_array()
        view[2, 0:3] = (-forward).to_array()
        
        # 平移部分（点乘）
        view[0:3, 3] = [
            -Vector3.dot(right, position),
            -Vector3.dot(up, position),
            Vector3.dot(forward, position)
        ]
        
        return view
    
    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """
        计算透视投影矩阵（观察空间 → 裁剪空间）
        
        Args:
            aspect: 宽高比（width / height）
        
        Returns:
            4x4 投影矩阵
        """
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        
        return proj
    
    # ===== 交互控制 =====
    
    def rotate(self, delta_azimuth: float, delta_elevation: float):
        """
        旋转相机
        
        Args:
            delta_azimuth: 方位角增量（度）
            delta_elevation: 仰角增量（度）
        """
        self.azimuth += math.radians(delta_azimuth)
        self.elevation += math.radians(delta_elevation)
        
        # 限制仰角（防止翻转）
        self.elevation = max(-self.elevation_limit, 
                            min(self.elevation_limit, self.elevation))
    
    def zoom(self, delta: float):
        """
        缩放（调整距离）
        
        Args:
            delta: 距离增量
        """
        self.distance = max(self.min_distance, 
                           min(self.max_distance, self.distance + delta))
    
    def pan(self, delta_x: float, delta_y: float):
        """
        平移目标点（可选功能）
        
        Args:
            delta_x: X方向增量（观察空间）
            delta_y: Y方向增量（观察空间）
        """
        position = self.get_position()
        forward = (self.target - position).normalize()
        right = Vector3.cross(forward, Vector3(0, 0, 1)).normalize()
        up = Vector3.cross(right, forward).normalize()
        
        self.target = self.target + right * delta_x + up * delta_y
    
    def reset(self):
        """重置相机到初始状态"""
        self.__init__(self.target, 3.0, 45.0, 30.0)
