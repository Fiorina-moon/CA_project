"""
相机控制
"""
import math
import numpy as np
from utils.math_utils import Vector3


class Camera:
    """轨道相机"""
    
    def __init__(self, target: Vector3 = None, distance: float = 3.0, 
                 azimuth: float = 45.0, elevation: float = 30.0):
        """
        Args:
            target: 观察目标点
            distance: 距离目标的距离
            azimuth: 方位角（度）
            elevation: 仰角（度）
        """
        self.target = target if target else Vector3(0, 0, 1)
        self.distance = distance
        self.azimuth = math.radians(azimuth)
        self.elevation = math.radians(elevation)
        
        self.fov = 45.0  # 视场角
        self.near = 0.1
        self.far = 100.0
    
    def get_position(self) -> Vector3:
        """计算相机位置"""
        x = self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        z = self.distance * math.sin(self.elevation)
        
        return Vector3(
            self.target.x + x,
            self.target.y + y,
            self.target.z + z
        )
    
    def get_view_matrix(self) -> np.ndarray:
        """获取视图矩阵"""
        position = self.get_position()
        
        # 使用lookAt计算
        forward = (self.target - position).normalize()
        right = Vector3.cross(forward, Vector3(0, 0, 1)).normalize()
        up = Vector3.cross(right, forward).normalize()
        
        view = np.eye(4, dtype=np.float32)
        view[0, 0:3] = right.to_array()
        view[1, 0:3] = up.to_array()
        view[2, 0:3] = (-forward).to_array()
        view[0:3, 3] = [-Vector3.dot(right, position),
                        -Vector3.dot(up, position),
                        Vector3.dot(forward, position)]
        
        return view
    
    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """获取投影矩阵"""
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        
        return proj
    
    def rotate(self, delta_azimuth: float, delta_elevation: float):
        """旋转相机"""
        self.azimuth += math.radians(delta_azimuth)
        self.elevation += math.radians(delta_elevation)
        
        # 限制仰角
        self.elevation = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.elevation))
    
    def zoom(self, delta: float):
        """缩放"""
        self.distance = max(0.5, min(10.0, self.distance + delta))