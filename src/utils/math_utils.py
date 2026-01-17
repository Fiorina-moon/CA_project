"""
数学工具：向量、矩阵运算 - 修复版
"""
import numpy as np
from typing import List, Tuple, Union


class Vector3:
    """3D向量类"""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.data = np.array([x, y, z], dtype=np.float32)
    
    @property
    def x(self) -> float:
        return self.data[0]
    
    @property
    def y(self) -> float:
        return self.data[1]
    
    @property
    def z(self) -> float:
        return self.data[2]
    
    def length(self) -> float:
        """向量长度"""
        return np.linalg.norm(self.data)
    
    def normalize(self) -> 'Vector3':
        """归一化"""
        length = self.length()
        if length > 1e-8:
            return Vector3(*(self.data / length))
        return Vector3(0, 0, 0)
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(*(self.data + other.data))
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(*(self.data - other.data))
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(*(self.data * scalar))
    
    def __rmul__(self, scalar: float) -> 'Vector3':
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
    
    def to_array(self) -> np.ndarray:
        return self.data
    
    @staticmethod
    def from_array(arr: Union[List, np.ndarray]) -> 'Vector3':
        """从数组创建"""
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))
    
    @staticmethod
    def dot(v1: 'Vector3', v2: 'Vector3') -> float:
        """点积"""
        return np.dot(v1.data, v2.data)
    
    @staticmethod
    def cross(v1: 'Vector3', v2: 'Vector3') -> 'Vector3':
        """叉积"""
        return Vector3(*np.cross(v1.data, v2.data))
    
    @staticmethod
    def distance(v1: 'Vector3', v2: 'Vector3') -> float:
        """两点间距离"""
        return (v1 - v2).length()


class Matrix4:
    """4x4变换矩阵"""
    
    def __init__(self, data: np.ndarray = None):
        if data is None:
            self.data = np.eye(4, dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
    
    @staticmethod
    def identity() -> 'Matrix4':
        """单位矩阵"""
        return Matrix4()
    
    @staticmethod
    def translation(x: float, y: float, z: float) -> 'Matrix4':
        """平移矩阵"""
        mat = Matrix4()
        mat.data[0, 3] = x
        mat.data[1, 3] = y
        mat.data[2, 3] = z
        return mat
    
    @staticmethod
    def scale(x: float, y: float, z: float) -> 'Matrix4':
        """缩放矩阵"""
        mat = Matrix4()
        mat.data[0, 0] = x
        mat.data[1, 1] = y
        mat.data[2, 2] = z
        return mat
    
    @staticmethod
    def rotation_x(angle: float) -> 'Matrix4':
        """绕X轴旋转（弧度）"""
        mat = Matrix4()
        c, s = np.cos(angle), np.sin(angle)
        mat.data[1, 1] = c
        mat.data[1, 2] = -s
        mat.data[2, 1] = s
        mat.data[2, 2] = c
        return mat
    
    @staticmethod
    def rotation_y(angle: float) -> 'Matrix4':
        """绕Y轴旋转（弧度）"""
        mat = Matrix4()
        c, s = np.cos(angle), np.sin(angle)
        mat.data[0, 0] = c
        mat.data[0, 2] = s
        mat.data[2, 0] = -s
        mat.data[2, 2] = c
        return mat
    
    @staticmethod
    def rotation_z(angle: float) -> 'Matrix4':
        """绕Z轴旋转（弧度）"""
        mat = Matrix4()
        c, s = np.cos(angle), np.sin(angle)
        mat.data[0, 0] = c
        mat.data[0, 1] = -s
        mat.data[1, 0] = s
        mat.data[1, 1] = c
        return mat
    
    @staticmethod
    def from_euler(rx: float, ry: float, rz: float) -> 'Matrix4':
        """
        从欧拉角创建旋转矩阵（XYZ顺序）
        
        Args:
            rx: 绕X轴旋转（弧度）
            ry: 绕Y轴旋转（弧度）
            rz: 绕Z轴旋转（弧度）
        
        Returns:
            旋转矩阵
        """
        Rx = Matrix4.rotation_x(rx)
        Ry = Matrix4.rotation_y(ry)
        Rz = Matrix4.rotation_z(rz)
        
        # XYZ欧拉角顺序：Rz * Ry * Rx
        return Rz * Ry * Rx
    
    def inverse(self) -> 'Matrix4':
        """矩阵求逆"""
        try:
            inv_data = np.linalg.inv(self.data)
            return Matrix4(inv_data)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，返回单位矩阵
            print("Warning: Matrix is singular, returning identity")
            return Matrix4.identity()
    
    def __mul__(self, other: 'Matrix4') -> 'Matrix4':
        """矩阵乘法"""
        return Matrix4(np.dot(self.data, other.data))
    
    def transform_point(self, point: Vector3) -> Vector3:
        """变换一个点（齐次坐标）"""
        p = np.array([point.x, point.y, point.z, 1.0], dtype=np.float32)
        result = np.dot(self.data, p)
        # 齐次坐标归一化（通常w=1，但保险起见）
        if abs(result[3]) > 1e-8:
            return Vector3(result[0]/result[3], result[1]/result[3], result[2]/result[3])
        return Vector3(result[0], result[1], result[2])
    
    def __repr__(self) -> str:
        return f"Matrix4:\n{self.data}"


def lerp(a: float, b: float, t: float) -> float:
    """线性插值"""
    return a + (b - a) * t


def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制范围"""
    return max(min_val, min(max_val, value))