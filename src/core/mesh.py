"""
Mesh数据结构
"""
import numpy as np
from typing import List, Tuple
from src.utils.math_utils import Vector3


class Vertex:
    """顶点类"""
    
    def __init__(self, position: Vector3, normal: Vector3 = None, texcoord: Tuple[float, float] = None):
        self.position = position
        self.normal = normal if normal else Vector3(0, 1, 0)
        self.texcoord = texcoord if texcoord else (0.0, 0.0)
        self.weights = {}  # {bone_index: weight}
    
    def __repr__(self) -> str:
        return f"Vertex(pos={self.position}, normal={self.normal})"


class Face:
    """面（三角形）"""
    
    def __init__(self, vertex_indices: List[int], normal_indices: List[int] = None, 
                 texcoord_indices: List[int] = None):
        self.vertex_indices = vertex_indices
        self.normal_indices = normal_indices if normal_indices else []
        self.texcoord_indices = texcoord_indices if texcoord_indices else []
    
    def __repr__(self) -> str:
        return f"Face(vertices={self.vertex_indices})"


class Mesh:
    """Mesh类"""
    
    def __init__(self):
        self.vertices: List[Vector3] = []  # 顶点位置列表
        self.normals: List[Vector3] = []   # 法线列表
        self.texcoords: List[Tuple[float, float]] = []  # 纹理坐标列表
        self.faces: List[Face] = []  # 面列表
        
        # 蒙皮权重（顶点数 × 骨骼数）
        self.weights: np.ndarray = None
        
        self.name = "Mesh"
    
    def get_vertex_count(self) -> int:
        """获取顶点数量"""
        return len(self.vertices)
    
    def get_face_count(self) -> int:
        """获取面数量"""
        return len(self.faces)
    
    def get_bounding_box(self) -> Tuple[Vector3, Vector3]:
        """获取包围盒"""
        if not self.vertices:
            return Vector3(0, 0, 0), Vector3(0, 0, 0)
        
        positions = np.array([v.to_array() for v in self.vertices])
        min_pos = Vector3.from_array(positions.min(axis=0))
        max_pos = Vector3.from_array(positions.max(axis=0))
        
        return min_pos, max_pos
    
    def compute_normals(self):
        """计算顶点法线（如果没有）"""
        if self.normals:
            return
        
        # 初始化法线为零向量
        vertex_normals = [Vector3(0, 0, 0) for _ in self.vertices]
        
        # 累加每个面的法线
        for face in self.faces:
            if len(face.vertex_indices) < 3:
                continue
            
            v0 = self.vertices[face.vertex_indices[0]]
            v1 = self.vertices[face.vertex_indices[1]]
            v2 = self.vertices[face.vertex_indices[2]]
            
            # 计算面法线
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = Vector3.cross(edge1, edge2)
            
            # 累加到顶点法线
            for idx in face.vertex_indices:
                vertex_normals[idx] = vertex_normals[idx] + face_normal
        
        # 归一化
        self.normals = [n.normalize() for n in vertex_normals]
    
    def __repr__(self) -> str:
        return f"Mesh(vertices={len(self.vertices)}, faces={len(self.faces)})"