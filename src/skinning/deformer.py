"""
蒙皮变形器 - 极简版
"""
import numpy as np
from typing import List
from core.mesh import Mesh
from core.skeleton import Skeleton
from utils.math_utils import Vector3


class SkinDeformer:
    """极简蒙皮变形器"""
    
    def __init__(self, mesh: Mesh, skeleton: Skeleton, weights: np.ndarray):
        self.mesh = mesh
        self.skeleton = skeleton
        self.weights = weights
        
        # 保存原始顶点
        self.bind_vertices = [Vector3(v.x, v.y, v.z) for v in mesh.vertices]
        self.deformed_vertices = [Vector3(v.x, v.y, v.z) for v in mesh.vertices]
    
    def update(self):
        """
        最简LBS：顶点 = 原始位置 + 加权骨骼偏移
        """
        for i in range(len(self.bind_vertices)):
            v_bind = self.bind_vertices[i]
            
            # 累加偏移
            offset_x, offset_y, offset_z = 0.0, 0.0, 0.0
            
            for bone_idx in range(self.skeleton.get_bone_count()):
                weight = self.weights[i, bone_idx]
                if weight < 1e-6:
                    continue
                
                bone = self.skeleton.bones[bone_idx]
                joint = bone.end_joint
                
                # 骨骼偏移 = 当前位置 - 绑定位置
                # 使用head作为绑定位置
                delta_x = joint.current_position.x - joint.head.x
                delta_y = joint.current_position.y - joint.head.y
                delta_z = joint.current_position.z - joint.head.z
                
                # 加权累加
                offset_x += weight * delta_x
                offset_y += weight * delta_y
                offset_z += weight * delta_z
            
            # 应用偏移
            self.deformed_vertices[i] = Vector3(
                v_bind.x + offset_x,
                v_bind.y + offset_y,
                v_bind.z + offset_z
            )
    
    def get_deformed_vertices(self) -> List[Vector3]:
        return self.deformed_vertices
    
    def get_vertices_array(self) -> np.ndarray:
        return np.array([[v.x, v.y, v.z] for v in self.deformed_vertices], dtype=np.float32)