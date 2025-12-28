"""
蒙皮变形器 - 完整LBS实现
"""
import numpy as np
from typing import List
from core.mesh import Mesh
from core.skeleton import Skeleton
from utils.math_utils import Vector3


class SkinDeformer:
    """完整Linear Blend Skinning实现"""
    
    def __init__(self, mesh: Mesh, skeleton: Skeleton, weights: np.ndarray):
        self.mesh = mesh
        self.skeleton = skeleton
        self.weights = weights
        
        # 保存原始顶点（绑定姿态）
        self.bind_vertices = [Vector3(v.x, v.y, v.z) for v in mesh.vertices]
        self.deformed_vertices = [Vector3(v.x, v.y, v.z) for v in mesh.vertices]
        
        print(f"[Deformer] 初始化：{len(self.bind_vertices)}个顶点，{skeleton.get_bone_count()}根骨骼")
    
    def update(self):
        """
        完整LBS算法：
        v' = Σ(w_i × M_i × B_i^(-1) × v)
        
        其中：
        - v: 绑定姿态顶点（世界空间）
        - B_i^(-1): 骨骼i的绑定姿态逆矩阵
        - M_i: 骨骼i的当前全局变换矩阵
        - w_i: 权重
        """
        for i in range(len(self.bind_vertices)):
            v_bind = self.bind_vertices[i]
            
            # 累加变换后的顶点
            v_deformed = Vector3(0, 0, 0)
            
            for bone_idx in range(self.skeleton.get_bone_count()):
                weight = self.weights[i, bone_idx]
                if weight < 1e-6:
                    continue
                
                bone = self.skeleton.bones[bone_idx]
                joint = bone.end_joint
                
                # LBS核心公式：
                # skinning_matrix = current_global × inverse_bind
                skinning_matrix = joint.global_transform * joint.inverse_bind_matrix
                
                # 变换顶点
                v_transformed = skinning_matrix.transform_point(v_bind)
                
                # 加权累加
                v_deformed = v_deformed + (v_transformed * weight)
            
            self.deformed_vertices[i] = v_deformed
    
    def get_deformed_vertices(self) -> List[Vector3]:
        return self.deformed_vertices
    
    def get_vertices_array(self) -> np.ndarray:
        return np.array([[v.x, v.y, v.z] for v in self.deformed_vertices], dtype=np.float32)