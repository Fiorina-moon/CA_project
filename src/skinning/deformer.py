"""
蒙皮变形器 - 完全修复版
"""
import numpy as np
from typing import List
from core.mesh import Mesh
from core.skeleton import Skeleton
from utils.math_utils import Vector3

class SkinDeformer:
    """Linear Blend Skinning变形器 - 完全修复版"""
    
    def __init__(self, mesh: Mesh, skeleton: Skeleton, weights: np.ndarray):
        self.mesh = mesh
        self.skeleton = skeleton
        self.weights = weights
        
        self.bind_vertices = np.array(
            [[v.x, v.y, v.z] for v in mesh.vertices],
            dtype=np.float32
        )
        
        self.deformed_vertices = self.bind_vertices.copy()
        
        # 🔧 修复：正确计算绑定逆矩阵
        self._compute_bone_bind_inverse()
        
        print(f"[Deformer] 初始化: {len(self.bind_vertices)}个顶点, {skeleton.get_bone_count()}根骨骼")
        print(f"[Deformer] 权重矩阵形状: {weights.shape}")
    
    def _compute_bone_bind_inverse(self):
        """
        🔧 完全修复：正确计算绑定姿态逆矩阵
        
        关键修复：
        1. 使用骨骼的起始关节（start_joint）而不是结束关节
        2. 正确构建变换矩阵
        """
        num_bones = self.skeleton.get_bone_count()
        self.bone_bind_inverse = np.zeros((num_bones, 4, 4), dtype=np.float32)
        
        for bone_idx, bone in enumerate(self.skeleton.bones):
            # 🔧 修复：使用骨骼的起始关节位置
            # 绑定矩阵定义了骨骼在绑定姿态下的世界变换
            joint_pos = bone.start_joint.head
            
            # 构建绑定变换矩阵（从骨骼局部空间到世界空间）
            bind_mat = np.eye(4, dtype=np.float32)
            bind_mat[0, 3] = joint_pos.x
            bind_mat[1, 3] = joint_pos.y
            bind_mat[2, 3] = joint_pos.z
            
            # 计算逆矩阵（从世界空间到骨骼局部空间）
            self.bone_bind_inverse[bone_idx] = np.linalg.inv(bind_mat)
        
        print(f"[Deformer] 绑定逆矩阵已计算 ({num_bones} 根骨骼)")
    
    def update(self):
        """
        应用Linear Blend Skinning变形
        
        公式：v' = Σ(w_i * M_i * B_i^(-1) * v)
        其中：
        - v: 绑定姿态顶点
        - B_i^(-1): 骨骼i的绑定逆矩阵
        - M_i: 骨骼i的当前全局变换
        - w_i: 顶点对骨骼i的权重
        """
        N = self.bind_vertices.shape[0]
        
        # 转换为齐次坐标
        V_homo = np.hstack([
            self.bind_vertices,
            np.ones((N, 1), dtype=np.float32)
        ])
        
        # 获取所有骨骼的当前全局变换
        G_current = self._get_current_global_matrices()
        
        # LBS累加
        result = np.zeros((N, 4), dtype=np.float32)
        
        for bone_idx in range(len(self.skeleton.bones)):
            w = self.weights[:, bone_idx:bone_idx+1]  # (N, 1)
            
            if w.max() < 1e-6:
                continue
            
            # 🔧 修复：使用骨骼起始关节的变换
            bone = self.skeleton.bones[bone_idx]
            joint_idx = bone.start_joint.index
            G_bone = G_current[joint_idx]
            
            # 蒙皮矩阵 = 当前全局变换 × 绑定逆矩阵
            T = G_bone @ self.bone_bind_inverse[bone_idx]
            
            # 变换顶点并累加
            result += w * (V_homo @ T.T)
        
        self.deformed_vertices = result[:, :3]
    
    def _get_current_global_matrices(self) -> np.ndarray:
        """获取所有关节的当前全局变换矩阵"""
        num_joints = self.skeleton.get_joint_count()
        G = np.zeros((num_joints, 4, 4), dtype=np.float32)
        
        for i, joint in enumerate(self.skeleton.joints):
            G[i] = joint.global_transform.data.astype(np.float32)
        
        return G
    
    def get_deformed_vertices(self) -> List[Vector3]:
        """返回变形后的顶点列表"""
        return [Vector3(v[0], v[1], v[2]) for v in self.deformed_vertices]
    
    def get_vertices_array(self) -> np.ndarray:
        """返回变形后的顶点数组（副本）"""
        return self.deformed_vertices.copy()
    
    def get_vertices_for_rendering(self) -> np.ndarray:
        """返回用于渲染的顶点数组"""
        return self.deformed_vertices.astype(np.float32)