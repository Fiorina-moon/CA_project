"""
蒙皮变形器
实现 Linear Blend Skinning (LBS) 算法
"""
import numpy as np
from typing import List
from src.core.mesh import Mesh
from src.core.skeleton import Skeleton
from src.utils.math_utils import Vector3


class SkinDeformer:
    """Linear Blend Skinning 变形器"""
    
    def __init__(self, mesh: Mesh, skeleton: Skeleton, weights: np.ndarray):
        """
        初始化蒙皮变形器
        
        Args:
            mesh: 网格模型
            skeleton: 骨架
            weights: 蒙皮权重矩阵 (num_vertices, num_bones)
        """
        self.mesh = mesh
        self.skeleton = skeleton
        self.weights = weights
        
        # 保存绑定姿态顶点
        self.bind_vertices = np.array(
            [[v.x, v.y, v.z] for v in mesh.vertices],
            dtype=np.float32
        )
        
        # 变形后的顶点（初始为绑定姿态）
        self.deformed_vertices = self.bind_vertices.copy()
        
        # 计算绑定姿态逆矩阵
        self.bone_bind_inverse = self._compute_bind_inverse_matrices()
        
        print(f"[Deformer] 初始化完成")
        print(f"  顶点数: {len(self.bind_vertices)}")
        print(f"  骨骼数: {skeleton.get_bone_count()}")
        print(f"  权重矩阵形状: {weights.shape}")
    
    def _compute_bind_inverse_matrices(self) -> np.ndarray:
        """
        计算每根骨骼的绑定姿态逆矩阵
        
        绑定矩阵定义：将顶点从世界空间变换到骨骼局部空间
        逆矩阵用于将绑定姿态顶点转换到骨骼局部坐标系
        
        Returns:
            形状为 (num_bones, 4, 4) 的逆矩阵数组
        """
        num_bones = self.skeleton.get_bone_count()
        bind_inverse = np.zeros((num_bones, 4, 4), dtype=np.float32)
        
        for bone_idx, bone in enumerate(self.skeleton.bones):
            # 使用骨骼起始关节的位置作为绑定位置
            joint_pos = bone.start_joint.head
            
            # 构建绑定变换矩阵（平移矩阵）
            # 从骨骼局部空间（原点在关节处）到世界空间
            bind_matrix = np.eye(4, dtype=np.float32)
            bind_matrix[0, 3] = joint_pos.x
            bind_matrix[1, 3] = joint_pos.y
            bind_matrix[2, 3] = joint_pos.z
            
            # 计算逆矩阵（从世界空间到骨骼局部空间）
            bind_inverse[bone_idx] = np.linalg.inv(bind_matrix)
        
        return bind_inverse
    
    def update(self):
        """
        应用 Linear Blend Skinning 变形
        
        公式：v' = Σ(w_i × M_i × B_i^(-1) × v)
        
        其中：
        - v: 绑定姿态顶点（世界空间）
        - B_i^(-1): 骨骼 i 的绑定逆矩阵
        - M_i: 骨骼 i 的当前全局变换矩阵
        - w_i: 顶点对骨骼 i 的权重
        - v': 变形后的顶点
        """
        num_vertices = self.bind_vertices.shape[0]
        
        # 转换为齐次坐标 (N, 4)
        vertices_homo = np.hstack([
            self.bind_vertices,
            np.ones((num_vertices, 1), dtype=np.float32)
        ])
        
        # 获取所有关节的当前全局变换矩阵
        global_transforms = self._get_global_transforms()
        
        # 初始化结果（齐次坐标）
        result = np.zeros((num_vertices, 4), dtype=np.float32)
        
        # 对每根骨骼进行加权变换
        for bone_idx, bone in enumerate(self.skeleton.bones):
            # 获取该骨骼的权重列 (N, 1)
            bone_weights = self.weights[:, bone_idx:bone_idx+1]
            
            # 跳过权重为 0 的骨骼（优化性能）
            if bone_weights.max() < 1e-6:
                continue
            
            # 获取骨骼起始关节的全局变换
            joint_idx = bone.start_joint.index
            global_transform = global_transforms[joint_idx]
            
            # 计算蒙皮矩阵：当前变换 × 绑定逆矩阵
            skinning_matrix = global_transform @ self.bone_bind_inverse[bone_idx]
            
            # 应用变换并加权累加
            # bone_weights * (vertices_homo @ skinning_matrix.T)
            result += bone_weights * (vertices_homo @ skinning_matrix.T)
        
        # 提取 3D 坐标（丢弃齐次坐标的 w 分量）
        self.deformed_vertices = result[:, :3]
    
    def _get_global_transforms(self) -> np.ndarray:
        """
        获取所有关节的当前全局变换矩阵
        
        Returns:
            形状为 (num_joints, 4, 4) 的变换矩阵数组
        """
        num_joints = self.skeleton.get_joint_count()
        transforms = np.zeros((num_joints, 4, 4), dtype=np.float32)
        
        for i, joint in enumerate(self.skeleton.joints):
            transforms[i] = joint.global_transform.data.astype(np.float32)
        
        return transforms
    
    # ===== 数据访问接口 =====
    
    def get_deformed_vertices(self) -> List[Vector3]:
        """
        返回变形后的顶点列表
        
        Returns:
            Vector3 对象列表
        """
        return [Vector3(v[0], v[1], v[2]) for v in self.deformed_vertices]
    
    def get_vertices_array(self) -> np.ndarray:
        """
        返回变形后的顶点数组（副本）
        
        Returns:
            形状为 (num_vertices, 3) 的 NumPy 数组
        """
        return self.deformed_vertices.copy()
    
    def get_vertices_for_rendering(self) -> np.ndarray:
        """
        返回用于渲染的顶点数组
        
        Returns:
            形状为 (num_vertices, 3) 的 float32 数组
        """
        return self.deformed_vertices.astype(np.float32)
    
    def get_bind_vertices(self) -> np.ndarray:
        """
        返回绑定姿态顶点数组（副本）
        
        Returns:
            形状为 (num_vertices, 3) 的 NumPy 数组
        """
        return self.bind_vertices.copy()
    