"""
蒙皮变形器 - 完整LBS实现（参考animation-skeleton项目）
"""
import numpy as np
from typing import List
from core.mesh import Mesh
from core.skeleton import Skeleton
from utils.math_utils import Vector3


class SkinDeformer:
    """Linear Blend Skinning变形器 - 向量化实现"""
    
    def __init__(self, mesh: Mesh, skeleton: Skeleton, weights: np.ndarray):
        """
        Args:
            mesh: 网格模型
            skeleton: 骨架
            weights: 权重矩阵 (N × B)，N=顶点数，B=骨骼数
        """
        self.mesh = mesh
        self.skeleton = skeleton
        self.weights = weights
        
        # 转换为numpy数组（绑定姿态）
        self.bind_vertices = np.array(
            [[v.x, v.y, v.z] for v in mesh.vertices],
            dtype=np.float32
        )  # (N, 3)
        
        # 变形后的顶点
        self.deformed_vertices = self.bind_vertices.copy()
        
        # 计算绑定姿态逆矩阵（只需计算一次）
        self._compute_bind_inverse()
        
        print(f"[Deformer] 初始化：{len(self.bind_vertices)}个顶点，{skeleton.get_bone_count()}根骨骼")
        print(f"[Deformer] 权重矩阵形状: {weights.shape}")
        print(f"[Deformer] 绑定逆矩阵已计算")
    
    def _compute_bind_inverse(self):
        """
        计算每个关节的绑定姿态逆矩阵
        
        关键：LBS需要 T = G_current × G_bind^(-1)
        其中 G_bind 是绑定姿态下的全局矩阵
        """
        num_joints = self.skeleton.get_joint_count()
        self.bind_inverse = np.zeros((num_joints, 4, 4), dtype=np.float32)
        
        for i, joint in enumerate(self.skeleton.joints):
            # 使用skeleton中已经计算好的bind_matrix
            if hasattr(joint, 'bind_matrix'):
                # 转换为numpy并求逆
                bind_mat = joint.bind_matrix.data.astype(np.float32)
                try:
                    self.bind_inverse[i] = np.linalg.inv(bind_mat)
                except np.linalg.LinAlgError:
                    print(f"警告: 关节 {joint.name} 的绑定矩阵不可逆，使用单位矩阵")
                    self.bind_inverse[i] = np.eye(4, dtype=np.float32)
            else:
                # 如果没有bind_matrix，手动构建
                # bind = T(head)
                bind_mat = np.eye(4, dtype=np.float32)
                bind_mat[:3, 3] = [joint.head.x, joint.head.y, joint.head.z]
                self.bind_inverse[i] = np.linalg.inv(bind_mat)
    
    def update(self):
        """
        应用Linear Blend Skinning变形
        
        LBS公式：
        v' = Σ(w_k × (G_current[k] × G_bind_inv[k]) × v)
        
        其中：
        - v: 绑定姿态顶点（齐次坐标）
        - G_current[k]: 骨骼k的当前全局变换矩阵
        - G_bind_inv[k]: 骨骼k的绑定姿态逆矩阵
        - w_k: 骨骼k对顶点的权重
        """
        N = self.bind_vertices.shape[0]
        
        # 1. 转换为齐次坐标 (N, 4)
        V_homogeneous = np.hstack([
            self.bind_vertices,
            np.ones((N, 1), dtype=np.float32)
        ])
        
        # 2. 获取当前全局矩阵
        G_current = self._get_current_global_matrices()
        
        # 3. LBS主循环 - 按骨骼累加
        result = np.zeros((N, 4), dtype=np.float32)
        
        for bone_idx, bone in enumerate(self.skeleton.bones):
            # 获取骨骼对应的关节索引（使用end_joint）
            joint_idx = bone.end_joint.index
            
            # 获取权重列 (N, 1)
            w = self.weights[:, bone_idx:bone_idx+1]
            
            # 蒙皮矩阵 = 当前全局 × 绑定逆
            T = G_current[joint_idx] @ self.bind_inverse[joint_idx]
            
            # 变换顶点并加权累加
            # (N,4) = (N,1) * ((N,4) @ (4,4).T)
            result += w * (V_homogeneous @ T.T)
        
        # 4. 提取xyz坐标
        self.deformed_vertices = result[:, :3]
    
    def _get_current_global_matrices(self) -> np.ndarray:
        """
        获取所有关节的当前全局变换矩阵
        
        Returns:
            (J, 4, 4) numpy数组
        """
        num_joints = self.skeleton.get_joint_count()
        G = np.zeros((num_joints, 4, 4), dtype=np.float32)
        
        for i, joint in enumerate(self.skeleton.joints):
            # 从Matrix4转换为numpy
            G[i] = joint.global_transform.data.astype(np.float32)
        
        return G
    
    def get_deformed_vertices(self) -> List[Vector3]:
        """返回Vector3列表（兼容旧接口）"""
        return [Vector3(v[0], v[1], v[2]) for v in self.deformed_vertices]
    
    def get_vertices_array(self) -> np.ndarray:
        """返回numpy数组 (N, 3)"""
        return self.deformed_vertices.copy()
    
    def get_vertices_for_rendering(self) -> np.ndarray:
        """
        返回适合OpenGL渲染的顶点数组
        Returns: (N, 3) float32
        """
        return self.deformed_vertices.astype(np.float32)


class SkinDeformerDebug(SkinDeformer):
    """带调试信息的变形器"""
    
    def update(self):
        """带调试输出的update"""
        super().update()
        
        # 检查变形是否合理
        bind_center = self.bind_vertices.mean(axis=0)
        deformed_center = self.deformed_vertices.mean(axis=0)
        
        bind_size = np.linalg.norm(self.bind_vertices.max(axis=0) - self.bind_vertices.min(axis=0))
        deformed_size = np.linalg.norm(self.deformed_vertices.max(axis=0) - self.deformed_vertices.min(axis=0))
        
        displacement = np.linalg.norm(deformed_center - bind_center)
        scale_change = deformed_size / (bind_size + 1e-8)
        
        if displacement > bind_size * 0.5:
            print(f"⚠ 警告: 模型中心移动过大 ({displacement:.4f})")
        
        if scale_change < 0.5 or scale_change > 2.0:
            print(f"⚠ 警告: 模型尺寸变化异常 (×{scale_change:.2f})")
        
        # 检查是否有NaN
        if np.isnan(self.deformed_vertices).any():
            print(f"❌ 错误: 变形顶点包含NaN!")