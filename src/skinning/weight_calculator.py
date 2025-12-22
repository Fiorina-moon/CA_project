"""
蒙皮权重计算器 - 双线性插值法
"""
import numpy as np
from typing import List, Tuple
from core.mesh import Mesh
from core.skeleton import Skeleton, Bone
from utils.math_utils import Vector3
from utils.geometry import point_to_segment_distance


class WeightCalculator:
    """权重计算器"""
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: 极小值阈值，用于处理重合情况
        """
        self.epsilon = epsilon
    
    def compute_weights_bilinear(self, mesh: Mesh, skeleton: Skeleton) -> np.ndarray:
        """
        使用双线性插值计算蒙皮权重
        
        Args:
            mesh: 网格模型
            skeleton: 骨架
        
        Returns:
            权重矩阵 (N × M)，N=顶点数，M=骨骼数
        """
        num_vertices = mesh.get_vertex_count()
        num_bones = skeleton.get_bone_count()
        
        print(f"\n计算蒙皮权重...")
        print(f"  顶点数: {num_vertices}")
        print(f"  骨骼数: {num_bones}")
        
        # 初始化权重矩阵
        weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
        
        # 对每个顶点计算权重
        for i, vertex in enumerate(mesh.vertices):
            if (i + 1) % 1000 == 0:
                print(f"  进度: {i + 1}/{num_vertices}")
            
            # 计算到所有骨骼的距离
            distances = []
            for bone in skeleton.bones:
                dist = point_to_segment_distance(
                    vertex,
                    bone.get_start_position(),
                    bone.get_end_position()
                )
                distances.append(dist)
            
            # 找到距离最近的两个骨骼
            sorted_indices = np.argsort(distances)
            nearest_idx1 = sorted_indices[0]
            nearest_idx2 = sorted_indices[1]
            
            d1 = distances[nearest_idx1]
            d2 = distances[nearest_idx2]
            
            # 双线性插值
            total_dist = d1 + d2
            
            if total_dist < self.epsilon:
                # 特殊情况：顶点非常接近某个骨骼
                weights[i, nearest_idx1] = 1.0
            else:
                # 交叉分配：距离越近权重越大
                weights[i, nearest_idx1] = d2 / total_dist
                weights[i, nearest_idx2] = d1 / total_dist
        
        print(f"✓ 权重计算完成")
        
        # 验证权重
        self._validate_weights(weights)
        
        return weights
    
    def compute_weights_nearest(self, mesh: Mesh, skeleton: Skeleton) -> np.ndarray:
        """
        使用最近邻法计算权重（简单版本）
        
        Args:
            mesh: 网格模型
            skeleton: 骨架
        
        Returns:
            权重矩阵 (N × M)
        """
        num_vertices = mesh.get_vertex_count()
        num_bones = skeleton.get_bone_count()
        
        weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
        
        for i, vertex in enumerate(mesh.vertices):
            # 找到最近的骨骼
            min_dist = float('inf')
            nearest_bone_idx = 0
            
            for bone_idx, bone in enumerate(skeleton.bones):
                dist = point_to_segment_distance(
                    vertex,
                    bone.get_start_position(),
                    bone.get_end_position()
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_bone_idx = bone_idx
            
            # 最近骨骼权重为1
            weights[i, nearest_bone_idx] = 1.0
        
        return weights
    
    def _validate_weights(self, weights: np.ndarray):
        """验证权重矩阵"""
        # 检查权重和是否为1
        row_sums = weights.sum(axis=1)
        
        # 允许微小误差
        invalid_rows = np.abs(row_sums - 1.0) > 1e-4
        num_invalid = invalid_rows.sum()
        
        if num_invalid > 0:
            print(f"  ⚠ 警告: {num_invalid} 个顶点的权重和不为1")
            print(f"    最大偏差: {np.abs(row_sums - 1.0).max():.6f}")
        else:
            print(f"  ✓ 权重验证通过 (所有顶点权重和 = 1.0)")
        
        # 检查权重范围
        if weights.min() < 0 or weights.max() > 1:
            print(f"  ⚠ 警告: 权重超出[0,1]范围")
            print(f"    Min: {weights.min():.6f}, Max: {weights.max():.6f}")
        
        # 统计信息
        non_zero = (weights > 1e-6).sum()
        total = weights.size
        sparsity = 1.0 - (non_zero / total)
        
        print(f"  权重稀疏度: {sparsity*100:.2f}% ({non_zero}/{total} 非零)")