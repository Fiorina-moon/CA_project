"""
蒙皮权重计算器 - 改进版
"""
import numpy as np
from typing import List, Tuple
from core.mesh import Mesh
from core.skeleton import Skeleton, Bone
from utils.math_utils import Vector3
from utils.geometry import point_to_segment_distance


class WeightCalculator:
    """权重计算器 - 改进版"""
    
    def __init__(self, max_influences: int = 4, influence_radius: float = None, epsilon: float = 1e-6):
        """
        Args:
            max_influences: 每个顶点最多受几个骨骼影响 (推荐4-6)
            influence_radius: 影响半径，超过这个距离的骨骼不影响顶点 (None=自动计算)
            epsilon: 极小值阈值
        """
        self.max_influences = max_influences
        self.influence_radius = influence_radius
        self.epsilon = epsilon
    
    def compute_weights_bilinear(self, mesh: Mesh, skeleton: Skeleton) -> np.ndarray:
        """
        使用改进的距离加权法计算蒙皮权重
        
        改进：
        1. 支持4-6个骨骼影响（而非2个）
        2. 添加影响半径限制
        3. 使用指数衰减权重
        
        Args:
            mesh: 网格模型
            skeleton: 骨架
        
        Returns:
            权重矩阵 (N × M)，N=顶点数，M=骨骼数
        """
        num_vertices = mesh.get_vertex_count()
        num_bones = skeleton.get_bone_count()
        
        print(f"\n计算蒙皮权重 (改进版)...")
        print(f"  顶点数: {num_vertices}")
        print(f"  骨骼数: {num_bones}")
        print(f"  最大影响数: {self.max_influences}")
        
        # 自动计算影响半径
        if self.influence_radius is None:
            # 根据模型大小自动设置
            bbox_min, bbox_max = mesh.get_bounding_box()
            model_size = max(
                bbox_max.x - bbox_min.x,
                bbox_max.y - bbox_min.y,
                bbox_max.z - bbox_min.z
            )
            self.influence_radius = model_size * 0.3  # 30%的模型尺寸
            print(f"  自动影响半径: {self.influence_radius:.4f}")
        else:
            print(f"  影响半径: {self.influence_radius:.4f}")
        
        # 初始化权重矩阵
        weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
        
        # 对每个顶点计算权重
        for i, vertex in enumerate(mesh.vertices):
            if (i + 1) % 1000 == 0:
                print(f"  进度: {i + 1}/{num_vertices}")
            
            # 计算到所有骨骼的距离
            distances = []
            for bone_idx, bone in enumerate(skeleton.bones):
                dist = point_to_segment_distance(
                    vertex,
                    bone.get_start_position(),
                    bone.get_end_position()
                )
                distances.append((bone_idx, dist))
            
            # 按距离排序
            distances.sort(key=lambda x: x[1])
            
            # 只保留影响半径内的骨骼
            valid_bones = []
            for bone_idx, dist in distances:
                if dist <= self.influence_radius:
                    valid_bones.append((bone_idx, dist))
                if len(valid_bones) >= self.max_influences:
                    break
            
            # 如果没有骨骼在影响半径内，使用最近的1个
            if len(valid_bones) == 0:
                valid_bones = [distances[0]]
            
            # 计算权重（使用指数衰减）
            total_weight = 0.0
            bone_weights = []
            
            for bone_idx, dist in valid_bones:
                # 权重 = exp(-dist^2 / (2 * sigma^2))
                # sigma = influence_radius / 3 (3-sigma rule)
                sigma = self.influence_radius / 3.0
                weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                bone_weights.append((bone_idx, weight))
                total_weight += weight
            
            # 归一化权重
            if total_weight > self.epsilon:
                for bone_idx, weight in bone_weights:
                    weights[i, bone_idx] = weight / total_weight
            else:
                # 特殊情况：顶点完全重合
                weights[i, valid_bones[0][0]] = 1.0
        
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
        
        # 统计每个顶点的影响骨骼数
        influences_per_vertex = (weights > 1e-6).sum(axis=1)
        print(f"  平均影响骨骼数: {influences_per_vertex.mean():.2f}")
        print(f"  最大影响骨骼数: {influences_per_vertex.max()}")