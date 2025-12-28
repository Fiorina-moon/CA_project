"""
权重计算器 v3 - 针对腿部穿出问题的激进修复
"""
import numpy as np
from typing import List, Tuple
from core.mesh import Mesh
from core.skeleton import Skeleton, Bone
from utils.math_utils import Vector3
from utils.geometry import point_to_segment_distance


class WeightCalculatorV3:
    """
    权重计算器 v3 - 激进修复版
    
    针对腿部顶点主要不受腿部骨骼控制的问题：
    1. 大幅增加影响半径（60%模型尺寸）
    2. 更激进的距离权重（近距离骨骼占主导）
    3. 增加最大影响骨骼数到8
    """
    
    def __init__(self, max_influences: int = 8, influence_radius_ratio: float = 0.6, epsilon: float = 1e-6):
        """
        Args:
            max_influences: 每个顶点最多受几个骨骼影响 (默认8)
            influence_radius_ratio: 影响半径占模型尺寸的比例 (默认0.6=60%)
            epsilon: 极小值阈值
        """
        self.max_influences = max_influences
        self.influence_radius_ratio = influence_radius_ratio
        self.epsilon = epsilon
    
    def compute_weights_bilinear(self, mesh: Mesh, skeleton: Skeleton) -> np.ndarray:
        """
        激进的权重计算
        
        关键改进：
        1. 影响半径增加到60%模型尺寸（vs之前的45%）
        2. 权重公式更激进：距离近的骨骼权重急剧增大
        3. 最大影响骨骼数增加到8（vs之前的6）
        
        Args:
            mesh: 网格模型
            skeleton: 骨架
        
        Returns:
            权重矩阵 (N × M)
        """
        num_vertices = mesh.get_vertex_count()
        num_bones = skeleton.get_bone_count()
        
        print(f"\n计算蒙皮权重 (激进版 v3)...")
        print(f"  顶点数: {num_vertices}")
        print(  f"  骨骼数: {num_bones}")
        print(f"  最大影响数: {self.max_influences}")
        
        # 计算影响半径
        bbox_min, bbox_max = mesh.get_bounding_box()
        model_size = max(
            bbox_max.x - bbox_min.x,
            bbox_max.y - bbox_min.y,
            bbox_max.z - bbox_min.z
        )
        influence_radius = model_size * self.influence_radius_ratio
        print(f"  影响半径: {influence_radius:.4f} ({self.influence_radius_ratio*100:.0f}%模型尺寸)")
        print(f"  ⚠️  使用激进权重公式（近距离骨骼占主导）")
        
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
                    bone.start_joint.head,
                    bone.end_joint.head
                )
                distances.append((bone_idx, dist))
            
            # 按距离排序
            distances.sort(key=lambda x: x[1])
            
            # 只保留影响半径内的骨骼
            valid_bones = []
            for bone_idx, dist in distances:
                if dist <= influence_radius:
                    valid_bones.append((bone_idx, dist))
                if len(valid_bones) >= self.max_influences:
                    break
            
            # 如果没有骨骼在影响半径内，强制使用最近的3个
            if len(valid_bones) == 0:
                valid_bones = distances[:3]
            elif len(valid_bones) < 2:
                # 至少要有2个骨骼
                valid_bones = distances[:2]
            
            # 计算权重 - 激进版本
            total_weight = 0.0
            bone_weights = []
            
            for bone_idx, dist in valid_bones:
                # 使用更激进的权重函数
                # 1. 距离立方反比（vs之前的平方反比）
                inv_weight = 1.0 / (dist ** 3 + 0.001)
                
                # 2. 更窄的高斯分布
                sigma = influence_radius / 4.0  # vs之前的/3.0
                exp_weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                
                # 3. 更侧重反比权重（近距离主导）
                weight = 0.3 * exp_weight + 0.7 * inv_weight  # vs之前的0.7/0.3
                
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
        self._validate_weights(weights, skeleton)
        
        return weights
    
    def _validate_weights(self, weights: np.ndarray, skeleton: Skeleton):
        """验证权重矩阵"""
        # 检查权重和是否为1
        row_sums = weights.sum(axis=1)
        
        invalid_rows = np.abs(row_sums - 1.0) > 1e-4
        num_invalid = invalid_rows.sum()
        
        if num_invalid > 0:
            print(f"  ⚠ 警告: {num_invalid} 个顶点的权重和不为1")
        else:
            print(f"  ✓ 权重验证通过")
        
        # 统计信息
        non_zero = (weights > 1e-6).sum()
        total = weights.size
        sparsity = 1.0 - (non_zero / total)
        
        print(f"  权重稀疏度: {sparsity*100:.2f}%")
        
        influences_per_vertex = (weights > 1e-6).sum(axis=1)
        print(f"  平均影响骨骼数: {influences_per_vertex.mean():.2f}")
        print(f"  最大影响骨骼数: {influences_per_vertex.max()}")
        
        # 检查腿部骨骼的主导顶点数
        print(f"\n  腿部骨骼主导顶点分析 (权重>0.3):")
        leg_bones = []
        for i, bone in enumerate(skeleton.bones):
            if 'leg' in bone.name.lower():
                leg_bones.append(i)
        
        if leg_bones:
            for bone_idx in leg_bones[:4]:
                bone = skeleton.bones[bone_idx]
                dominant = (weights[:, bone_idx] > 0.3).sum()
                max_weight = weights[:, bone_idx].max()
                print(f"    {bone.name[:35]:35s}: {dominant:4d}顶点 (max={max_weight:.3f})")


# 向后兼容
WeightCalculator = WeightCalculatorV3
WeightCalculatorImproved = WeightCalculatorV3