"""
蒙皮权重计算器
基于区域分割和解剖学约束的权重计算算法
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from src.core.mesh import Mesh
from src.core.skeleton import Skeleton
from src.utils.math_utils import Vector3
from src.utils.geometry import point_to_segment_distance
from .bone_classifier import BoneClassifier


class WeightCalculator:
    """蒙皮权重计算器"""
    
    def __init__(self, max_influences: int = 4, epsilon: float = 1e-6):
        """
        初始化权重计算器
        
        Args:
            max_influences: 每个顶点的最大影响骨骼数量
            epsilon: 数值精度阈值
        """
        self.max_influences = max_influences
        self.epsilon = epsilon
        self.classifier = BoneClassifier()
    
    def compute_weights(self, mesh: Mesh, skeleton: Skeleton) -> np.ndarray:
        """
        计算蒙皮权重
        
        Args:
            mesh: 网格模型
            skeleton: 骨架
        
        Returns:
            权重矩阵 (num_vertices, num_bones)
        """
        num_vertices = mesh.get_vertex_count()
        num_bones = skeleton.get_bone_count()
        
        print(f"\n计算蒙皮权重...")
        print(f"  顶点数: {num_vertices}")
        print(f"  骨骼数: {num_bones}")
        
        # 分类骨骼
        bone_regions = self.classifier.classify_bones(skeleton)
        
        # 识别关键骨骼
        key_bones = self.classifier.identify_key_bones(skeleton, bone_regions)
        
        # 计算模型边界信息
        bbox_min, bbox_max = mesh.get_bounding_box()
        model_info = {
            'min': bbox_min,
            'max': bbox_max,
            'height': bbox_max.z - bbox_min.z,
            'width': bbox_max.x - bbox_min.x,
            'length': bbox_max.y - bbox_min.y,
        }
        
        # 计算头部区域边界
        head_bounds = self._compute_head_bounds(skeleton, bone_regions, model_info)
        head_bone_chain = self.classifier.get_bones_by_regions(bone_regions, {'head', 'neck'})
        excluded_bones = self.classifier.get_bones_by_regions(
            bone_regions, 
            {'front_leg_L', 'front_leg_R', 'ankle_FL', 'ankle_FR',
             'back_leg_L', 'back_leg_R', 'ankle_BL', 'ankle_BR',
             'spine', 'tail'}
        )
        
        print(f"  头部骨骼链: {[skeleton.bones[i].name for i in head_bone_chain]}")
        print(f"  头部区域: Y > {head_bounds['min_y']:.3f}, Z > {head_bounds['min_z']:.3f}")
        
        # 初始化权重矩阵
        weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
        
        # 统计信息
        stats = {'head': 0, 'ankle': 0, 'shoulder': 0, 'normal': 0}
        
        # 逐顶点计算权重
        for i, vertex in enumerate(mesh.vertices):
            if (i + 1) % 2000 == 0:
                print(f"  进度: {i + 1}/{num_vertices}")
            
            # 1. 检查是否为脚踝顶点
            ankle_bone = self._check_ankle_region(vertex, key_bones, model_info)
            if ankle_bone is not None:
                weights[i, ankle_bone] = 1.0
                stats['ankle'] += 1
                continue
            
            # 2. 检查是否为肩部顶点
            if self._is_shoulder_region(vertex, key_bones, model_info):
                self._compute_shoulder_weights(i, vertex, weights, skeleton, bone_regions)
                stats['shoulder'] += 1
                continue
            
            # 3. 检查是否在头部区域
            if self._is_in_head_region(vertex, head_bounds):
                self._compute_weights_with_exclusion(
                    i, vertex, weights, skeleton, 
                    head_bone_chain, excluded_bones
                )
                stats['head'] += 1
            else:
                # 4. 普通区域
                self._compute_normal_weights(i, vertex, weights, skeleton, bone_regions)
                stats['normal'] += 1
        
        print(f"\n  统计: 头部={stats['head']}, 脚踝={stats['ankle']}, "
              f"肩部={stats['shoulder']}, 普通={stats['normal']}")
        
        # 验证和修正权重
        self._validate_weights(weights)
        
        return weights
    
    # ===== 区域边界计算 =====
    
    def _compute_head_bounds(self, skeleton: Skeleton, bone_regions: Dict[int, str],
                             model_info: Dict) -> Dict:
        """
        计算头部区域的边界
        
        Returns:
            {'min_y': float, 'min_z': float}
        """
        head_min_y = float('inf')
        head_min_z = float('inf')
        
        # 找到头部和颈部骨骼的位置范围
        for bone_idx, region in bone_regions.items():
            if region in ['head', 'neck']:
                bone = skeleton.bones[bone_idx]
                head_min_y = min(head_min_y, bone.start_joint.head.y, bone.end_joint.head.y)
                head_min_z = min(head_min_z, bone.start_joint.head.z, bone.end_joint.head.z)
        
        # 添加容差
        bounds = {
            'min_y': head_min_y - model_info['length'] * 0.05,
            'min_z': head_min_z - model_info['height'] * 0.05,
        }
        
        return bounds
    
    def _is_in_head_region(self, vertex: Vector3, head_bounds: Dict) -> bool:
        """
        判断顶点是否在头部区域
        
        条件：
        1. Y 坐标在颈部前方
        2. Z 坐标在颈部高度以上
        """
        if vertex.y < head_bounds['min_y']:
            return False
        if vertex.z < head_bounds['min_z']:
            return False
        return True
    
    # ===== 特殊区域检测 =====
    
    def _check_ankle_region(self, vertex: Vector3, key_bones: Dict,
                            model_info: Dict) -> Optional[int]:
        """
        检查顶点是否在脚踝区域
        
        Returns:
            脚踝骨骼索引，如果不在脚踝区域则返回 None
        """
        height = model_info['height']
        ankle_radius = height * 0.04  # 脚踝影响半径
        
        closest_ankle = None
        closest_dist = float('inf')
        
        for region, (bone_idx, ankle_pos) in key_bones['ankles'].items():
            # 左右侧匹配
            is_left_bone = 'L' in region
            is_left_vertex = vertex.x > 0
            if is_left_bone != is_left_vertex:
                continue
            
            # 计算距离
            dx = vertex.x - ankle_pos.x
            dy = vertex.y - ankle_pos.y
            dz = vertex.z - ankle_pos.z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # 高度约束（只影响脚踝以下）
            if vertex.z < ankle_pos.z + height * 0.02 and dist < ankle_radius:
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ankle = bone_idx
        
        return closest_ankle
    
    def _is_shoulder_region(self, vertex: Vector3, key_bones: Dict,
                            model_info: Dict) -> bool:
        """
        判断顶点是否在肩部区域
        
        肩部区域：胸部骨骼附近，左右两侧
        """
        if key_bones['chest_pos'] is None:
            return False
        
        chest = key_bones['chest_pos']
        dx = abs(vertex.x - chest.x)
        dy = vertex.y - chest.y
        dz = vertex.z - chest.z
        
        # 肩部区域边界
        return (-0.25 < dy < 0.25 and 
                0.03 < dx < 0.35 and 
                -0.15 < dz < 0.30)
    
    # ===== 权重计算 =====
    
    def _compute_weights_with_exclusion(self, vertex_idx: int, vertex: Vector3,
                                         weights: np.ndarray, skeleton: Skeleton,
                                         allowed_bones: Set[int],
                                         excluded_bones: Set[int]):
        """
        计算权重时排除指定骨骼
        
        Args:
            vertex_idx: 顶点索引
            vertex: 顶点位置
            weights: 权重矩阵
            skeleton: 骨架
            allowed_bones: 允许的骨骼集合（用于 fallback）
            excluded_bones: 排除的骨骼集合
        """
        # 计算到所有非排除骨骼的距离
        distances = []
        for bone_idx, bone in enumerate(skeleton.bones):
            if bone_idx in excluded_bones:
                continue
            dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
            distances.append((bone_idx, dist))
        
        # 如果没有可用骨骼，fallback 到所有骨骼
        if not distances:
            for bone_idx, bone in enumerate(skeleton.bones):
                dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
                distances.append((bone_idx, dist))
        
        # 选择最近的骨骼
        distances.sort(key=lambda x: x[1])
        top_bones = distances[:self.max_influences]
        
        # 分配权重
        self._assign_weights(vertex_idx, top_bones, weights)
    
    def _compute_shoulder_weights(self, vertex_idx: int, vertex: Vector3,
                                   weights: np.ndarray, skeleton: Skeleton,
                                   bone_regions: Dict[int, str]):
        """
        计算肩部区域的权重
        
        肩部允许躯干、前腿、颈部骨骼
        """
        shoulder_bones = []
        for bone_idx, region in bone_regions.items():
            if region in ['spine', 'front_leg_L', 'front_leg_R', 'neck']:
                bone = skeleton.bones[bone_idx]
                dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
                shoulder_bones.append((bone_idx, dist))
        
        if not shoulder_bones:
            return
        
        shoulder_bones.sort(key=lambda x: x[1])
        top_bones = shoulder_bones[:self.max_influences]
        
        # 分配权重（使用更柔和的衰减）
        self._assign_weights(vertex_idx, top_bones, weights, falloff=1.5)
    
    def _compute_normal_weights(self, vertex_idx: int, vertex: Vector3,
                                weights: np.ndarray, skeleton: Skeleton,
                                bone_regions: Dict[int, str]):
        """
        计算普通区域的权重
        
        基于最近骨骼的区域，只使用相邻区域的骨骼
        """
        # 找到最近的骨骼
        min_dist = float('inf')
        nearest_bone = 0
        
        for bone_idx, bone in enumerate(skeleton.bones):
            dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
            if dist < min_dist:
                min_dist = dist
                nearest_bone = bone_idx
        
        # 获取允许的骨骼
        nearest_region = bone_regions[nearest_bone]
        allowed_bones = self.classifier.get_allowed_bones(nearest_region, bone_regions, nearest_bone)
        
        # 计算距离
        distances = []
        for bone_idx in allowed_bones:
            bone = skeleton.bones[bone_idx]
            dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
            distances.append((bone_idx, dist))
        
        distances.sort(key=lambda x: x[1])
        top_bones = distances[:self.max_influences]
        
        # 分配权重
        self._assign_weights(vertex_idx, top_bones, weights)
    
    def _assign_weights(self, vertex_idx: int, bone_distances: List[Tuple[int, float]],
                        weights: np.ndarray, falloff: float = 2.0):
        """
        根据距离分配权重
        
        Args:
            vertex_idx: 顶点索引
            bone_distances: [(bone_idx, distance), ...]
            weights: 权重矩阵
            falloff: 距离衰减指数（越大衰减越快）
        """
        if not bone_distances:
            return
        
        # 计算权重
        total = 0.0
        bone_weights = []
        min_dist = max(bone_distances[0][1], 0.001)  # 避免除零
        
        for bone_idx, dist in bone_distances:
            w = 1.0 / ((dist / min_dist) ** falloff + 0.01)
            bone_weights.append((bone_idx, w))
            total += w
        
        # 归一化
        if total > self.epsilon:
            for bone_idx, w in bone_weights:
                weights[vertex_idx, bone_idx] = w / total
        else:
            weights[vertex_idx, bone_distances[0][0]] = 1.0
    
    # ===== 验证与修正 =====
    
    def _validate_weights(self, weights: np.ndarray):
        """
        验证并修正权重
        
        确保每个顶点的权重和为 1
        """
        row_sums = weights.sum(axis=1)
        invalid = np.abs(row_sums - 1.0) > 1e-4
        
        if invalid.sum() > 0:
            for i in np.where(invalid)[0]:
                s = row_sums[i]
                if s > self.epsilon:
                    weights[i] /= s
                else:
                    # 如果权重和为 0，分配给第一个骨骼
                    weights[i, 0] = 1.0
        
        print(f"  ✓ 权重验证通过")
