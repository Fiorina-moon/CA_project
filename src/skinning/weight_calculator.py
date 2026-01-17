"""
权重计算器 v15 - 限制前腿影响区域
核心思想：前腿骨骼只能影响身体下半部分和前方的顶点
"""
import numpy as np
from typing import List, Dict, Set
from core.mesh import Mesh
from core.skeleton import Skeleton
from utils.math_utils import Vector3
from utils.geometry import point_to_segment_distance


class WeightCalculatorV15:
    
    def __init__(self, max_influences: int = 4, epsilon: float = 1e-6):
        self.max_influences = max_influences
        self.epsilon = epsilon
        
        self.region_keywords = {
            'head': ['righead', 'rigjaw', 'rigtongue', 'rigeyelid', 'rigear'],
            'neck': ['rigneck'],
            'spine': ['rigroot', 'rigpelvis', 'rigspine', 'rigchest'],
            'tail': ['rigtail'],
            'back_leg_L': ['riglbleg'],
            'back_leg_R': ['rigrbleg'],
            'front_leg_L': ['riglfleg', 'riglflegcollarbone'],
            'front_leg_R': ['rigrfleg', 'rigrflegcollarbone'],
            'ankle_BL': ['riglblegankle'],
            'ankle_BR': ['rigrblegankle'],
            'ankle_FL': ['riglflegankle'],
            'ankle_FR': ['rigrflegankle'],
        }
   
    def compute_weights(self, mesh: Mesh, skeleton: Skeleton) -> np.ndarray:
        num_vertices = mesh.get_vertex_count()
        num_bones = skeleton.get_bone_count()
        
        print(f"\n计算蒙皮权重 (v15 - 限制前腿影响区域)...")
        print(f"  顶点数: {num_vertices}")
        print(f"  骨骼数: {num_bones}")
        
        bone_regions = self._classify_bones(skeleton)
        key_bones = self._get_key_bones(skeleton, bone_regions)
        head_bone_chain = self._build_head_bone_chain(skeleton, bone_regions)
        
        # 获取需要排除的骨骼（腿部+脊柱+尾巴）
        excluded_bones = self._get_excluded_bones(bone_regions)
        
        print(f"  头部骨骼链: {[skeleton.bones[i].name for i in head_bone_chain]}")
        print(f"  排除骨骼: {[skeleton.bones[i].name for i in excluded_bones]}")
        
        bbox_min, bbox_max = mesh.get_bounding_box()
        model_info = {
            'min': bbox_min,
            'max': bbox_max,
            'height': bbox_max.z - bbox_min.z,
            'width': bbox_max.x - bbox_min.x,
            'length': bbox_max.y - bbox_min.y,
        }
        
        # 🔧 计算头部区域边界（替代原来的前腿边界）
        head_bounds = self._compute_head_bounds(skeleton, bone_regions, model_info)
        print(f"  头部区域: Y > {head_bounds['min_y']:.3f}, Z > {head_bounds['min_z']:.3f}")
        
        weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
        stats = {'head_region': 0, 'ankle': 0, 'shoulder': 0, 'normal': 0}
        
        for i, vertex in enumerate(mesh.vertices):
            if (i + 1) % 2000 == 0:
                print(f"  进度: {i + 1}/{num_vertices}")
            
            # 🔧 判断该顶点是否在头部区域内
            in_head_region = self._is_in_head_region(vertex, head_bounds)
            
            # 脚踝检测
            ankle_result = self._check_ankle_strict(vertex, key_bones, model_info)
            if ankle_result is not None:
                weights[i, ankle_result] = 1.0
                stats['ankle'] += 1
                continue
            
            # 肩部检测
            if self._is_shoulder_enhanced(vertex, key_bones, model_info):
                self._compute_shoulder_weight(i, vertex, weights, skeleton, bone_regions)
                stats['shoulder'] += 1
                continue
            
            # 🔧 如果顶点在头部区域内，只使用头部和颈部骨骼
            if in_head_region:
                self._compute_weight_excluding_bones(
                    i, vertex, weights, skeleton, bone_regions, 
                    head_bone_chain, excluded_bones
                )
                stats['head_region'] += 1
            else:
                # 非头部区域：正常计算
                stats['normal'] += 1
                self._compute_normal_weight(i, vertex, weights, skeleton, bone_regions)
        
        print(f"\n  统计: 头部区域={stats['head_region']}, 脚踝={stats['ankle']}, 肩部={stats['shoulder']}, 正常={stats['normal']}")
        
        self._validate_weights(weights, skeleton)
        return weights


    def _compute_head_bounds(self, skeleton: Skeleton, bone_regions: Dict[int, str], 
                            model_info: Dict) -> Dict:
        """
        计算头部区域的边界
        只有在这个区域内的顶点才排除非头部骨骼
        """
        head_min_y = float('inf')
        head_max_y = float('-inf')
        head_min_z = float('inf')
        head_max_z = float('-inf')
        
        # 找到头部和颈部骨骼的位置范围
        for bone_idx, region in bone_regions.items():
            if region in ['head', 'neck']:
                bone = skeleton.bones[bone_idx]
                head_min_y = min(head_min_y, bone.start_joint.head.y, bone.end_joint.head.y)
                head_max_y = max(head_max_y, bone.start_joint.head.y, bone.end_joint.head.y)
                head_min_z = min(head_min_z, bone.start_joint.head.z, bone.end_joint.head.z)
                head_max_z = max(head_max_z, bone.start_joint.head.z, bone.end_joint.head.z)
        
        # 头部区域边界
        bounds = {
            # Y 方向：从颈部开始往前（头部方向）
            'min_y': head_min_y - model_info['length'] * 0.05,  # 稍微往后延伸一点
            # Z 方向：从颈部高度开始
            'min_z': head_min_z - model_info['height'] * 0.05,  # 颈部底部
        }
        
        return bounds


    def _is_in_head_region(self, vertex: Vector3, head_bounds: Dict) -> bool:
        """
        判断顶点是否在头部区域
        必须同时满足：
        1. Y 坐标足够靠前（在颈部前方）
        2. Z 坐标足够高（在颈部高度以上）
        """
        # 必须在颈部前方
        if vertex.y < head_bounds['min_y']:
            return False
        
        # 必须在颈部高度以上
        if vertex.z < head_bounds['min_z']:
            return False
        
        return True

    
    def _compute_front_leg_bounds(self, skeleton: Skeleton, front_leg_bones: Set[int], 
                                   model_info: Dict) -> Dict:
        """计算前腿骨骼的影响边界"""
        max_z = model_info['min'].z
        min_y = float('inf')
        max_y = float('-inf')
        
        for bone_idx in front_leg_bones:
            bone = skeleton.bones[bone_idx]
            max_z = max(max_z, bone.start_joint.head.z, bone.end_joint.head.z)
            min_y = min(min_y, bone.start_joint.head.y, bone.end_joint.head.y)
            max_y = max(max_y, bone.start_joint.head.y, bone.end_joint.head.y)
        
        bounds = {
            'max_z': max_z + model_info['height'] * 0.09,
            'min_y': min_y - model_info['length'] * 0.08,
            'max_y': max_y + model_info['length'] * 0.04,
        }
        
        return bounds
    
    def _is_outside_front_leg_zone(self, vertex: Vector3, front_leg_bounds: Dict) -> bool:
        """判断顶点是否在排除区域之外"""
        if vertex.z > front_leg_bounds['max_z']:
            return True
        if vertex.y > front_leg_bounds['max_y']:
            return True
        return False
    
    def _get_front_leg_bones_only(self, bone_regions: Dict[int, str]) -> Set[int]:
        """仅获取前腿骨骼（用于计算边界）"""
        front_leg_bones = set()
        for bone_idx, region in bone_regions.items():
            if region in ['front_leg_L', 'front_leg_R', 'ankle_FL', 'ankle_FR']:
                front_leg_bones.add(bone_idx)
        return front_leg_bones
    
    def _get_excluded_bones(self, bone_regions: Dict[int, str]) -> Set[int]:
        """
        获取在头部区域需要排除的骨骼
        排除：前腿、后腿、脊柱、尾巴
        保留：头、颈（这样鹿角才能绑定到头部）
        """
        excluded_bones = set()
        for bone_idx, region in bone_regions.items():
            if region in ['front_leg_L', 'front_leg_R', 'ankle_FL', 'ankle_FR',
                          'back_leg_L', 'back_leg_R', 'ankle_BL', 'ankle_BR',
                          'spine', 'tail']:
                excluded_bones.add(bone_idx)
        return excluded_bones
    
    def _compute_weight_excluding_bones(self, vertex_idx: int, vertex: Vector3,
                                         weights: np.ndarray, skeleton: Skeleton,
                                         bone_regions: Dict, head_bone_chain: Set[int],
                                         excluded_bones: Set[int]):
        """计算权重时排除指定骨骼"""
        distances = []
        
        for bone_idx, bone in enumerate(skeleton.bones):
            # 跳过排除的骨骼
            if bone_idx in excluded_bones:
                continue
            
            dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
            distances.append((bone_idx, dist))
        
        # 如果没有可用骨骼，fallback 到所有骨骼
        if not distances:
            for bone_idx, bone in enumerate(skeleton.bones):
                dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
                distances.append((bone_idx, dist))
        
        distances.sort(key=lambda x: x[1])
        top_bones = distances[:self.max_influences]
        
        if not top_bones:
            return
        
        total = 0.0
        bone_weights = []
        min_d = max(top_bones[0][1], 0.001)
        
        for bone_idx, dist in top_bones:
            w = 1.0 / ((dist / min_d) ** 2 + 0.01)
            bone_weights.append((bone_idx, w))
            total += w
        
        if total > self.epsilon:
            for bone_idx, w in bone_weights:
                weights[vertex_idx, bone_idx] = w / total
        else:
            weights[vertex_idx, top_bones[0][0]] = 1.0

    def _compute_normal_weight(self, vertex_idx, vertex, weights, skeleton, bone_regions):
        """普通顶点权重"""
        min_dist = float('inf')
        nearest_bone = 0
        
        for bone_idx, bone in enumerate(skeleton.bones):
            dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
            if dist < min_dist:
                min_dist = dist
                nearest_bone = bone_idx
        
        nearest_region = bone_regions[nearest_bone]
        allowed_bones = self._get_allowed_bones(nearest_region, bone_regions, nearest_bone)
        
        distances = []
        for bone_idx in allowed_bones:
            bone = skeleton.bones[bone_idx]
            dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
            distances.append((bone_idx, dist))
        
        distances.sort(key=lambda x: x[1])
        top_bones = distances[:self.max_influences]
        
        total = 0.0
        bone_weights = []
        
        if top_bones:
            min_d = max(top_bones[0][1], 0.001)
            for bone_idx, dist in top_bones:
                w = 1.0 / ((dist / min_d) ** 2 + 0.01)
                bone_weights.append((bone_idx, w))
                total += w
        
        if total > self.epsilon:
            for bone_idx, w in bone_weights:
                weights[vertex_idx, bone_idx] = w / total
        elif top_bones:
            weights[vertex_idx, top_bones[0][0]] = 1.0
        else:
            weights[vertex_idx, nearest_bone] = 1.0
    
    # ===== 辅助方法 =====
    
    def _build_head_bone_chain(self, skeleton: Skeleton, bone_regions: Dict[int, str]) -> Set[int]:
        head_chain = set()
        for bone_idx, region in bone_regions.items():
            if region in ['head', 'neck']:
                head_chain.add(bone_idx)
        return head_chain
    
    def _compute_head_bounds(self, skeleton: Skeleton, bone_regions: Dict[int, str], 
                            model_info: Dict) -> Dict:
        """
        计算头部区域的边界
        只有在这个区域内的顶点才排除非头部骨骼
        """
        head_min_y = float('inf')
        head_max_y = float('-inf')
        head_min_z = float('inf')
        head_max_z = float('-inf')
        
        # 找到头部和颈部骨骼的位置范围
        for bone_idx, region in bone_regions.items():
            if region in ['head', 'neck']:
                bone = skeleton.bones[bone_idx]
                head_min_y = min(head_min_y, bone.start_joint.head.y, bone.end_joint.head.y)
                head_max_y = max(head_max_y, bone.start_joint.head.y, bone.end_joint.head.y)
                head_min_z = min(head_min_z, bone.start_joint.head.z, bone.end_joint.head.z)
                head_max_z = max(head_max_z, bone.start_joint.head.z, bone.end_joint.head.z)
        
        # 头部区域边界
        bounds = {
            # Y 方向：从颈部开始往前（头部方向）
            'min_y': head_min_y - model_info['length'] * 0.05,  # 稍微往后延伸一点
            # Z 方向：从颈部高度开始
            'min_z': head_min_z - model_info['height'] * 0.05,  # 颈部底部
        }
        
        return bounds


    def _is_in_head_region(self, vertex: Vector3, head_bounds: Dict) -> bool:
        """
        判断顶点是否在头部区域
        必须同时满足：
        1. Y 坐标足够靠前（在颈部前方）
        2. Z 坐标足够高（在颈部高度以上）
        """
        # 必须在颈部前方
        if vertex.y < head_bounds['min_y']:
            return False
        
        # 必须在颈部高度以上
        if vertex.z < head_bounds['min_z']:
            return False
        
        return True

        
    def _is_shoulder_enhanced(self, vertex: Vector3, key_bones: Dict, model_info: Dict) -> bool:
        if key_bones['chest_pos'] is None:
            return False
        chest = key_bones['chest_pos']
        dx = abs(vertex.x - chest.x)
        dy = vertex.y - chest.y
        dz = vertex.z - chest.z
        return (-0.25 < dy < 0.25 and 0.03 < dx < 0.35 and -0.15 < dz < 0.30)
    
    def _check_ankle_strict(self, vertex: Vector3, key_bones: Dict, model_info: Dict) -> int:
        height = model_info['height']
        ankle_radius = height * 0.04
        closest_ankle = None
        closest_dist = float('inf')
        
        for region, (bone_idx, ankle_pos) in key_bones['ankles'].items():
            is_left = 'L' in region
            is_left_vertex = vertex.x > 0
            if is_left != is_left_vertex:
                continue
            
            dx = vertex.x - ankle_pos.x
            dy = vertex.y - ankle_pos.y
            dz = vertex.z - ankle_pos.z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if vertex.z < ankle_pos.z + height * 0.02 and dist < ankle_radius:
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ankle = bone_idx
        
        return closest_ankle
    
    def _compute_shoulder_weight(self, vertex_idx, vertex, weights, skeleton, bone_regions):
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
        
        total = 0.0
        bone_weights = []
        min_d = max(top_bones[0][1], 0.001)
        
        for bone_idx, dist in top_bones:
            w = 1.0 / ((dist / min_d) ** 1.5 + 0.01)
            bone_weights.append((bone_idx, w))
            total += w
        
        if total > self.epsilon:
            for bone_idx, w in bone_weights:
                weights[vertex_idx, bone_idx] = w / total
    
    def _classify_bones(self, skeleton: Skeleton) -> Dict[int, str]:
        bone_regions = {}
        for bone_idx, bone in enumerate(skeleton.bones):
            bone_name = bone.name.lower().replace('_', '').replace('-', '').replace('to', '')
            assigned = 'spine'
            
            priority = [
                'ankle_BL', 'ankle_BR', 'ankle_FL', 'ankle_FR',
                'head', 'neck', 'tail',
                'back_leg_L', 'back_leg_R', 'front_leg_L', 'front_leg_R',
                'spine'
            ]
            
            for region in priority:
                for keyword in self.region_keywords.get(region, []):
                    if keyword in bone_name:
                        assigned = region
                        break
                if assigned != 'spine':
                    break
            
            bone_regions[bone_idx] = assigned
        return bone_regions
    
    def _get_key_bones(self, skeleton: Skeleton, bone_regions: Dict[int, str]) -> Dict:
        key_bones = {
            'head_idx': None, 'head_pos': None,
            'ankles': {},
            'chest_idx': None, 'chest_pos': None
        }
        
        for bone_idx, region in bone_regions.items():
            bone = skeleton.bones[bone_idx]
            name_lower = bone.name.lower()
            
            if region == 'head' and key_bones['head_idx'] is None:
                if 'righead' in name_lower:
                    end_name = bone.end_joint.name.lower()
                    if 'righead' in end_name:
                        key_bones['head_idx'] = bone_idx
                        key_bones['head_pos'] = bone.end_joint.head
                        print(f"  ✓ 头部: [{bone_idx}] {bone.name}")
            
            if 'chest' in name_lower and key_bones['chest_idx'] is None:
                key_bones['chest_idx'] = bone_idx
                key_bones['chest_pos'] = bone.start_joint.head
            
            if region.startswith('ankle_'):
                key_bones['ankles'][region] = (bone_idx, bone.end_joint.head)
        
        return key_bones
    
    def _get_allowed_bones(self, region, bone_regions, nearest) -> List[int]:
        groups = {
            'head': {'head', 'neck'},
            'neck': {'head', 'neck', 'spine'},
            'spine': {'spine', 'neck'},
            'tail': {'tail', 'spine'},
            'front_leg_L': {'front_leg_L', 'spine'},
            'front_leg_R': {'front_leg_R', 'spine'},
            'back_leg_L': {'back_leg_L', 'spine'},
            'back_leg_R': {'back_leg_R', 'spine'},
            'ankle_BL': {'ankle_BL', 'back_leg_L'},
            'ankle_BR': {'ankle_BR', 'back_leg_R'},
            'ankle_FL': {'ankle_FL', 'front_leg_L'},
            'ankle_FR': {'ankle_FR', 'front_leg_R'},
        }
        allowed = groups.get(region, {region, 'spine'})
        bones = [idx for idx, r in bone_regions.items() if r in allowed]
        return bones if bones else [nearest]
    
    def _validate_weights(self, weights, skeleton):
        row_sums = weights.sum(axis=1)
        invalid = np.abs(row_sums - 1.0) > 1e-4
        if invalid.sum() > 0:
            for i in np.where(invalid)[0]:
                s = row_sums[i]
                if s > self.epsilon:
                    weights[i] /= s
                else:
                    weights[i, 0] = 1.0
        print(f"  ✓ 权重验证通过")


WeightCalculator = WeightCalculatorV15
