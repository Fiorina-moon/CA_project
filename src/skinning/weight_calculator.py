"""
权重计算器 v11 - 修复鹿角误绑前腿问题
"""
import numpy as np
from typing import List, Dict
from core.mesh import Mesh
from core.skeleton import Skeleton
from utils.math_utils import Vector3
from utils.geometry import point_to_segment_distance


class WeightCalculatorV11:
    """
    关键修复：
    1. 鹿角识别：增加排除前腿区域的逻辑
    2. 鹿角识别：扩大识别范围，使用更宽松的条件
    3. 前腿：修复肩部混合区域判断
    """
    
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
        """计算蒙皮权重"""
        num_vertices = mesh.get_vertex_count()
        num_bones = skeleton.get_bone_count()
        
        print(f"\n计算蒙皮权重 (v11 - 修复鹿角误绑)...")
        print(f"  顶点数: {num_vertices}")
        print(f"  骨骼数: {num_bones}")
        
        bone_regions = self._classify_bones(skeleton)
        key_bones = self._get_key_bones(skeleton, bone_regions)
        
        bbox_min, bbox_max = mesh.get_bounding_box()
        model_info = {
            'min': bbox_min,
            'max': bbox_max,
            'height': bbox_max.z - bbox_min.z,
            'width': bbox_max.x - bbox_min.x,
            'length': bbox_max.y - bbox_min.y,
        }
        
        print(f"\n  模型尺寸: 高(Z)={model_info['height']:.3f}, 宽(X)={model_info['width']:.3f}, 长(Y)={model_info['length']:.3f}")
        
        if key_bones['head_pos']:
            print(f"  头部位置: X={key_bones['head_pos'].x:.3f}, Y={key_bones['head_pos'].y:.3f}, Z={key_bones['head_pos'].z:.3f}")
        
        # 🔧 获取前腿骨骼位置（用于排除）
        fleg_positions = self._get_front_leg_positions(skeleton, bone_regions)
        
        weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
        stats = {'antler': 0, 'ankle': 0, 'shoulder': 0, 'normal': 0}
        
        for i, vertex in enumerate(mesh.vertices):
            if (i + 1) % 2000 == 0:
                print(f"  进度: {i + 1}/{num_vertices}")
            
            # 🔧 增强版鹿角判断
            if self._is_antler_enhanced(vertex, key_bones, model_info, fleg_positions):
                if key_bones['head_idx'] is not None:
                    weights[i, key_bones['head_idx']] = 1.0
                    stats['antler'] += 1
                    if stats['antler'] <= 15:
                        print(f"    [鹿角] 顶点{i}: X={vertex.x:.3f}, Y={vertex.y:.3f}, Z={vertex.z:.3f}")
                    continue
            
            # 脚踝
            ankle_result = self._check_ankle_strict(vertex, key_bones, model_info)
            if ankle_result is not None:
                weights[i, ankle_result] = 1.0
                stats['ankle'] += 1
                continue
            
            # 🔧 修复前腿肩部混合
            if self._is_shoulder_enhanced(vertex, key_bones, model_info):
                self._compute_shoulder_weight(i, vertex, weights, skeleton, bone_regions)
                stats['shoulder'] += 1
                continue
            
            # 普通顶点
            stats['normal'] += 1
            self._compute_normal_weight(i, vertex, weights, skeleton, bone_regions, key_bones)
        
        print(f"\n  顶点分类统计:")
        print(f"    🦌 鹿角顶点: {stats['antler']}")
        print(f"    🦶 脚踝顶点: {stats['ankle']}")
        print(f"    💪 肩部顶点: {stats['shoulder']}")
        print(f"    📍 普通顶点: {stats['normal']}")
        
        self._validate_weights(weights, skeleton)
        
        return weights
    
    def _get_front_leg_positions(self, skeleton: Skeleton, bone_regions: Dict) -> List:
        """获取前腿骨骼的位置范围（用于排除鹿角误判）"""
        fleg_positions = []
        
        for bone_idx, region in bone_regions.items():
            if region in ['front_leg_L', 'front_leg_R']:
                bone = skeleton.bones[bone_idx]
                fleg_positions.append({
                    'start': bone.start_joint.head,
                    'end': bone.end_joint.head,
                    'name': bone.name
                })
        
        return fleg_positions
    
    def _is_antler_enhanced(self, vertex: Vector3, key_bones: Dict, model_info: Dict, fleg_positions: List) -> bool:
        """
        🔧 增强版鹿角判断：
        1. 使用更宽松的Z高度条件
        2. 明确排除前腿区域
        3. 增加对称性检查
        """
        if key_bones['head_pos'] is None:
            return False
        
        head = key_bones['head_pos']
        
        # === 基础条件 ===
        dx = abs(vertex.x - head.x)
        dy = vertex.y - head.y
        dz = vertex.z - head.z
        
        # 1. Z高度：必须高于头部（鹿角在头上方）
        if dz < -0.02:  # 允许略低于头部2cm（考虑头部建模误差）
            return False
        
        # 2. 🔧 排除前腿区域（关键修复！）
        # 前腿在身体前方且较低，如果顶点接近前腿，则不是鹿角
        for fleg in fleg_positions:
            fleg_y = fleg['start'].y
            fleg_z = fleg['start'].z
            
            # 前腿通常在 Y < -0.1（身体前方）且 Z < 1.2（较低位置）
            if vertex.y < -0.05 and vertex.z < 1.3:  # 可能接近前腿
                dist_to_fleg = np.sqrt(
                    (vertex.x - fleg['start'].x)**2 +
                    (vertex.y - fleg['start'].y)**2 +
                    (vertex.z - fleg['start'].z)**2
                )
                if dist_to_fleg < 0.25:  # 距离前腿太近
                    return False
        
        # 3. 鹿角特征范围（放宽条件）
        antler_z_min = -0.02  # 允许略低于头部
        antler_z_max = 0.80   # 增大最大高度
        antler_y_min = -0.35  # 扩大后方范围
        antler_y_max = 0.12   # 扩大前方范围
        antler_x_max = 0.50   # 增大横向范围
        
        # 横向距离随高度增加（鹿角向外展开）
        if dz > 0:
            max_x_at_height = 0.10 + dz * 2.0  # 更陡峭的展开曲线
        else:
            max_x_at_height = 0.15  # 头部附近的基础宽度
        
        is_in_antler_box = (
            antler_z_min < dz < antler_z_max and
            antler_y_min < dy < antler_y_max and
            dx < min(antler_x_max, max_x_at_height)
        )
        
        # 4. 额外验证：检查是否明显偏向前腿方向
        if is_in_antler_box:
            # 如果顶点Y坐标远小于头部（说明在头部前方很远），且Z不够高，可能是前腿
            if dy < -0.15 and dz < 0.15:
                return False
        
        return is_in_antler_box
    
    def _is_shoulder_enhanced(self, vertex: Vector3, key_bones: Dict, model_info: Dict) -> bool:
        """
        🔧 增强版肩部判断
        """
        if key_bones['chest_pos'] is None:
            return False
        
        chest = key_bones['chest_pos']
        
        dx = abs(vertex.x - chest.x)
        dy = vertex.y - chest.y
        dz = vertex.z - chest.z
        
        # 🔧 修复：调整肩部识别范围
        is_shoulder = (
            -0.25 < dy < 0.25 and   # 扩大前后范围
            0.03 < dx < 0.35 and    # 扩大横向范围
            -0.15 < dz < 0.30       # 扩大高度范围
        )
        
        return is_shoulder
    
    def _check_ankle_strict(self, vertex: Vector3, key_bones: Dict, model_info: Dict) -> int:
        """脚踝检测"""
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
        """肩部混合权重"""
        shoulder_bones = []
        
        for bone_idx, region in bone_regions.items():
            if region in ['spine', 'front_leg_L', 'front_leg_R', 'neck']:  # 添加neck
                bone = skeleton.bones[bone_idx]
                dist = point_to_segment_distance(vertex, bone.start_joint.head, bone.end_joint.head)
                shoulder_bones.append((bone_idx, dist))
        
        if not shoulder_bones:
            return
        
        shoulder_bones.sort(key=lambda x: x[1])
        top_bones = shoulder_bones[:self.max_influences]
        
        total = 0.0
        bone_weights = []
        min_d = top_bones[0][1]
        
        for bone_idx, dist in top_bones:
            w = 1.0 / ((dist / (min_d + 0.001)) ** 1.5 + 0.01)
            bone_weights.append((bone_idx, w))
            total += w
        
        if total > self.epsilon:
            for bone_idx, w in bone_weights:
                weights[vertex_idx, bone_idx] = w / total
    
    def _compute_normal_weight(self, vertex_idx, vertex, weights, skeleton, bone_regions, key_bones):
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
            min_d = top_bones[0][1]
            for bone_idx, dist in top_bones:
                w = 1.0 / ((dist / (min_d + 0.001)) ** 2 + 0.01)
                bone_weights.append((bone_idx, w))
                total += w
        
        if total > self.epsilon:
            for bone_idx, w in bone_weights:
                weights[vertex_idx, bone_idx] = w / total
        elif top_bones:
            weights[vertex_idx, top_bones[0][0]] = 1.0
        else:
            weights[vertex_idx, nearest_bone] = 1.0
    
    def _classify_bones(self, skeleton: Skeleton) -> Dict[int, str]:
        """骨骼分类"""
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
        """获取关键骨骼"""
        key_bones = {
            'head_idx': None,
            'head_pos': None,
            'ankles': {},
            'chest_idx': None,
            'chest_pos': None
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
        """获取允许的骨骼"""
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
        """验证权重"""
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


# 兼容
WeightCalculator = WeightCalculatorV11