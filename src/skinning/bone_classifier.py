"""
骨骼分类器
负责骨骼区域划分和关键骨骼识别
"""
from typing import Dict, Set, List
from src.core.skeleton import Skeleton


class BoneClassifier:
    """骨骼分类器"""
    
    # 骨骼区域关键字映射
    REGION_KEYWORDS = {
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
    
    # 区域允许的相邻骨骼
    REGION_GROUPS = {
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
    
    def classify_bones(self, skeleton: Skeleton) -> Dict[int, str]:
        """
        根据名称将骨骼分类到不同区域
        
        Args:
            skeleton: 骨架
        
        Returns:
            骨骼索引 -> 区域名称的映射
        """
        bone_regions = {}
        
        # 优先级顺序（脚踝 > 头部 > 颈部 > 尾巴 > 腿部 > 躯干）
        priority = [
            'ankle_BL', 'ankle_BR', 'ankle_FL', 'ankle_FR',
            'head', 'neck', 'tail',
            'back_leg_L', 'back_leg_R', 'front_leg_L', 'front_leg_R',
            'spine'
        ]
        
        for bone_idx, bone in enumerate(skeleton.bones):
            # 标准化骨骼名称
            normalized_name = bone.name.lower().replace('_', '').replace('-', '').replace('to', '')
            
            # 默认分类为躯干
            assigned_region = 'spine'
            
            # 按优先级匹配
            for region in priority:
                keywords = self.REGION_KEYWORDS.get(region, [])
                if any(keyword in normalized_name for keyword in keywords):
                    assigned_region = region
                    break
            
            bone_regions[bone_idx] = assigned_region
        
        return bone_regions
    
    def identify_key_bones(self, skeleton: Skeleton, bone_regions: Dict[int, str]) -> Dict:
        """
        识别关键骨骼（头部、胸部、脚踝）
        
        Args:
            skeleton: 骨架
            bone_regions: 骨骼区域映射
        
        Returns:
            关键骨骼信息字典 {
                'head_idx': int,
                'head_pos': Vector3,
                'chest_idx': int,
                'chest_pos': Vector3,
                'ankles': {region: (bone_idx, position)}
            }
        """
        key_bones = {
            'head_idx': None,
            'head_pos': None,
            'chest_idx': None,
            'chest_pos': None,
            'ankles': {},  # region -> (bone_idx, position)
        }
        
        for bone_idx, region in bone_regions.items():
            bone = skeleton.bones[bone_idx]
            name_lower = bone.name.lower()
            
            # 识别头部骨骼
            if region == 'head' and key_bones['head_idx'] is None:
                if 'righead' in name_lower:
                    end_name = bone.end_joint.name.lower()
                    if 'righead' in end_name:
                        key_bones['head_idx'] = bone_idx
                        key_bones['head_pos'] = bone.end_joint.head
                        print(f"  ✓ 头部骨骼: [{bone_idx}] {bone.name}")
            
            # 识别胸部骨骼
            if 'chest' in name_lower and key_bones['chest_idx'] is None:
                key_bones['chest_idx'] = bone_idx
                key_bones['chest_pos'] = bone.start_joint.head
            
            # 识别脚踝骨骼
            if region.startswith('ankle_'):
                key_bones['ankles'][region] = (bone_idx, bone.end_joint.head)
        
        return key_bones
    
    def get_bones_by_regions(self, bone_regions: Dict[int, str], 
                             target_regions: Set[str]) -> Set[int]:
        """
        获取指定区域的所有骨骼索引
        
        Args:
            bone_regions: 骨骼区域映射
            target_regions: 目标区域集合
        
        Returns:
            骨骼索引集合
        """
        return {idx for idx, region in bone_regions.items() if region in target_regions}
    
    def get_allowed_bones(self, region: str, bone_regions: Dict[int, str],
                          nearest_bone: int) -> List[int]:
        """
        获取区域允许的骨骼
        
        Args:
            region: 区域名称
            bone_regions: 骨骼区域映射
            nearest_bone: 最近骨骼索引（fallback）
        
        Returns:
            允许的骨骼索引列表
        """
        allowed_regions = self.REGION_GROUPS.get(region, {region, 'spine'})
        bones = [idx for idx, r in bone_regions.items() if r in allowed_regions]
        return bones if bones else [nearest_bone]
