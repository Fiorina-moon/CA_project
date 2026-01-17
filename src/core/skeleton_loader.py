"""
骨架JSON加载器
"""
import json
from pathlib import Path
from core.skeleton import Skeleton, Joint
from utils.math_utils import Vector3


class SkeletonLoader:
    """骨架加载器"""
    
    @staticmethod
    def load(filepath: Path) -> Skeleton:
        """
        加载骨架JSON文件
        
        Args:
            filepath: JSON文件路径
        
        Returns:
            Skeleton对象
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        skeleton = Skeleton()
        
        # 加载关节
        for joint_data in data['joints']:
            head = Vector3.from_array(joint_data['head'])
            tail = Vector3.from_array(joint_data['tail'])
            
            # 🔧 重新启用坐标转换：(x, y, z) -> (x, z, -y)
            head_rotated = Vector3(head.x, head.z, -head.y)
            tail_rotated = Vector3(tail.x, tail.z, -tail.y)
            
            joint = Joint(
                name=joint_data['name'],
                index=joint_data['index'],
                head=head_rotated,
                tail=tail_rotated,
                parent=joint_data.get('parent')
            )
            skeleton.add_joint(joint)

        
        # 构建层级关系
        skeleton.build_hierarchy()
        
        # 构建骨骼列表
        skeleton.build_bones()
        
        print(f"✓ 加载骨架: {filepath.name}")
        print(f"  关节: {skeleton.get_joint_count()}")
        print(f"  骨骼: {skeleton.get_bone_count()}")
        print(f"  根节点: {skeleton.root_joint.name if skeleton.root_joint else 'None'}")
        
        return skeleton