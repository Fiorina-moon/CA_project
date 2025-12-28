"""
测试新的deformer实现
"""
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import ELK_OBJ_PATH, SKELETON_JSON_PATH, WEIGHTS_DIR
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.utils.file_io import load_weights_npz


def test_deformer_basic():
    """测试基本的deformer功能"""
    print("="*60)
    print("测试新的Deformer实现")
    print("="*60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    weights = load_weights_npz(WEIGHTS_DIR / "elk_weights.npz")
    
    print(f"  模型: {mesh.get_vertex_count()} 顶点")
    print(f"  骨架: {skeleton.get_joint_count()} 关节, {skeleton.get_bone_count()} 骨骼")
    print(f"  权重: {weights.shape}")
    
    # 导入新的deformer
    print("\n[2/4] 创建Deformer...")
    from skinning.deformer import SkinDeformer
    
    deformer = SkinDeformer(mesh, skeleton, weights)
    
    # 测试初始状态（应该和绑定姿态一致）
    print("\n[3/4] 测试绑定姿态...")
    deformer.update()
    
    bind_verts = deformer.bind_vertices
    deformed_verts = deformer.get_vertices_array()
    
    diff = np.linalg.norm(deformed_verts - bind_verts, axis=1)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"  绑定姿态vs变形顶点差异:")
    print(f"    最大: {max_diff:.6f}")
    print(f"    平均: {mean_diff:.6f}")
    
    if max_diff < 0.01:
        print(f"  ✅ 绑定姿态保持正确")
    else:
        print(f"  ⚠️  差异过大，可能有问题")
    
    # 测试简单动画（旋转一个关节）
    print("\n[4/4] 测试简单动画...")
    
    # 找到尾巴关节
    tail_joint = None
    for joint in skeleton.joints:
        if 'Tail1' in joint.name:
            tail_joint = joint
            break
    
    if tail_joint:
        print(f"  旋转关节: {tail_joint.name}")
        
        # 保存原始变换
        original_transform = tail_joint.local_transform.data.copy()
        
        # 应用简单旋转（绕Z轴30度）
        from utils.math_utils import Matrix4
        angle = np.radians(30)
        rotation = Matrix4.rotation_z(angle)
        tail_joint.local_transform = rotation
        
        # 更新全局变换
        skeleton.update_global_transforms()
        
        # 更新变形
        deformer.update()
        
        # 检查变形
        new_verts = deformer.get_vertices_array()
        diff = np.linalg.norm(new_verts - bind_verts, axis=1)
        
        # 找到变化最大的顶点
        max_change_idx = diff.argmax()
        max_change = diff[max_change_idx]
        
        print(f"  最大顶点位移: {max_change:.4f}")
        print(f"  平均顶点位移: {diff.mean():.4f}")
        print(f"  移动的顶点数 (>0.01): {(diff > 0.01).sum()}")
        
        # 恢复原始变换
        tail_joint.local_transform.data = original_transform
        skeleton.update_global_transforms()
        
        if max_change > 0.001:
            print(f"  ✅ 动画变形正常")
        else:
            print(f"  ⚠️  变形幅度太小")
    else:
        print(f"  未找到尾巴关节，跳过动画测试")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


def test_bind_matrices():
    """测试绑定矩阵计算"""
    print("\n测试绑定矩阵...")
    
    from src.config import SKELETON_JSON_PATH
    from src.core.skeleton_loader import SkeletonLoader
    
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    print(f"\n检查前3个关节的绑定矩阵:")
    for i in range(min(3, len(skeleton.joints))):
        joint = skeleton.joints[i]
        
        if hasattr(joint, 'bind_matrix'):
            bind = joint.bind_matrix.data
            inv_bind = joint.inverse_bind_matrix.data
            
            # 验证：bind × inv_bind ≈ I
            identity = bind @ inv_bind
            is_identity = np.allclose(identity, np.eye(4), atol=1e-3)
            
            print(f"\n  [{i}] {joint.name}")
            print(f"    绑定位置: ({joint.head.x:.3f}, {joint.head.y:.3f}, {joint.head.z:.3f})")
            print(f"    bind矩阵平移: ({bind[0,3]:.3f}, {bind[1,3]:.3f}, {bind[2,3]:.3f})")
            print(f"    bind × inv ≈ I: {'✅' if is_identity else '❌'}")
        else:
            print(f"\n  [{i}] {joint.name}")
            print(f"    ❌ 缺少bind_matrix属性")


if __name__ == "__main__":
    # 先测试绑定矩阵
    test_bind_matrices()
    
    # 再测试deformer
    print("\n" + "="*60 + "\n")
    test_deformer_basic()