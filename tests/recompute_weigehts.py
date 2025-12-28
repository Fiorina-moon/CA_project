"""
重新计算权重
使用改进的算法重新生成elk_weights.npz
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import ELK_OBJ_PATH, SKELETON_JSON_PATH, WEIGHTS_DIR
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.skinning.weight_calculator import WeightCalculator
from src.utils.file_io import save_weights_npz
import numpy as np


def recompute_weights(max_influences=4, influence_radius=None):
    """
    重新计算权重
    
    Args:
        max_influences: 每个顶点最多受几个骨骼影响 (推荐4)
        influence_radius: 影响半径 (None=自动)
    """
    print("="*60)
    print("重新计算蒙皮权重")
    print("="*60)
    
    # 加载数据
    print("\n[1/4] 加载模型和骨架...")
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    print(f"  模型: {mesh.get_vertex_count()} 顶点")
    print(f"  骨架: {skeleton.get_bone_count()} 骨骼")
    
    # 计算权重
    print("\n[2/4] 计算权重...")
    calculator = WeightCalculator(
        max_influences=max_influences,
        influence_radius=influence_radius
    )
    weights = calculator.compute_weights_bilinear(mesh, skeleton)
    
    # 备份旧文件
    print("\n[3/4] 备份旧权重文件...")
    old_weights_path = WEIGHTS_DIR / "elk_weights.npz"
    if old_weights_path.exists():
        backup_path = WEIGHTS_DIR / "elk_weights_old.npz"
        import shutil
        shutil.copy(old_weights_path, backup_path)
        print(f"  ✓ 已备份到: {backup_path}")
    
    # 保存新权重
    print("\n[4/4] 保存新权重...")
    save_weights_npz(weights, old_weights_path)
    print(f"  ✓ 已保存到: {old_weights_path}")
    
    # 分析权重分布
    print("\n" + "="*60)
    print("权重分析")
    print("="*60)
    
    # 检查每个骨骼影响了多少顶点
    print("\n骨骼影响顶点数统计:")
    for bone_idx, bone in enumerate(skeleton.bones):
        affected_vertices = (weights[:, bone_idx] > 1e-6).sum()
        if affected_vertices > 0:
            avg_weight = weights[weights[:, bone_idx] > 1e-6, bone_idx].mean()
            print(f"  [{bone_idx:2d}] {bone.name[:30]:30s} → {affected_vertices:4d} 顶点 (平均权重: {avg_weight:.3f})")
    
    # 检查关键部位
    print("\n关键部位检查:")
    
    # 查找尾巴骨骼索引
    tail_bone_idx = None
    for idx, bone in enumerate(skeleton.bones):
        if 'Tail1' in bone.end_joint.name:
            tail_bone_idx = idx
            break
    
    if tail_bone_idx is not None:
        tail_vertices = np.where(weights[:, tail_bone_idx] > 0.1)[0]
        print(f"  尾巴骨骼 ({skeleton.bones[tail_bone_idx].name})")
        print(f"    影响顶点数: {len(tail_vertices)}")
        if len(tail_vertices) > 0:
            print(f"    前5个顶点索引: {tail_vertices[:5]}")
    
    # 查找脖子骨骼索引
    neck_bone_idx = None
    for idx, bone in enumerate(skeleton.bones):
        if 'Neck1' in bone.end_joint.name:
            neck_bone_idx = idx
            break
    
    if neck_bone_idx is not None:
        neck_vertices = np.where(weights[:, neck_bone_idx] > 0.1)[0]
        print(f"  脖子骨骼 ({skeleton.bones[neck_bone_idx].name})")
        print(f"    影响顶点数: {len(neck_vertices)}")
        if len(neck_vertices) > 0:
            print(f"    前5个顶点索引: {neck_vertices[:5]}")
    
    # 检查是否有顶点同时受尾巴和脖子影响
    if tail_bone_idx is not None and neck_bone_idx is not None:
        both = np.where((weights[:, tail_bone_idx] > 0.01) & (weights[:, neck_bone_idx] > 0.01))[0]
        if len(both) > 0:
            print(f"  ⚠ 警告: {len(both)} 个顶点同时受尾巴和脖子影响 (不应该！)")
            print(f"    这些顶点: {both[:10]}")
        else:
            print(f"  ✓ 没有顶点同时受尾巴和脖子影响")
    
    print("\n" + "="*60)
    print("✓ 权重重新计算完成！")
    print("="*60)
    print("\n现在可以运行测试:")
    print("  cd tests")
    print("  python quick_test.py tail_wag")


if __name__ == "__main__":
    print("\n推荐设置:")
    print("  max_influences=4  (每个顶点最多4个骨骼)")
    print("  influence_radius=自动 (模型尺寸的30%)")
    print()
    
    choice = input("使用推荐设置? (y/n): ").lower()
    
    if choice == 'y' or choice == '':
        recompute_weights(max_influences=4)
    else:
        try:
            max_inf = int(input("max_influences (推荐4): "))
            recompute_weights(max_influences=max_inf)
        except:
            print("使用默认值")
            recompute_weights(max_influences=4)